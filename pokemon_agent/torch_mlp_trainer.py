from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .mlp_ranker import MODEL_TYPE, SCHEMA_VERSION, build_feature_stats


def _iter_records(labels_path: str | Path, max_examples: int | None = None) -> Iterable[dict[str, Any]]:
    seen = 0
    with Path(labels_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            candidates = record.get("candidates") or []
            label_index = int(record.get("label_index", -1))
            if label_index < 0 or label_index >= len(candidates):
                continue
            if not bool(record.get("matched", label_index >= 0)):
                continue
            yield record
            seen += 1
            if max_examples is not None and seen >= max_examples:
                break


def _clean_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not number or not math.isfinite(number):
        return None
    return number


def _batch_records(
    records: list[dict[str, Any]],
    *,
    feature_index: dict[str, int],
    input_scale: list[float],
    input_clip: float,
    device: Any,
) -> tuple[Any, Any, Any]:
    import torch

    batch_size = len(records)
    max_candidates = max(len(record.get("candidates") or []) for record in records)
    input_dim = len(feature_index)
    inputs = torch.zeros((batch_size, max_candidates, input_dim), dtype=torch.float32, device=device)
    labels = torch.zeros((batch_size,), dtype=torch.long, device=device)
    mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool, device=device)

    for row_index, record in enumerate(records):
        labels[row_index] = int(record["label_index"])
        candidates = record.get("candidates") or []
        mask[row_index, : len(candidates)] = True
        for cand_index, candidate in enumerate(candidates):
            features = candidate.get("features") if isinstance(candidate, dict) else {}
            if not isinstance(features, dict):
                continue
            for key, value in features.items():
                input_index = feature_index.get(str(key))
                if input_index is None:
                    continue
                number = _clean_number(value)
                if number is None:
                    continue
                scaled = number / input_scale[input_index]
                clipped = max(-input_clip, min(input_clip, scaled)) if input_clip > 0 else scaled
                if clipped:
                    inputs[row_index, cand_index, input_index] = clipped
    return inputs, labels, mask


def _save_torch_checkpoint(
    path: str | Path,
    *,
    model: Any,
    feature_names: list[str],
    input_scale: list[float],
    input_clip: float,
    summary: dict[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    w1 = model.hidden.weight.detach().cpu().tolist()
    b1 = model.hidden.bias.detach().cpu().tolist()
    w2 = model.output.weight.detach().cpu().squeeze(0).tolist()
    b2 = float(model.output.bias.detach().cpu().item())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "feature_names": feature_names,
        "input_mean": [0.0 for _ in feature_names],
        "input_scale": input_scale,
        "hidden_dim": int(model.hidden.out_features),
        "activation": "relu",
        "normalization": "sparse",
        "input_clip": float(input_clip),
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "summary": summary,
    }
    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def train_torch_mlp_ranker(
    labels_path: str | Path,
    *,
    out_dir: str = "artifacts/checkpoints/torch_mlp_ranker",
    epochs: int = 5,
    lr: float = 0.001,
    hidden_dim: int = 64,
    max_features: int = 256,
    min_feature_count: int = 1,
    max_examples: int | None = None,
    batch_size: int = 128,
    seed: int = 13,
    l2: float = 1e-5,
    input_clip: float = 8.0,
    device: str = "auto",
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for torch_mlp_trainer; install torch in the training environment.") from exc

    if device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = device
    torch_device = torch.device(device_name)
    torch.manual_seed(seed)

    stats = build_feature_stats(
        labels_path,
        max_features=max_features,
        min_feature_count=min_feature_count,
        max_examples=max_examples,
    )
    feature_names = stats.names
    feature_index = {name: index for index, name in enumerate(feature_names)}
    input_scale = stats.scale

    class Ranker(nn.Module):
        def __init__(self, input_dim: int, hidden: int):
            super().__init__()
            self.hidden = nn.Linear(input_dim, hidden)
            self.output = nn.Linear(hidden, 1)

        def forward(self, inputs: Any) -> Any:
            return self.output(torch.relu(self.hidden(inputs))).squeeze(-1)

    model = Ranker(len(feature_names), hidden_dim).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    latest_path = out_path / "latest.json"
    best_path = out_path / "best.json"
    best_acc = -1.0
    summary: dict[str, Any] = {}

    for epoch in range(1, max(1, int(epochs)) + 1):
        examples = 0
        total_loss = 0.0
        top1 = 0
        top3 = 0
        pending: list[dict[str, Any]] = []

        def flush_batch(batch: list[dict[str, Any]]) -> None:
            nonlocal examples, total_loss, top1, top3
            if not batch:
                return
            inputs, labels, mask = _batch_records(
                batch,
                feature_index=feature_index,
                input_scale=input_scale,
                input_clip=input_clip,
                device=torch_device,
            )
            logits = model(inputs)
            logits = logits.masked_fill(~mask, -1.0e9)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                predictions = torch.argsort(logits, dim=1, descending=True)
                batch_size_actual = labels.shape[0]
                examples += int(batch_size_actual)
                total_loss += float(loss.item()) * int(batch_size_actual)
                top1 += int((predictions[:, 0] == labels).sum().item())
                k = min(3, predictions.shape[1])
                top3 += int((predictions[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item())

        for record in _iter_records(labels_path, max_examples=max_examples):
            pending.append(record)
            if len(pending) >= batch_size:
                flush_batch(pending)
                pending = []
        flush_batch(pending)

        summary = {
            "epoch": epoch,
            "labels": str(labels_path),
            "out_dir": str(out_path),
            "examples": examples,
            "candidate_vectors_for_stats": stats.candidate_count,
            "feature_count": len(feature_names),
            "hidden_dim": hidden_dim,
            "normalization": "sparse",
            "average_loss": total_loss / examples if examples else 0.0,
            "top1_accuracy": top1 / examples if examples else 0.0,
            "top3_accuracy": top3 / examples if examples else 0.0,
            "lr": lr,
            "l2": l2,
            "batch_size": batch_size,
            "device": device_name,
            "trainer": "torch",
        }
        _save_torch_checkpoint(
            out_path / f"checkpoint_epoch_{epoch}.json",
            model=model,
            feature_names=feature_names,
            input_scale=input_scale,
            input_clip=input_clip,
            summary=summary,
        )
        _save_torch_checkpoint(
            latest_path,
            model=model,
            feature_names=feature_names,
            input_scale=input_scale,
            input_clip=input_clip,
            summary=summary,
        )
        if summary["top1_accuracy"] >= best_acc:
            best_acc = summary["top1_accuracy"]
            _save_torch_checkpoint(
                best_path,
                model=model,
                feature_names=feature_names,
                input_scale=input_scale,
                input_clip=input_clip,
                summary=summary,
            )

    summary["latest_checkpoint"] = str(latest_path)
    summary["best_checkpoint"] = str(best_path)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch MLP ranker and export stdlib JSON inference weights.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", default="artifacts/checkpoints/torch_mlp_ranker")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-features", type=int, default=256)
    parser.add_argument("--min-feature-count", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--input-clip", type=float, default=8.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device string.")
    args = parser.parse_args()
    train_torch_mlp_ranker(
        args.labels,
        out_dir=args.out_dir,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        max_features=args.max_features,
        min_feature_count=args.min_feature_count,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        seed=args.seed,
        l2=args.l2,
        input_clip=args.input_clip,
        device=args.device,
    )


if __name__ == "__main__":
    main()
