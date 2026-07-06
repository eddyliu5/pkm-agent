from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MODEL_TYPE = "mlp_sequence_ranker"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FeatureStats:
    names: list[str]
    mean: list[float]
    scale: list[float]
    feature_counts: dict[str, int]
    example_count: int
    candidate_count: int


def _clean_features(features: Any) -> dict[str, float]:
    if not isinstance(features, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key, value in features.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number and math.isfinite(number):
            cleaned[str(key)] = number
    return cleaned


def _iter_label_examples(labels_path: str | Path, max_examples: int | None = None) -> Iterable[tuple[int, list[dict[str, float]]]]:
    seen = 0
    with Path(labels_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            candidates = record.get("candidates") or []
            if not isinstance(candidates, list) or not candidates:
                continue
            try:
                label_index = int(record.get("label_index", -1))
            except (TypeError, ValueError):
                continue
            if label_index < 0 or label_index >= len(candidates):
                continue
            if not bool(record.get("matched", label_index >= 0)):
                continue
            candidate_features = [_clean_features(candidate.get("features") if isinstance(candidate, dict) else {}) for candidate in candidates]
            if not candidate_features:
                continue
            yield label_index, candidate_features
            seen += 1
            if max_examples is not None and seen >= max_examples:
                break


def build_feature_stats(
    labels_path: str | Path,
    *,
    max_features: int = 256,
    min_feature_count: int = 1,
    max_examples: int | None = None,
) -> FeatureStats:
    counts: Counter[str] = Counter()
    example_count = 0
    candidate_count = 0
    for _label_index, candidates in _iter_label_examples(labels_path, max_examples=max_examples):
        example_count += 1
        candidate_count += len(candidates)
        for features in candidates:
            counts.update(features.keys())

    ranked = [(key, count) for key, count in counts.items() if count >= min_feature_count]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    if max_features > 0:
        ranked = ranked[:max_features]
    names = [key for key, _count in ranked]
    if not names:
        raise ValueError(f"No usable candidate features found in {labels_path}")

    selected = set(names)
    sums = {name: 0.0 for name in names}
    sums_sq = {name: 0.0 for name in names}
    vector_count = 0
    for _label_index, candidates in _iter_label_examples(labels_path, max_examples=max_examples):
        for features in candidates:
            vector_count += 1
            for key, value in features.items():
                if key in selected:
                    sums[key] += value
                    sums_sq[key] += value * value

    if vector_count <= 0:
        raise ValueError(f"No matched label examples found in {labels_path}")

    mean: list[float] = []
    scale: list[float] = []
    for name in names:
        avg = sums[name] / float(vector_count)
        variance = max(0.0, (sums_sq[name] / float(vector_count)) - (avg * avg))
        std = math.sqrt(variance)
        mean.append(avg)
        scale.append(std if std >= 1e-6 else 1.0)

    return FeatureStats(
        names=names,
        mean=mean,
        scale=scale,
        feature_counts={key: int(value) for key, value in counts.items()},
        example_count=example_count,
        candidate_count=candidate_count,
    )


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    best = max(logits)
    values = [math.exp(max(-40.0, min(40.0, value - best))) for value in logits]
    total = sum(values) or 1.0
    return [value / total for value in values]


def _clip(value: float, limit: float) -> float:
    if limit <= 0:
        return value
    return max(-limit, min(limit, value))


class MLPSequenceRanker:
    """Tiny dependency-free MLP that scores sequence feature dictionaries."""

    def __init__(
        self,
        *,
        feature_names: list[str],
        input_mean: list[float],
        input_scale: list[float],
        hidden_dim: int,
        w1: list[list[float]],
        b1: list[float],
        w2: list[float],
        b2: float,
        input_clip: float = 8.0,
        activation: str = "relu",
        normalization: str = "standard",
    ):
        if activation != "relu":
            raise ValueError(f"Unsupported activation: {activation}")
        if normalization not in {"standard", "sparse"}:
            raise ValueError(f"Unsupported normalization: {normalization}")
        if len(feature_names) != len(input_mean) or len(feature_names) != len(input_scale):
            raise ValueError("Feature names, means, and scales must have the same length")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.feature_names = [str(name) for name in feature_names]
        self.feature_index = {name: index for index, name in enumerate(self.feature_names)}
        self.input_mean = [float(value) for value in input_mean]
        self.input_scale = [float(value) if float(value) else 1.0 for value in input_scale]
        self.hidden_dim = int(hidden_dim)
        self.w1 = [[float(value) for value in row] for row in w1]
        self.b1 = [float(value) for value in b1]
        self.w2 = [float(value) for value in w2]
        self.b2 = float(b2)
        self.input_clip = float(input_clip)
        self.activation = activation
        self.normalization = normalization
        input_dim = len(self.feature_names)
        if len(self.w1) != self.hidden_dim or any(len(row) != input_dim for row in self.w1):
            raise ValueError("w1 shape does not match hidden_dim x input_dim")
        if len(self.b1) != self.hidden_dim or len(self.w2) != self.hidden_dim:
            raise ValueError("Hidden layer parameter shapes do not match hidden_dim")

    @classmethod
    def initialize(
        cls,
        *,
        feature_stats: FeatureStats,
        hidden_dim: int = 64,
        seed: int = 13,
        input_clip: float = 8.0,
        normalization: str = "sparse",
    ) -> "MLPSequenceRanker":
        rng = random.Random(seed)
        input_dim = len(feature_stats.names)
        limit1 = math.sqrt(6.0 / float(max(1, input_dim + hidden_dim)))
        limit2 = math.sqrt(6.0 / float(max(1, hidden_dim + 1)))
        input_mean = [0.0 for _ in feature_stats.names] if normalization == "sparse" else feature_stats.mean
        w1 = [[rng.uniform(-limit1, limit1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        b1 = [0.0 for _ in range(hidden_dim)]
        w2 = [rng.uniform(-limit2, limit2) for _ in range(hidden_dim)]
        return cls(
            feature_names=feature_stats.names,
            input_mean=input_mean,
            input_scale=feature_stats.scale,
            hidden_dim=hidden_dim,
            w1=w1,
            b1=b1,
            w2=w2,
            b2=0.0,
            input_clip=input_clip,
            normalization=normalization,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MLPSequenceRanker":
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError(f"Not an {MODEL_TYPE} payload")
        return cls(
            feature_names=list(payload["feature_names"]),
            input_mean=list(payload["input_mean"]),
            input_scale=list(payload["input_scale"]),
            hidden_dim=int(payload["hidden_dim"]),
            w1=[list(row) for row in payload["w1"]],
            b1=list(payload["b1"]),
            w2=list(payload["w2"]),
            b2=float(payload["b2"]),
            input_clip=float(payload.get("input_clip", 8.0)),
            activation=str(payload.get("activation", "relu")),
            normalization=str(payload.get("normalization", "standard")),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "MLPSequenceRanker":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_payload(json.load(fh))

    def to_payload(self, summary: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_type": MODEL_TYPE,
            "feature_names": self.feature_names,
            "input_mean": self.input_mean,
            "input_scale": self.input_scale,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "normalization": self.normalization,
            "input_clip": self.input_clip,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "summary": summary or {},
        }

    def save(self, path: str | Path, summary: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self.to_payload(summary=summary), fh, indent=2, sort_keys=True)

    def vectorize(self, features: dict[str, float]) -> list[float]:
        values: list[float] = []
        for name, mean, scale in zip(self.feature_names, self.input_mean, self.input_scale):
            raw = float(features.get(name, 0.0))
            normalized = (raw - mean) / scale
            values.append(_clip(normalized, self.input_clip))
        return values

    def active_vector_items(self, features: dict[str, float]) -> list[tuple[int, float]]:
        if self.normalization != "sparse":
            return [(index, value) for index, value in enumerate(self.vectorize(features)) if value]
        items: list[tuple[int, float]] = []
        for key, value in features.items():
            index = self.feature_index.get(str(key))
            if index is None:
                continue
            normalized = _clip(float(value) / self.input_scale[index], self.input_clip)
            if normalized:
                items.append((index, normalized))
        return items

    def _forward_vector(self, vector: list[float]) -> tuple[float, list[float], list[float]]:
        pre_activation: list[float] = []
        hidden: list[float] = []
        for row, bias in zip(self.w1, self.b1):
            value = bias
            for weight, item in zip(row, vector):
                value += weight * item
            pre_activation.append(value)
            hidden.append(value if value > 0.0 else 0.0)
        score = self.b2
        for weight, item in zip(self.w2, hidden):
            score += weight * item
        return score, pre_activation, hidden

    def _forward_items(self, items: list[tuple[int, float]]) -> tuple[float, list[float], list[float]]:
        pre_activation: list[float] = []
        hidden: list[float] = []
        for row, bias in zip(self.w1, self.b1):
            value = bias
            for index, item in items:
                value += row[index] * item
            pre_activation.append(value)
            hidden.append(value if value > 0.0 else 0.0)
        score = self.b2
        for weight, item in zip(self.w2, hidden):
            score += weight * item
        return score, pre_activation, hidden

    def score_features(self, features: dict[str, float]) -> float:
        if self.normalization == "sparse":
            score, _pre_activation, _hidden = self._forward_items(self.active_vector_items(features))
            return score
        score, _pre_activation, _hidden = self._forward_vector(self.vectorize(features))
        return score

    def score_batch(self, candidates: list[dict[str, float]]) -> list[float]:
        return [self.score_features(features) for features in candidates]

    def train_step(
        self,
        candidates: list[dict[str, float]],
        label_index: int,
        *,
        lr: float,
        temperature: float = 1.0,
        l2: float = 0.0,
        grad_clip: float = 5.0,
        loss_scale: float = 1.0,
    ) -> tuple[float, bool, bool]:
        if label_index < 0 or label_index >= len(candidates):
            raise ValueError("label_index is out of range")
        temperature = max(1e-6, float(temperature))
        sparse = self.normalization == "sparse"
        vectors = [self.active_vector_items(features) if sparse else self.vectorize(features) for features in candidates]
        caches = [self._forward_items(vector) if sparse else self._forward_vector(vector) for vector in vectors]
        scores = [cache[0] for cache in caches]
        probabilities = _softmax([score / temperature for score in scores])
        loss = -math.log(max(1e-12, probabilities[label_index]))

        input_dim = len(self.feature_names)
        if sparse:
            grad_w1_sparse: list[dict[int, float]] = [{} for _ in range(self.hidden_dim)]
            grad_w1: list[list[float]] = []
        else:
            grad_w1 = [[0.0 for _ in range(input_dim)] for _ in range(self.hidden_dim)]
            grad_w1_sparse = []
        grad_b1 = [0.0 for _ in range(self.hidden_dim)]
        grad_w2 = [0.0 for _ in range(self.hidden_dim)]
        grad_b2 = 0.0
        old_w2 = list(self.w2)

        for cand_index, (vector, (_score, pre_activation, hidden)) in enumerate(zip(vectors, caches)):
            dscore = loss_scale * (probabilities[cand_index] - (1.0 if cand_index == label_index else 0.0)) / temperature
            grad_b2 += dscore
            for hidden_index, hidden_value in enumerate(hidden):
                grad_w2[hidden_index] += dscore * hidden_value
                if pre_activation[hidden_index] <= 0.0:
                    continue
                dhidden = dscore * old_w2[hidden_index]
                grad_b1[hidden_index] += dhidden
                if sparse:
                    row_sparse = grad_w1_sparse[hidden_index]
                    for input_index, input_value in vector:
                        row_sparse[input_index] = row_sparse.get(input_index, 0.0) + dhidden * input_value
                else:
                    row = grad_w1[hidden_index]
                    for input_index, input_value in enumerate(vector):
                        row[input_index] += dhidden * input_value

        for hidden_index in range(self.hidden_dim):
            self.w2[hidden_index] -= lr * (_clip(grad_w2[hidden_index], grad_clip) + l2 * self.w2[hidden_index])
        self.b2 -= lr * _clip(grad_b2, grad_clip)

        for hidden_index in range(self.hidden_dim):
            self.b1[hidden_index] -= lr * _clip(grad_b1[hidden_index], grad_clip)
            row = self.w1[hidden_index]
            if sparse:
                for input_index, gradient in grad_w1_sparse[hidden_index].items():
                    row[input_index] -= lr * (_clip(gradient, grad_clip) + l2 * row[input_index])
            else:
                grad_row = grad_w1[hidden_index]
                for input_index in range(input_dim):
                    row[input_index] -= lr * (_clip(grad_row[input_index], grad_clip) + l2 * row[input_index])

        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return loss, order[0] == label_index, label_index in order[:3]


def pretrain_mlp_ranker(
    labels_path: str | Path,
    *,
    out_dir: str = "artifacts/checkpoints/mlp_ranker",
    epochs: int = 3,
    lr: float = 0.001,
    hidden_dim: int = 64,
    max_features: int = 256,
    min_feature_count: int = 1,
    max_examples: int | None = None,
    seed: int = 13,
    temperature: float = 1.0,
    l2: float = 1e-5,
    input_clip: float = 8.0,
    grad_clip: float = 5.0,
    normalization: str = "sparse",
) -> dict[str, Any]:
    stats = build_feature_stats(
        labels_path,
        max_features=max_features,
        min_feature_count=min_feature_count,
        max_examples=max_examples,
    )
    model = MLPSequenceRanker.initialize(
        feature_stats=stats,
        hidden_dim=hidden_dim,
        seed=seed,
        input_clip=input_clip,
        normalization=normalization,
    )
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
        for label_index, candidates in _iter_label_examples(labels_path, max_examples=max_examples):
            loss, is_top1, is_top3 = model.train_step(
                candidates,
                label_index,
                lr=lr,
                temperature=temperature,
                l2=l2,
                grad_clip=grad_clip,
            )
            examples += 1
            total_loss += loss
            top1 += int(is_top1)
            top3 += int(is_top3)

        summary = {
            "epoch": epoch,
            "labels": str(labels_path),
            "out_dir": str(out_path),
            "examples": examples,
            "candidate_vectors_for_stats": stats.candidate_count,
            "feature_count": len(stats.names),
            "hidden_dim": hidden_dim,
            "normalization": normalization,
            "average_loss": total_loss / examples if examples else 0.0,
            "top1_accuracy": top1 / examples if examples else 0.0,
            "top3_accuracy": top3 / examples if examples else 0.0,
            "lr": lr,
            "temperature": temperature,
            "l2": l2,
        }
        model.save(out_path / f"checkpoint_epoch_{epoch}.json", summary=summary)
        model.save(latest_path, summary=summary)
        if summary["top1_accuracy"] >= best_acc:
            best_acc = summary["top1_accuracy"]
            model.save(best_path, summary=summary)

    summary["latest_checkpoint"] = str(latest_path)
    summary["best_checkpoint"] = str(best_path)
    return summary
