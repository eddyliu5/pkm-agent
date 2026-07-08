from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

from .cg_loader import PROJECT_ROOT, cg_root
from .config import DEFAULT_CONFIG, load_config, read_deck
from .deck_db import deck_heuristic_config, expand_deck, get_deck, load_deck_db
from .features import default_weights
from .sequence import default_clean_sequence_weights


def build_submission(
    weights_path: str | None = None,
    mlp_model_path: str | None = None,
    output: str = "artifacts/submission/submission.tar.gz",
    clean_evaluator: bool = False,
    deck_id: str | None = None,
) -> Path:
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    build_dir = output_path.parent / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    packaged_config = load_config()
    if deck_id:
        db = load_deck_db()
        deck_record = get_deck(deck_id, db)
        deck = expand_deck(deck_record)
        packaged_config.update(deck_heuristic_config(deck_record, db))
    else:
        deck = read_deck(config=packaged_config)
    with (build_dir / "deck.csv").open("w", encoding="utf-8") as fh:
        for card_id in deck:
            fh.write(f"{card_id}\n")

    if weights_path:
        with Path(weights_path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        weights = payload.get("weights", payload)
    elif clean_evaluator:
        weights = default_clean_sequence_weights()
    else:
        weights = default_weights()
    with (build_dir / "weights.json").open("w", encoding="utf-8") as fh:
        json.dump({"weights": weights}, fh, indent=2, sort_keys=True)
    packaged_mlp_model = None
    if mlp_model_path:
        source_mlp_model = Path(mlp_model_path)
        if source_mlp_model.suffix.lower() in {".pt", ".pth"}:
            packaged_mlp_model = "mlp_model.pt"
        else:
            packaged_mlp_model = "mlp_model.json"
        shutil.copy2(source_mlp_model, build_dir / packaged_mlp_model)
    linear_score_weight = 0.0 if mlp_model_path else 1.0
    with (build_dir / "runtime_config.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "clean_evaluator": bool(clean_evaluator),
                "probability_features_enabled": True,
                "mlp_enabled": bool(mlp_model_path),
                "value_proxy_enabled": True,
                "fallback_guardrail_enabled": True,
                "engine_exception_fallback_enabled": False,
                "num_hidden_samples": 1,
                "linear_score_weight": linear_score_weight,
                "mlp_score_weight": 1.0,
                "mlp_model_file": packaged_mlp_model,
            },
            fh,
            indent=2,
            sort_keys=True,
        )

    (build_dir / "main.py").write_text("from pokemon_agent.runtime_agent import agent\n", encoding="utf-8")
    ignore_generated = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(PROJECT_ROOT / "pokemon_agent", build_dir / "pokemon_agent", ignore=ignore_generated)
    config_dir = build_dir / "configs"
    config_dir.mkdir()
    packaged_config["deck_path"] = "deck.csv"
    with (config_dir / DEFAULT_CONFIG.name).open("w", encoding="utf-8") as fh:
        json.dump(packaged_config, fh, indent=2, sort_keys=True)
    deck_db_src = PROJECT_ROOT / "data" / "deck_db"
    if deck_db_src.exists():
        shutil.copytree(deck_db_src, build_dir / "data" / "deck_db")
    shutil.copytree(cg_root() / "cg", build_dir / "cg", ignore=ignore_generated)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    with tarfile.open(output_path, "w:gz") as tar:
        for item in build_dir.iterdir():
            tar.add(item, arcname=item.name)
    print(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Kaggle submission bundle.")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--mlp-model", default=None)
    parser.add_argument("--output", default="artifacts/submission/submission.tar.gz")
    parser.add_argument("--clean", action="store_true", help="Package the clean sequence evaluator runtime.")
    parser.add_argument("--deck-id", default=None, help="Package this deck_db deck id instead of the default deck.")
    args = parser.parse_args()
    build_submission(
        weights_path=args.weights,
        mlp_model_path=args.mlp_model,
        output=args.output,
        clean_evaluator=args.clean,
        deck_id=args.deck_id,
    )


if __name__ == "__main__":
    main()
