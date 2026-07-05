from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cg_loader import PROJECT_ROOT


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "starmie_froslass.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_deck(path: str | Path | None = None, config: dict[str, Any] | None = None) -> list[int]:
    if path is None:
        cfg = config or load_config()
        path = cfg["deck_path"]
    deck_path = Path(path)
    if not deck_path.is_absolute():
        deck_path = PROJECT_ROOT / deck_path
    with deck_path.open("r", encoding="utf-8") as fh:
        deck = [int(line.strip()) for line in fh if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"Expected a 60-card deck, got {len(deck)} cards from {deck_path}")
    return deck


def card_ids(config: dict[str, Any]) -> dict[str, int]:
    return {key: int(value) for key, value in config.get("card_ids", {}).items()}
