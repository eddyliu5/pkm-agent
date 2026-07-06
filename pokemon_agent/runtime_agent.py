from __future__ import annotations

import json
import os
from pathlib import Path

from .cg_loader import ensure_cg_path
from .config import load_config
from .decision_engine import DecisionEngine
from .mlp_ranker import MLPSequenceRanker
from .sequence import SequenceSearchConfig

ensure_cg_path()

from cg.api import to_observation_class  # noqa: E402


def _agent_root() -> Path:
    bundle_root = Path(__file__).resolve().parents[1]
    if (bundle_root / "deck.csv").exists():
        return bundle_root
    local = Path.cwd()
    if (local / "deck.csv").exists():
        return local
    kaggle = Path("/kaggle_simulations/agent")
    if kaggle.exists():
        return kaggle
    return local


def _read_deck(root: Path) -> list[int]:
    with (root / "deck.csv").open("r", encoding="utf-8") as fh:
        return [int(line.strip()) for line in fh if line.strip()][:60]


def _load_weights(root: Path) -> dict[str, float]:
    path = root / "weights.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    weights = payload.get("weights", payload)
    return {str(key): float(value) for key, value in weights.items()}


def _load_mlp_ranker(root: Path) -> MLPSequenceRanker | None:
    path = root / "mlp_model.json"
    if not path.exists():
        return None
    return MLPSequenceRanker.from_file(path)


ROOT = _agent_root()
DECK = _read_deck(ROOT)
try:
    CONFIG = load_config(ROOT / "configs" / "starmie_froslass.json")
except FileNotFoundError:
    try:
        CONFIG = load_config(ROOT / "configs" / "mega_lucario_v62.json")
    except FileNotFoundError:
        try:
            CONFIG = load_config(ROOT / "configs" / "mega_lucario.json")
        except FileNotFoundError:
            CONFIG = load_config()
WEIGHTS = _load_weights(ROOT)
MLP_RANKER = _load_mlp_ranker(ROOT)
ENGINE = DecisionEngine(
    deck=DECK,
    config=CONFIG,
    weights=WEIGHTS,
    mlp_ranker=MLP_RANKER,
    search_config=SequenceSearchConfig(beam_width=8, max_depth=8, max_nodes=128, time_budget_ms=500),
)


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return DECK
    selected = ENGINE.choose(obs)
    option_count = len(obs.select.option)
    selected = [index for index in selected if isinstance(index, int) and 0 <= index < option_count]
    if len(selected) < obs.select.minCount:
        for index in range(option_count):
            if index not in selected:
                selected.append(index)
            if len(selected) >= obs.select.minCount:
                break
    return selected[: obs.select.maxCount]
