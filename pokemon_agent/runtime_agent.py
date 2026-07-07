from __future__ import annotations

import json
from pathlib import Path

from .config import load_config
from .decision_engine import DecisionEngine
from .mlp_ranker import MLPSequenceRanker
from .runtime import RuntimeAdapter
from .sequence import SequenceSearchConfig


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


def _load_runtime_options(root: Path) -> dict:
    path = root / "runtime_config.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


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
RUNTIME_OPTIONS = _load_runtime_options(ROOT)
WEIGHTS = _load_weights(ROOT)
MLP_RANKER = _load_mlp_ranker(ROOT)
MLP_ENABLED = bool(RUNTIME_OPTIONS.get("mlp_enabled", MLP_RANKER is not None))
if MLP_ENABLED and MLP_RANKER is None:
    raise FileNotFoundError("runtime_config enables MLP, but mlp_model.json is missing")
DEFAULT_LINEAR_SCORE_WEIGHT = 0.0 if MLP_ENABLED else 1.0
DEFAULT_MLP_SCORE_WEIGHT = 1.0
ADAPTER = RuntimeAdapter()
ENGINE = DecisionEngine(
    deck=DECK,
    config=CONFIG,
    weights=WEIGHTS,
    mlp_ranker=MLP_RANKER,
    clean_evaluator=bool(RUNTIME_OPTIONS.get("clean_evaluator", False)),
    probability_features_enabled=bool(RUNTIME_OPTIONS.get("probability_features_enabled", True)),
    mlp_enabled=MLP_ENABLED,
    value_proxy_enabled=bool(RUNTIME_OPTIONS.get("value_proxy_enabled", True)),
    fallback_guardrail_enabled=bool(RUNTIME_OPTIONS.get("fallback_guardrail_enabled", True)),
    engine_exception_fallback_enabled=bool(RUNTIME_OPTIONS.get("engine_exception_fallback_enabled", False)),
    linear_score_weight=float(RUNTIME_OPTIONS.get("linear_score_weight", DEFAULT_LINEAR_SCORE_WEIGHT)),
    mlp_score_weight=float(RUNTIME_OPTIONS.get("mlp_score_weight", DEFAULT_MLP_SCORE_WEIGHT)),
    search_config=SequenceSearchConfig(
        beam_width=8,
        max_depth=8,
        max_nodes=128,
        time_budget_ms=500,
        num_hidden_samples=max(1, int(RUNTIME_OPTIONS.get("num_hidden_samples", 1))),
    ),
)


def agent(obs_dict: dict) -> list[int]:
    obs = ADAPTER.parse(obs_dict)
    return ADAPTER.clamp(obs, ENGINE.choose(obs), DECK)
