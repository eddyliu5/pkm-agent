from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import load_config
from .decision_engine import DecisionEngine
from .mlp_ranker import MLPSequenceRanker
from .runtime import RuntimeAdapter
from .sequence import SequenceSearchConfig
from .xgb_root_ranker import load_root_ranker


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


def _load_mlp_ranker(root: Path, options: dict) -> MLPSequenceRanker | None:
    configured = options.get("mlp_model_file")
    candidates: list[Path] = []
    if configured:
        candidates.append(root / str(configured))
    candidates.extend([root / "mlp_model.pt", root / "mlp_model.json"])

    seen: set[Path] = set()
    last_error: Exception | None = None
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        try:
            return MLPSequenceRanker.from_file(path)
        except Exception as exc:  # pragma: no cover - only exercised in packaged runtimes.
            last_error = exc
            print(f"Failed to load MLP model {path}: {exc}", file=sys.stderr)
    if last_error is not None:
        raise last_error
    return None


def _load_xgb_ranker(root: Path, options: dict):
    configured = options.get("xgb_model_file")
    if not configured:
        return None
    path = root / str(configured)
    if not path.is_file():
        raise FileNotFoundError(f"runtime_config XGB model is missing: {path}")
    return load_root_ranker(path)


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
XGB_MODEL_CONFIGURED = bool(RUNTIME_OPTIONS.get("xgb_model_file"))
if XGB_MODEL_CONFIGURED and (
    RUNTIME_OPTIONS.get("mlp_model_file") or bool(RUNTIME_OPTIONS.get("mlp_enabled", False))
):
    raise ValueError("MLP and XGB rankers cannot be enabled together")
MLP_RANKER = None if XGB_MODEL_CONFIGURED else _load_mlp_ranker(ROOT, RUNTIME_OPTIONS)
MLP_ENABLED = bool(RUNTIME_OPTIONS.get("mlp_enabled", MLP_RANKER is not None))
if MLP_ENABLED and MLP_RANKER is None:
    raise FileNotFoundError("runtime_config enables MLP, but mlp_model.pt is missing")
XGB_RANKER = _load_xgb_ranker(ROOT, RUNTIME_OPTIONS)
if MLP_ENABLED and XGB_RANKER is not None:
    raise ValueError("MLP and XGB rankers cannot be enabled together")
DEFAULT_LINEAR_SCORE_WEIGHT = 0.0 if MLP_ENABLED else 1.0
DEFAULT_MLP_SCORE_WEIGHT = 1.0
ADAPTER = RuntimeAdapter()
ENGINE = DecisionEngine(
    deck=DECK,
    config=CONFIG,
    weights=WEIGHTS,
    mlp_ranker=MLP_RANKER,
    xgb_ranker=XGB_RANKER,
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
