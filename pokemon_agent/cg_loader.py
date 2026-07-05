from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CG_ROOT = (
    PROJECT_ROOT
    / "data"
    / "pokemon-tcg-ai-battle"
    / "pokemon-tcg-ai-battle"
    / "sample_submission"
    / "sample_submission"
)


def ensure_cg_path() -> None:
    """Make the bundled Kaggle battle engine importable in local runs."""
    if importlib.util.find_spec("cg") is not None:
        return
    if LOCAL_CG_ROOT.exists():
        sys.path.insert(0, str(LOCAL_CG_ROOT))


def cg_root() -> Path:
    return LOCAL_CG_ROOT
