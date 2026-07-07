from __future__ import annotations

from typing import Any

from ..cg_loader import ensure_cg_path

ensure_cg_path()

from cg.api import to_observation_class  # noqa: E402


class RuntimeAdapter:
    """Parse Kaggle observations and clamp engine selections to simulator bounds."""

    def parse(self, obs_dict: dict[str, Any]) -> Any:
        return to_observation_class(obs_dict)

    def clamp(self, obs: Any, selected: list[int], fallback_deck: list[int]) -> list[int]:
        if obs.select is None:
            return fallback_deck
        option_count = len(obs.select.option)
        clamped = [index for index in selected if isinstance(index, int) and 0 <= index < option_count]
        if len(clamped) < obs.select.minCount:
            for index in range(option_count):
                if index not in clamped:
                    clamped.append(index)
                if len(clamped) >= obs.select.minCount:
                    break
        return clamped[: obs.select.maxCount]
