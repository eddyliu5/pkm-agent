from __future__ import annotations

from typing import Any

from ..cg_loader import ensure_cg_path

ensure_cg_path()

from cg.api import search_begin, search_end, search_step  # noqa: E402


class SearchAdapter:
    """Thin wrapper around simulator search calls."""

    def begin(self, obs: Any, sample: Any) -> Any:
        return search_begin(
            obs,
            list(sample.your_deck),
            list(sample.your_prize),
            list(sample.opponent_deck),
            list(sample.opponent_prize),
            list(sample.opponent_hand),
            list(sample.opponent_active),
            False,
        )

    def step(self, search_id: int, selected: list[int]) -> Any:
        return search_step(int(search_id), selected)

    def end(self) -> None:
        search_end()
