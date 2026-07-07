from __future__ import annotations

import math
import random

from typing import Any

from ..decision_types import ScoredCandidate


class ActionSelector:
    """Select a scored candidate for eval or train mode."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select(self, candidates: list[Any] | tuple[Any, ...], mode: str = "eval") -> Any | None:
        if not candidates:
            return None
        if mode == "train":
            temperature = 250.0
            best = max(self._score(candidate) for candidate in candidates)
            weights = [math.exp(max(-20.0, min(20.0, (self._score(candidate) - best) / temperature))) for candidate in candidates]
            total = sum(weights) or 1.0
            pick = self.rng.random() * total
            running = 0.0
            for candidate, weight in zip(candidates, weights):
                running += weight
                if running >= pick:
                    return candidate
        return max(candidates, key=lambda candidate: (self._score(candidate), -len(self._steps(candidate))))

    @staticmethod
    def _score(candidate: Any) -> float:
        return float(getattr(candidate, "score", 0.0))

    @staticmethod
    def _steps(candidate: Any) -> tuple[Any, ...]:
        sequence = candidate.candidate if isinstance(candidate, ScoredCandidate) else candidate
        return tuple(getattr(sequence, "steps", ()))
