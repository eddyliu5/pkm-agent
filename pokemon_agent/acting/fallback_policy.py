from __future__ import annotations

from typing import Any

from ..policy import LinearQPolicy


class FallbackLinearPolicy(LinearQPolicy):
    """Post-evaluation fallback policy used by DecisionEngine act guardrails."""

    def choose(self, obs: Any) -> list[int]:
        return super().choose(obs)
