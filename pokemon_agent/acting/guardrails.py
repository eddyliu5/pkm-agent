from __future__ import annotations

from ..decision_types import ScoredCandidate


class FallbackGuardrail:
    """Post-evaluation fallback override checks."""

    def should_use_fallback(
        self,
        chosen: ScoredCandidate,
        candidates: list[ScoredCandidate] | tuple[ScoredCandidate, ...],
        fallback_selected: list[int],
    ) -> tuple[bool, str | None]:
        if not fallback_selected or chosen.candidate.first_selection == fallback_selected:
            return False, None
        features = chosen.features
        tactical_payoff = (
            features.get("seq_terminal_win", 0.0) > 0
            or features.get("attack_wins_game", 0.0) > 0
            or features.get("seq_my_prizes_gained", 0.0) > 0
            or features.get("attack_takes_ko", 0.0) > 0
            or features.get("value_model_score", 0.0) >= 0.35
        )
        if tactical_payoff:
            return False, None
        fallback_key = tuple(fallback_selected)
        fallback_candidate = next((item for item in candidates if tuple(item.candidate.first_selection) == fallback_key), None)
        if fallback_candidate is None:
            if chosen.score < 250.0:
                return True, "fallback_not_represented_low_confidence"
            return False, None
        if chosen.score < fallback_candidate.score + 250.0:
            return True, "fallback_margin_guardrail"
        return False, None
