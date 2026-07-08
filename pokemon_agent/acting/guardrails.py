from __future__ import annotations

from ..decision_types import ScoredCandidate


class FallbackGuardrail:
    """Post-evaluation fallback override checks."""

    LINEAR_MARGIN = 250.0
    MLP_DISAGREEMENT_TOLERANCE = 2.0

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
            if self._uses_mlp_score(chosen):
                if chosen.components.linear < self.LINEAR_MARGIN:
                    return True, "fallback_not_represented_low_linear_confidence"
                return False, None
            if chosen.score < self.LINEAR_MARGIN:
                return True, "fallback_not_represented_low_confidence"
            return False, None

        if self._uses_mlp_score(chosen) or self._uses_mlp_score(fallback_candidate):
            fallback_linear_edge = fallback_candidate.components.linear - chosen.components.linear
            mlp_disagreement = self._mlp_edge(chosen, fallback_candidate)
            if fallback_linear_edge >= self.LINEAR_MARGIN and mlp_disagreement <= self.MLP_DISAGREEMENT_TOLERANCE:
                return True, "fallback_linear_margin_guardrail"
            return False, None

        if chosen.score < fallback_candidate.score + self.LINEAR_MARGIN:
            return True, "fallback_margin_guardrail"
        return False, None

    @staticmethod
    def _uses_mlp_score(candidate: ScoredCandidate) -> bool:
        return candidate.components.mlp is not None

    @staticmethod
    def _mlp_edge(chosen: ScoredCandidate, fallback_candidate: ScoredCandidate) -> float:
        if chosen.components.mlp is not None and fallback_candidate.components.mlp is not None:
            return chosen.components.mlp - fallback_candidate.components.mlp
        return chosen.score - fallback_candidate.score
