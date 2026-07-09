from __future__ import annotations


class CounterplayModel:
    """Compact opponent-response heuristic derived from observed belief features."""

    def score(self, features: dict[str, float]) -> float:
        risk = 0.0
        risk += 1.0 * float(features.get("threat_can_win_next_turn", 0.0))
        risk += 0.4 * float(features.get("threat_can_ko_active", 0.0))
        risk += 0.3 * float(features.get("opponent_gust_risk", 0.0))
        risk += 0.1 * float(features.get("opponent_switch_risk", 0.0))
        risk += 0.1 * float(features.get("opponent_tool_risk", 0.0))
        risk += 0.2 * float(features.get("opponent_likely_setting_up_attacker", 0.0))
        risk += 0.2 * float(features.get("opponent_has_must_answer_targets", 0.0))
        risk -= 0.2 * float(features.get("opponent_exhausted_gust", 0.0))
        risk -= 0.1 * float(features.get("opponent_exhausted_switch", 0.0))
        risk -= 0.1 * float(features.get("opponent_exhausted_tool", 0.0))
        return -risk
