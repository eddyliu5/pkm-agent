from __future__ import annotations


class RiskModel:
    """Risk summary from belief-derived probability features."""

    NEGATIVE_KEYS = (
        "p_main_attacker_unavailable",
        "p_key_evolution_unavailable",
        "p_energy_unavailable",
        "p_search_target_unavailable",
        "p_any_key_prized",
    )
    POSITIVE_KEYS = (
        "p_energy_access_this_turn",
        "p_line_accessible_this_turn",
        "p_combo_line_live",
    )

    def score(self, features: dict[str, float]) -> float:
        downside = sum(float(features.get(key, 0.0)) for key in self.NEGATIVE_KEYS)
        upside = sum(float(features.get(key, 0.0)) for key in self.POSITIVE_KEYS)
        return (0.25 * upside) - downside
