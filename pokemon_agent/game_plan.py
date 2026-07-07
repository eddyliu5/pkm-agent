from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .card_db import card_data
from .state import AgentState


@dataclass(frozen=True)
class TurnPlan:
    phase: str
    primary_goal: str
    secondary_goals: tuple[str, ...]
    must_answer_threats: tuple[str, ...]
    desired_resources: tuple[str, ...]
    acceptable_risks: tuple[str, ...]
    avoid_patterns: tuple[str, ...]


@dataclass(frozen=True)
class PlanContext:
    deck_plan_type: str
    phase: str
    priority_weights: dict[str, float]
    preferred_attackers: tuple[int, ...]
    target_prize_route: tuple[int, ...]
    turn_plan: TurnPlan


class GamePlanManager:
    """Human-style strategic framing for the current decision."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def build(self, state: AgentState) -> PlanContext:
        deck_plan_type = self._deck_plan_type()
        phase = self._phase(state, deck_plan_type)
        weights = self._priority_weights(deck_plan_type, phase)
        preferred = tuple(int(card_id) for card_id in self.config.get("main_attackers", []))
        if not preferred:
            preferred = tuple(int(card_id) for card_id in self.config.get("backup_attackers", []))
        return PlanContext(
            deck_plan_type=deck_plan_type,
            phase=phase,
            priority_weights=weights,
            preferred_attackers=preferred,
            target_prize_route=self._target_prize_route(state),
            turn_plan=self._turn_plan(state, deck_plan_type, phase),
        )

    def _deck_plan_type(self) -> str:
        explicit = self.config.get("deck_plan_type")
        if explicit:
            return str(explicit)
        main_attackers = [card_data(int(card_id)) for card_id in self.config.get("main_attackers", [])]
        if any(data and getattr(data, "megaEx", False) for data in main_attackers):
            return "setup_big_attacker"
        notes = str(self.config.get("notes", "")).lower()
        if "control" in notes or "deck_out" in notes:
            return "control"
        if "tempo" in notes or "early" in notes:
            return "pressure"
        return "hybrid"

    def _phase(self, state: AgentState, deck_plan_type: str) -> str:
        if state.my_prizes_remaining <= 2 or state.opp_prizes_remaining <= 2:
            return "close"
        if state.opp_prizes_remaining < state.my_prizes_remaining and state.my_prizes_remaining >= 4:
            return "panic"
        if state.turn <= 2:
            return "setup"
        if deck_plan_type == "pressure":
            return "pressure"
        if deck_plan_type == "setup_big_attacker" and state.my_energy_in_play < 3:
            return "stabilize"
        return "pressure"

    def _priority_weights(self, deck_plan_type: str, phase: str) -> dict[str, float]:
        weights = {
            "early_damage": 0.5,
            "survival": 0.5,
            "energy_setup": 0.5,
            "hand_quality": 0.4,
            "closing": 0.5,
        }
        if deck_plan_type == "pressure":
            weights.update({"early_damage": 1.0, "energy_setup": 0.45, "survival": 0.35})
        elif deck_plan_type == "setup_big_attacker":
            weights.update({"energy_setup": 1.0, "survival": 0.85, "early_damage": 0.35})
        elif deck_plan_type == "control":
            weights.update({"survival": 1.0, "hand_quality": 0.75, "early_damage": 0.2})
        if phase == "close":
            weights["closing"] = 1.25
            weights["survival"] += 0.2
        elif phase == "panic":
            weights["survival"] = 1.25
            weights["hand_quality"] = 0.8
        elif phase == "setup":
            weights["energy_setup"] += 0.25
            weights["hand_quality"] += 0.25
        return weights

    def _turn_plan(self, state: AgentState, deck_plan_type: str, phase: str) -> TurnPlan:
        if phase == "setup":
            primary = "establish_board"
            secondary = ("find_main_line", "attach_energy", "improve_hand")
            desired = ("basic_attacker", "energy_attachment", "search_or_draw")
            avoid = ("overbench_liability", "pass_with_useful_action")
        elif phase == "stabilize":
            primary = "stabilize_attacker"
            secondary = ("evolve_main_line", "preserve_next_attacker", "prepare_attack")
            desired = ("evolution", "energy_in_play", "bench_attacker")
            avoid = ("expose_single_attacker", "waste_energy")
        elif phase == "close":
            primary = "close_prize_race"
            secondary = ("take_guaranteed_prize", "preserve_gust_outs", "deny_counter_ko")
            desired = ("ko_pressure", "gust_or_switch_out", "ready_attacker")
            avoid = ("unnecessary_setup", "miss_lethal")
        elif phase == "panic":
            primary = "avoid_immediate_loss"
            secondary = ("heal_or_switch", "deny_gust_target", "rebuild_hand")
            desired = ("defensive_switch", "backup_attacker", "draw_support")
            avoid = ("bench_liability", "low_payoff_attack")
        else:
            primary = "apply_pressure"
            secondary = ("take_prize", "deny_evolution", "prepare_followup")
            desired = ("ready_attack", "gust_pressure", "second_attacker")
            avoid = ("passive_turn", "unsupported_gust")

        threats: list[str] = []
        if state.my_active and state.my_active.hp <= 80:
            threats.append("active_ko_risk")
        if state.opp_prizes_remaining <= 2:
            threats.append("opponent_close_turn")
        if deck_plan_type == "setup_big_attacker" and state.my_energy_in_play < 3:
            threats.append("setup_window")

        risks = ("tempo_trade",) if phase in {"pressure", "close"} else ("small_prize_deficit",)
        return TurnPlan(
            phase=phase,
            primary_goal=primary,
            secondary_goals=secondary,
            must_answer_threats=tuple(threats),
            desired_resources=desired,
            acceptable_risks=risks,
            avoid_patterns=avoid,
        )

    @staticmethod
    def _target_prize_route(state: AgentState) -> tuple[int, ...]:
        values = sorted((pokemon.prize_value for pokemon in ([state.opp_active] if state.opp_active else []) + list(state.opp_bench)), reverse=True)
        route: list[int] = []
        total = 0
        for value in values:
            route.append(value)
            total += value
            if total >= state.my_prizes_remaining:
                break
        return tuple(route)
