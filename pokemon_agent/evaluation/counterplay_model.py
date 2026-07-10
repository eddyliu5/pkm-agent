from __future__ import annotations

from dataclasses import dataclass

from ..belief import OpponentBelief
from ..card_db import attack_data, card_data
from ..opponent_memory import ResponseOpportunity
from ..state import AgentState, PokemonView


@dataclass(frozen=True)
class OpponentResponseRisk:
    total: float = 0.0
    active_attack: float = 0.0
    gust_attack: float = 0.0
    switch_attack: float = 0.0
    energy_attack: float = 0.0
    tool_attack: float = 0.0
    evolution_attack: float = 0.0


@dataclass(frozen=True)
class CounterplayEvaluation:
    root_risk: float
    final_risk: OpponentResponseRisk
    fragility: float

    @property
    def score(self) -> float:
        return -self.fragility


def _attack_damage(attacker_id: int, target: PokemonView, available_energy: int) -> float:
    attacker = card_data(attacker_id)
    target_data = card_data(target.id)
    best = 0.0
    for attack_id in getattr(attacker, "attacks", ()) or ():
        attack = attack_data(attack_id)
        if attack is None or len(getattr(attack, "energies", ()) or ()) > available_energy:
            continue
        damage = float(getattr(attack, "damage", 0) or 0)
        if attacker is not None and target_data is not None:
            if target_data.weakness == attacker.energyType:
                damage *= 2.0
            if target_data.resistance == attacker.energyType:
                damage = max(0.0, damage - 30.0)
        best = max(best, damage)
    return best


def _response_impact(state: AgentState, target: PokemonView | None, damage: float) -> float:
    if target is None or damage <= 0:
        return 0.0
    fraction = min(1.0, damage / max(1.0, float(target.hp)))
    impact = 0.25 * fraction
    if damage >= target.hp:
        if state.opp_prizes_remaining <= target.prize_value:
            return 1.0
        impact += 0.35 + (0.20 * min(3, target.prize_value))
    return min(1.0, impact)


def _probability_from_count(count: float, hand_count: int, hidden_count: int) -> float:
    if count <= 0 or hand_count <= 0:
        return 0.0
    miss = max(0.0, float(hidden_count) - float(count)) / max(1.0, float(hidden_count))
    return max(0.0, min(1.0, 1.0 - (miss**hand_count)))


class CounterplayModel:
    """Estimate the strongest immediate opponent response to a completed sequence."""

    def response_risk(self, state: AgentState | None, opponent: OpponentBelief) -> OpponentResponseRisk:
        if state is None or state.result != -1 or state.my_active is None or state.opp_active is None:
            return OpponentResponseRisk()

        active_damage = _attack_damage(state.opp_active.id, state.my_active, state.opp_active.energy_count)
        active_attack = _response_impact(state, state.my_active, active_damage)

        gust_attack = 0.0
        if state.my_bench and active_damage > 0:
            bench_impact = max(_response_impact(state, target, _attack_damage(state.opp_active.id, target, state.opp_active.energy_count)) for target in state.my_bench)
            gust_attack = opponent.prob_has_gust * bench_impact

        switch_attack = 0.0
        for attacker in state.opp_bench:
            damage = _attack_damage(attacker.id, state.my_active, attacker.energy_count)
            switch_attack = max(switch_attack, opponent.prob_has_switch * _response_impact(state, state.my_active, damage))

        energy_damage = _attack_damage(state.opp_active.id, state.my_active, state.opp_active.energy_count + 1)
        energy_attack = opponent.prob_has_energy * _response_impact(state, state.my_active, energy_damage)

        tool_attack = 0.0
        if active_damage > 0 and active_damage < state.my_active.hp <= active_damage + 30.0:
            tool_attack = opponent.prob_has_tool * _response_impact(state, state.my_active, active_damage + 30.0)

        evolution_attack = self._evolution_attack_risk(state, opponent)
        values = (active_attack, gust_attack, switch_attack, energy_attack, tool_attack, evolution_attack)
        return OpponentResponseRisk(
            total=max(values),
            active_attack=active_attack,
            gust_attack=gust_attack,
            switch_attack=switch_attack,
            energy_attack=energy_attack,
            tool_attack=tool_attack,
            evolution_attack=evolution_attack,
        )

    def evaluate(
        self,
        root_state: AgentState,
        final_state: AgentState | None,
        opponent: OpponentBelief,
        *,
        root_risk: float | None = None,
    ) -> CounterplayEvaluation:
        baseline = self.response_risk(root_state, opponent).total if root_risk is None else float(root_risk)
        final = self.response_risk(final_state, opponent)
        return CounterplayEvaluation(
            root_risk=baseline,
            final_risk=final,
            fragility=max(0.0, final.total - baseline),
        )

    def response_opportunities(
        self,
        state: AgentState | None,
        opponent: OpponentBelief,
    ) -> tuple[ResponseOpportunity, ...]:
        if state is None or state.result != -1 or state.my_active is None or state.opp_active is None:
            return ()
        opportunities: list[ResponseOpportunity] = []
        active_damage = _attack_damage(state.opp_active.id, state.my_active, state.opp_active.energy_count)
        active_impact = _response_impact(state, state.my_active, active_damage)

        if state.my_bench and active_damage > 0:
            bench_impact = max(
                _response_impact(
                    state,
                    target,
                    _attack_damage(state.opp_active.id, target, state.opp_active.energy_count),
                )
                for target in state.my_bench
            )
            self._append_opportunity(opportunities, "gust", opponent.prob_has_gust, active_impact, bench_impact)

        best_switch_impact = max(
            (
                _response_impact(
                    state,
                    state.my_active,
                    _attack_damage(attacker.id, state.my_active, attacker.energy_count),
                )
                for attacker in state.opp_bench
            ),
            default=0.0,
        )
        self._append_opportunity(opportunities, "switch", opponent.prob_has_switch, active_impact, best_switch_impact)

        energy_damage = _attack_damage(state.opp_active.id, state.my_active, state.opp_active.energy_count + 1)
        energy_impact = _response_impact(state, state.my_active, energy_damage)
        self._append_opportunity(opportunities, "energy", opponent.prob_has_energy, active_impact, energy_impact)

        if active_damage > 0 and active_damage < state.my_active.hp <= active_damage + 30.0:
            tool_impact = _response_impact(state, state.my_active, active_damage + 30.0)
            self._append_opportunity(opportunities, "tool", opponent.prob_has_tool, active_impact, tool_impact)

        evolution_prior, evolution_impact = self._evolution_probability_and_conditional_impact(state, opponent)
        self._append_opportunity(opportunities, "evolution", evolution_prior, active_impact, evolution_impact)
        return tuple(opportunities)

    @staticmethod
    def _append_opportunity(
        opportunities: list[ResponseOpportunity],
        category: str,
        prior_probability: float,
        ordinary_impact: float,
        response_impact: float,
    ) -> None:
        gain = float(response_impact) - float(ordinary_impact)
        if response_impact < 0.55 or gain < 0.20 or prior_probability <= 0.0:
            return
        use_probability = min(0.95, 0.55 + (0.40 * gain))
        opportunities.append(
            ResponseOpportunity(
                category=category,
                prior_probability=float(prior_probability),
                use_probability=use_probability,
            )
        )

    @staticmethod
    def _evolution_attack_risk(state: AgentState, opponent: OpponentBelief) -> float:
        probability, impact = CounterplayModel._evolution_probability_and_conditional_impact(state, opponent)
        return probability * impact

    @staticmethod
    def _evolution_probability_and_conditional_impact(
        state: AgentState,
        opponent: OpponentBelief,
    ) -> tuple[float, float]:
        setup_ids = set(opponent.intent.setup_target_ids)
        setup = [pokemon for pokemon in ([state.opp_active] if state.opp_active else []) + list(state.opp_bench) if pokemon.id in setup_ids]
        if not setup or not opponent.main_attacker_ids:
            return (0.0, 0.0)
        remaining = sum(float(opponent.remaining_card_counts.get(card_id, 0.0)) for card_id in opponent.main_attacker_ids)
        hidden = max(1, state.opp_deck_count + state.opp_hand_count + state.opp_prizes_remaining)
        known_evolution = any(opponent.known_hand_counts.get(card_id, 0) > 0 for card_id in opponent.main_attacker_ids)
        evolution_probability = 1.0 if known_evolution else _probability_from_count(remaining, state.opp_hand_count, hidden)
        best = 0.0
        for pokemon in setup:
            switch_probability = 1.0 if pokemon is state.opp_active else opponent.prob_has_switch
            for attacker_id in opponent.main_attacker_ids:
                damage_now = _attack_damage(attacker_id, state.my_active, pokemon.energy_count)
                damage_plus_energy = _attack_damage(attacker_id, state.my_active, pokemon.energy_count + 1)
                impact_now = _response_impact(state, state.my_active, damage_now)
                impact_plus_energy = opponent.prob_has_energy * _response_impact(state, state.my_active, damage_plus_energy)
                best = max(best, switch_probability * max(impact_now, impact_plus_energy))
        return (evolution_probability, best)
