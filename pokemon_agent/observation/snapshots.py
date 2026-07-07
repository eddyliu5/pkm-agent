from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..belief import OpponentBelief
from ..card_db import attack_data, card_data, is_energy
from ..state import AgentState, PokemonView


@dataclass(frozen=True)
class ResourceSnapshot:
    ready_attackers: int
    bench_space: int
    energy_in_play: int
    hand_quality: float
    search_cards_in_hand: int
    draw_cards_in_hand: int
    energy_cards_in_hand: int
    once_per_turn_commitment_used: float


@dataclass(frozen=True)
class ThreatSnapshot:
    opponent_damage_ceiling: float
    opponent_can_ko_active: float
    opponent_can_win_next_turn: float
    opponent_gust_risk: float
    turns_until_my_win_estimate: float
    turns_until_opponent_win_estimate: float


def _ids(config: dict[str, Any], key: str) -> set[int]:
    return {int(card_id) for card_id in config.get(key, [])}


def _min_attack_cost(card_id: int) -> int:
    data = card_data(card_id)
    costs: list[int] = []
    for attack_id in getattr(data, "attacks", []) or []:
        attack = attack_data(attack_id)
        if attack is not None:
            costs.append(len(getattr(attack, "energies", []) or []))
    return min(costs) if costs else 99


def _attack_damage_ceiling(pokemon: PokemonView | None) -> float:
    if pokemon is None:
        return 0.0
    data = card_data(pokemon.id)
    ceiling = 0.0
    for attack_id in getattr(data, "attacks", []) or []:
        attack = attack_data(attack_id)
        if attack is not None:
            ceiling = max(ceiling, float(getattr(attack, "damage", 0) or 0))
    return ceiling


class ResourceEvaluator:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def evaluate(self, state: AgentState) -> ResourceSnapshot:
        hand_counts = Counter(card.id for card in state.my_hand)
        search = sum(hand_counts[card_id] for card_id in _ids(self.config, "search_cards"))
        draw = sum(hand_counts[card_id] for card_id in _ids(self.config, "draw_cards"))
        energy = sum(count for card_id, count in hand_counts.items() if card_id in _ids(self.config, "energy_cards") or is_energy(card_id))
        ready_attackers = 0
        for pokemon in ([state.my_active] if state.my_active else []) + list(state.my_bench):
            if pokemon.energy_count >= _min_attack_cost(pokemon.id):
                ready_attackers += 1
        hand_quality = min(1.0, (search * 0.35) + (draw * 0.35) + (energy * 0.15) + (len(state.my_hand) * 0.03))
        once_used = float(state.supporter_played) + float(state.energy_attached) + float(state.retreated)
        return ResourceSnapshot(
            ready_attackers=ready_attackers,
            bench_space=max(0, state.my_bench_max - len(state.my_bench)),
            energy_in_play=state.my_energy_in_play,
            hand_quality=hand_quality,
            search_cards_in_hand=search,
            draw_cards_in_hand=draw,
            energy_cards_in_hand=energy,
            once_per_turn_commitment_used=once_used / 3.0,
        )


class ThreatEvaluator:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def evaluate(self, state: AgentState, opponent: OpponentBelief) -> ThreatSnapshot:
        visible_ceiling = max(
            [_attack_damage_ceiling(state.opp_active)]
            + [_attack_damage_ceiling(pokemon) for pokemon in state.opp_bench]
            + [opponent.expected_damage_next_turn]
        )
        my_active_hp = float(state.my_active.hp if state.my_active else 999)
        can_ko_active = 1.0 if visible_ceiling >= my_active_hp else min(1.0, opponent.prob_can_attack_next_turn * 0.5)
        opponent_can_win = 1.0 if state.opp_prizes_remaining <= 1 and can_ko_active > 0.75 else opponent.prob_has_gust * 0.4
        my_take_rate = max(1.0, float(state.my_energy_in_play + max(1, len(state.my_bench))) / 3.0)
        opp_take_rate = max(1.0, float(state.opp_energy_in_play + max(1, len(state.opp_bench))) / 3.0)
        return ThreatSnapshot(
            opponent_damage_ceiling=visible_ceiling,
            opponent_can_ko_active=can_ko_active,
            opponent_can_win_next_turn=min(1.0, opponent_can_win),
            opponent_gust_risk=opponent.prob_has_gust,
            turns_until_my_win_estimate=float(state.my_prizes_remaining) / my_take_rate,
            turns_until_opponent_win_estimate=float(state.opp_prizes_remaining) / opp_take_rate,
        )
