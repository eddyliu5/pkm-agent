from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .card_db import is_energy
from .state import AgentState, PokemonView


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ids(values: Iterable[int] | None) -> set[int]:
    return {int(value) for value in values or ()}


def _count_group(counts: Counter[int], card_ids: set[int]) -> int:
    return sum(int(counts.get(card_id, 0)) for card_id in card_ids)


def _pokemon_ids(pokemon: PokemonView | None) -> tuple[int, ...]:
    if pokemon is None:
        return ()
    return (int(pokemon.id),) + tuple(int(card_id) for card_id in pokemon.energy_ids) + tuple(int(card_id) for card_id in pokemon.tool_ids)


def _in_play_ids(state: AgentState) -> Counter[int]:
    counts: Counter[int] = Counter()
    for pokemon in ([state.my_active] if state.my_active else []) + list(state.my_bench):
        counts.update(_pokemon_ids(pokemon))
    return counts


def _hand_ids(state: AgentState) -> Counter[int]:
    return Counter(int(card.id) for card in state.my_hand)


def _safe_comb(n: int, k: int) -> int:
    if k < 0 or n < 0 or k > n:
        return 0
    return math.comb(n, k)


def hypergeometric_pmf(total: int, successes: int, draws: int, hits: int) -> float:
    """Probability of drawing exactly `hits` successes without replacement."""

    total = int(total)
    successes = max(0, min(int(successes), total))
    draws = max(0, min(int(draws), total))
    hits = int(hits)
    denom = _safe_comb(total, draws)
    if denom <= 0:
        return 1.0 if draws == 0 and hits == 0 else 0.0
    numerator = _safe_comb(successes, hits) * _safe_comb(total - successes, draws - hits)
    return _clamp01(float(numerator) / float(denom))


def hypergeometric_at_least_one(total: int, successes: int, draws: int) -> float:
    """Probability that at least one success is present in a hidden draw."""

    total = int(total)
    successes = max(0, min(int(successes), total))
    draws = max(0, min(int(draws), total))
    if total <= 0 or successes <= 0 or draws <= 0:
        return 0.0

    # Prize-card reasoning is a hypergeometric draw: prizes are sampled without
    # replacement from the unknown pool. The stable exact form is:
    #
    #   P(at least one target prized) = 1 - C(U - k, P) / C(U, P)
    #
    # where U is the unknown pool size, k is the number of target copies in that
    # pool, and P is the count of unknown prizes. This is preferred over treating
    # each prize independently because prize slots are negatively correlated.
    denom = _safe_comb(total, draws)
    if denom <= 0:
        return 0.0
    miss = _safe_comb(total - successes, draws) / float(denom)
    return _clamp01(1.0 - miss)


def _hypergeometric_at_least(total: int, successes: int, draws: int, hits: int) -> float:
    if hits <= 0:
        return 1.0
    total = int(total)
    successes = max(0, min(int(successes), total))
    draws = max(0, min(int(draws), total))
    return _clamp01(sum(hypergeometric_pmf(total, successes, draws, hit) for hit in range(hits, min(successes, draws) + 1)))


def _expected_prized(successes: int, total: int, unknown_prizes: int) -> float:
    if total <= 0 or successes <= 0 or unknown_prizes <= 0:
        return 0.0
    return float(unknown_prizes) * (float(successes) / float(total))


def _prob_draw_from_deck(expected_successes: float, deck_count: int, draws: int) -> float:
    deck_count = max(0, int(deck_count))
    draws = max(0, min(int(draws), deck_count))
    expected_successes = max(0.0, min(float(expected_successes), float(deck_count)))
    if deck_count <= 0 or draws <= 0 or expected_successes <= 0.0:
        return 0.0

    # After conditioning on unknown prizes we only know the expected number of a
    # card group left in deck, not the exact deck composition. For draw features
    # use a conservative expected-count approximation:
    #
    #   P(draw at least one) ~= 1 - ((D - E[k]) / D) ** n
    #
    # where D is deck size, E[k] is expected successful cards in deck, and n is
    # draw count. Exact hypergeometric math is used for prizes; this approximation
    # keeps runtime cheap and avoids sampling during normal feature extraction.
    miss = ((float(deck_count) - expected_successes) / float(deck_count)) ** float(draws)
    return _clamp01(1.0 - miss)


@dataclass(frozen=True)
class ProbabilitySnapshot:
    unknown_pool_frac: float = 0.0
    unknown_prize_frac: float = 0.0
    p_any_key_prized: float = 0.0
    p_main_attacker_prized: float = 0.0
    p_key_evolution_prized: float = 0.0
    p_energy_prized: float = 0.0
    p_search_card_prized: float = 0.0
    p_main_attacker_unavailable: float = 0.0
    p_key_evolution_unavailable: float = 0.0
    p_energy_unavailable: float = 0.0
    p_search_target_unavailable: float = 0.0
    p_energy_draw_next_card: float = 0.0
    p_energy_access_this_turn: float = 0.0
    p_energy_access_next_turn: float = 0.0
    p_line_accessible_this_turn: float = 0.0
    p_line_accessible_next_turn: float = 0.0
    p_combo_line_live: float = 0.0

    @classmethod
    def empty(cls) -> "ProbabilitySnapshot":
        return cls()

    def to_features(self) -> dict[str, float]:
        return {
            "prob_unknown_pool_frac": self.unknown_pool_frac,
            "prob_unknown_prize_frac": self.unknown_prize_frac,
            "p_any_key_prized": self.p_any_key_prized,
            "p_main_attacker_prized": self.p_main_attacker_prized,
            "p_key_evolution_prized": self.p_key_evolution_prized,
            "p_energy_prized": self.p_energy_prized,
            "p_search_card_prized": self.p_search_card_prized,
            "p_main_attacker_unavailable": self.p_main_attacker_unavailable,
            "p_key_evolution_unavailable": self.p_key_evolution_unavailable,
            "p_energy_unavailable": self.p_energy_unavailable,
            "p_search_target_unavailable": self.p_search_target_unavailable,
            "p_energy_draw_next_card": self.p_energy_draw_next_card,
            "p_energy_access_this_turn": self.p_energy_access_this_turn,
            "p_energy_access_next_turn": self.p_energy_access_next_turn,
            "p_line_accessible_this_turn": self.p_line_accessible_this_turn,
            "p_line_accessible_next_turn": self.p_line_accessible_next_turn,
            "p_combo_line_live": self.p_combo_line_live,
        }


class ProbabilityModel:
    """Cheap hidden-information probabilities for sequence ranker features."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def evaluate(
        self,
        *,
        state: AgentState,
        deck_counts: Counter[int],
        known_prized: Counter[int],
        unknown_prize_pool: Counter[int],
        unknown_prize_count: int,
    ) -> ProbabilitySnapshot:
        main_attackers = _ids(self.config.get("main_attackers"))
        key_evolutions = _ids(self.config.get("key_evolutions")) or main_attackers
        search_cards = _ids(self.config.get("search_cards"))
        energy_cards = _ids(self.config.get("energy_cards"))
        energy_cards.update(card_id for card_id in deck_counts if is_energy(card_id))
        key_cards = set().union(main_attackers, key_evolutions, search_cards, energy_cards)

        total_unknown = sum(unknown_prize_pool.values())
        unknown_prizes = max(0, min(int(unknown_prize_count), total_unknown))
        hand = _hand_ids(state)
        in_play = _in_play_ids(state)

        p_main_prized = self._prob_any_prized(main_attackers, known_prized, unknown_prize_pool, total_unknown, unknown_prizes)
        p_evolution_prized = self._prob_any_prized(key_evolutions, known_prized, unknown_prize_pool, total_unknown, unknown_prizes)
        p_energy_prized = self._prob_any_prized(energy_cards, known_prized, unknown_prize_pool, total_unknown, unknown_prizes)
        p_search_prized = self._prob_any_prized(search_cards, known_prized, unknown_prize_pool, total_unknown, unknown_prizes)
        p_any_key_prized = self._prob_any_prized(key_cards, known_prized, unknown_prize_pool, total_unknown, unknown_prizes)

        p_main_unavailable = self._prob_all_group_prized(
            main_attackers,
            deck_counts,
            known_prized,
            unknown_prize_pool,
            total_unknown,
            unknown_prizes,
            hand + in_play,
        )
        p_evolution_unavailable = self._prob_all_group_prized(
            key_evolutions,
            deck_counts,
            known_prized,
            unknown_prize_pool,
            total_unknown,
            unknown_prizes,
            hand + in_play,
        )
        p_energy_unavailable = self._prob_all_group_prized(
            energy_cards,
            deck_counts,
            known_prized,
            unknown_prize_pool,
            total_unknown,
            unknown_prizes,
            hand + in_play,
        )

        search_in_hand = _count_group(hand, search_cards)
        draw_in_hand = _count_group(hand, _ids(self.config.get("draw_cards")))
        energy_in_hand = _count_group(hand, energy_cards)
        expected_energy_in_deck = self._expected_group_in_deck(energy_cards, unknown_prize_pool, total_unknown, unknown_prizes)
        p_energy_draw_next_card = _prob_draw_from_deck(expected_energy_in_deck, state.my_deck_count, 1)
        next_turn_draws = 1 + (2 if draw_in_hand > 0 else 0)
        p_energy_draw_next_turn = _prob_draw_from_deck(expected_energy_in_deck, state.my_deck_count, next_turn_draws)

        p_search_available = 1.0 if search_in_hand > 0 else 0.0
        p_energy_access_this_turn = max(1.0 if energy_in_hand > 0 else 0.0, p_search_available * (1.0 - p_energy_unavailable))
        p_energy_access_next_turn = 1.0 - ((1.0 - p_energy_access_this_turn) * (1.0 - p_energy_draw_next_turn))

        p_main_access = self._group_access_probability(main_attackers, hand, in_play, p_main_unavailable)
        p_evolution_access = self._group_access_probability(key_evolutions, hand, in_play, p_evolution_unavailable)
        p_line_this_turn = p_main_access * p_evolution_access * p_energy_access_this_turn
        p_line_next_turn = p_main_access * p_evolution_access * p_energy_access_next_turn
        p_search_target_unavailable = max(p_main_unavailable, p_evolution_unavailable, p_energy_unavailable)

        return ProbabilitySnapshot(
            unknown_pool_frac=min(1.0, float(total_unknown) / 60.0),
            unknown_prize_frac=min(1.0, float(unknown_prizes) / 6.0),
            p_any_key_prized=p_any_key_prized,
            p_main_attacker_prized=p_main_prized,
            p_key_evolution_prized=p_evolution_prized,
            p_energy_prized=p_energy_prized,
            p_search_card_prized=p_search_prized,
            p_main_attacker_unavailable=p_main_unavailable,
            p_key_evolution_unavailable=p_evolution_unavailable,
            p_energy_unavailable=p_energy_unavailable,
            p_search_target_unavailable=p_search_target_unavailable,
            p_energy_draw_next_card=p_energy_draw_next_card,
            p_energy_access_this_turn=_clamp01(p_energy_access_this_turn),
            p_energy_access_next_turn=_clamp01(p_energy_access_next_turn),
            p_line_accessible_this_turn=_clamp01(p_line_this_turn),
            p_line_accessible_next_turn=_clamp01(p_line_next_turn),
            p_combo_line_live=_clamp01(p_line_next_turn),
        )

    def _prob_any_prized(
        self,
        card_ids: set[int],
        known_prized: Counter[int],
        unknown_prize_pool: Counter[int],
        total_unknown: int,
        unknown_prizes: int,
    ) -> float:
        if not card_ids:
            return 0.0
        if _count_group(known_prized, card_ids) > 0:
            return 1.0
        return hypergeometric_at_least_one(total_unknown, _count_group(unknown_prize_pool, card_ids), unknown_prizes)

    def _prob_all_group_prized(
        self,
        card_ids: set[int],
        deck_counts: Counter[int],
        known_prized: Counter[int],
        unknown_prize_pool: Counter[int],
        total_unknown: int,
        unknown_prizes: int,
        visible_non_prize: Counter[int],
    ) -> float:
        if not card_ids:
            return 0.0
        total_group = _count_group(deck_counts, card_ids)
        if total_group <= 0:
            return 0.0
        if _count_group(visible_non_prize, card_ids) > 0:
            return 0.0
        known = _count_group(known_prized, card_ids)
        remaining_to_prize = total_group - known
        if remaining_to_prize <= 0:
            return 1.0
        successes = _count_group(unknown_prize_pool, card_ids)
        if remaining_to_prize > successes or remaining_to_prize > unknown_prizes:
            return 0.0
        return _hypergeometric_at_least(total_unknown, successes, unknown_prizes, remaining_to_prize)

    def _expected_group_in_deck(
        self,
        card_ids: set[int],
        unknown_prize_pool: Counter[int],
        total_unknown: int,
        unknown_prizes: int,
    ) -> float:
        successes = _count_group(unknown_prize_pool, card_ids)
        return max(0.0, float(successes) - _expected_prized(successes, total_unknown, unknown_prizes))

    @staticmethod
    def _group_access_probability(card_ids: set[int], hand: Counter[int], in_play: Counter[int], p_unavailable: float) -> float:
        if not card_ids:
            return 1.0
        if _count_group(hand, card_ids) > 0 or _count_group(in_play, card_ids) > 0:
            return 1.0
        return _clamp01(1.0 - p_unavailable)
