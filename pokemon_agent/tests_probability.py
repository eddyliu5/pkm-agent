from __future__ import annotations

import math
from collections import Counter

from .probability import ProbabilityModel, hypergeometric_at_least_one, hypergeometric_pmf
from .state import AgentState, CardView

ENERGY = 9001
MAIN = 9101
EVOLUTION = 9102
SEARCH = 9201
DRAW = 9301
FILLER = 9909


def _state(*, deck_count: int, hand: tuple[int, ...] = ()) -> AgentState:
    return AgentState(
        turn=1,
        turn_action_count=0,
        player_index=0,
        opponent_index=1,
        first_player=0,
        result=-1,
        supporter_played=False,
        stadium_played=False,
        energy_attached=False,
        retreated=False,
        stadium_id=None,
        stadium_name=None,
        my_active=None,
        my_bench=(),
        my_hand=tuple(CardView(id=card_id, name=str(card_id)) for card_id in hand),
        my_discard=(),
        my_prizes_remaining=2,
        my_deck_count=deck_count,
        my_status=(),
        opp_active=None,
        opp_bench=(),
        opp_hand_count=0,
        opp_discard=(),
        opp_prizes_remaining=6,
        opp_deck_count=60,
        opp_status=(),
        my_bench_max=5,
        opp_bench_max=5,
        select_type="NONE",
        select_context="NONE",
        select_min_count=0,
        select_max_count=0,
        select_option_count=0,
    )


def _assert_close(actual: float, expected: float, *, tolerance: float = 1e-12) -> None:
    assert abs(actual - expected) <= tolerance, f"{actual} != {expected}"


def test_hypergeometric_helpers_are_exact() -> None:
    _assert_close(hypergeometric_pmf(10, 3, 2, 1), (3 * 7) / math.comb(10, 2))
    _assert_close(hypergeometric_at_least_one(60, 4, 6), 1.0 - (math.comb(56, 6) / math.comb(60, 6)))


def test_known_deck_counts_use_exact_draw_odds() -> None:
    snapshot = ProbabilityModel({"energy_cards": [ENERGY]}).evaluate(
        state=_state(deck_count=10),
        deck_counts=Counter({ENERGY: 4, FILLER: 56}),
        known_prized=Counter(),
        unknown_prize_pool=Counter({ENERGY: 4, FILLER: 6}),
        unknown_prize_count=2,
        known_deck_counts=Counter({ENERGY: 2, FILLER: 8}),
    )

    _assert_close(snapshot.p_energy_in_deck, 1.0)
    _assert_close(snapshot.p_energy_draw_next_card, hypergeometric_at_least_one(10, 2, 1))
    _assert_close(snapshot.p_energy_access_next_turn, hypergeometric_at_least_one(10, 2, 1))


def test_known_deck_counts_override_hidden_pool_for_search_targets() -> None:
    snapshot = ProbabilityModel({"energy_cards": [ENERGY], "search_cards": [SEARCH]}).evaluate(
        state=_state(deck_count=10, hand=(SEARCH,)),
        deck_counts=Counter({ENERGY: 4, FILLER: 55, SEARCH: 1}),
        known_prized=Counter(),
        unknown_prize_pool=Counter({ENERGY: 4, FILLER: 6}),
        unknown_prize_count=2,
        known_deck_counts=Counter({FILLER: 10}),
    )

    _assert_close(snapshot.p_energy_in_deck, 0.0)
    _assert_close(snapshot.p_energy_draw_next_card, 0.0)
    _assert_close(snapshot.p_energy_access_this_turn, 0.0)
    _assert_close(snapshot.p_search_target_unavailable, 1.0)


def test_unknown_pool_estimates_deck_presence_not_just_not_prized() -> None:
    snapshot = ProbabilityModel({"main_attackers": [MAIN], "energy_cards": [ENERGY]}).evaluate(
        state=_state(deck_count=8),
        deck_counts=Counter({MAIN: 2, ENERGY: 2, FILLER: 56}),
        known_prized=Counter(),
        unknown_prize_pool=Counter({MAIN: 2, ENERGY: 2, FILLER: 6}),
        unknown_prize_count=2,
    )

    p_both_attackers_prized = hypergeometric_pmf(10, 2, 2, 2)
    _assert_close(snapshot.p_main_attacker_in_deck, 1.0 - p_both_attackers_prized)
    _assert_close(snapshot.p_main_attacker_unavailable, p_both_attackers_prized)


def test_missing_hidden_copies_are_not_treated_as_accessible() -> None:
    snapshot = ProbabilityModel({"main_attackers": [MAIN], "energy_cards": [ENERGY], "search_cards": [SEARCH]}).evaluate(
        state=_state(deck_count=8, hand=(SEARCH,)),
        deck_counts=Counter({MAIN: 1, ENERGY: 2, FILLER: 56, SEARCH: 1}),
        known_prized=Counter(),
        unknown_prize_pool=Counter({ENERGY: 2, FILLER: 6}),
        unknown_prize_count=2,
    )

    _assert_close(snapshot.p_main_attacker_in_deck, 0.0)
    _assert_close(snapshot.p_line_accessible_this_turn, 0.0)
    _assert_close(snapshot.p_search_target_unavailable, 1.0)


def test_probability_features_are_bounded() -> None:
    snapshot = ProbabilityModel(
        {
            "main_attackers": [MAIN],
            "key_evolutions": [EVOLUTION],
            "energy_cards": [ENERGY],
            "search_cards": [SEARCH],
            "draw_cards": [DRAW],
        }
    ).evaluate(
        state=_state(deck_count=9, hand=(SEARCH, DRAW)),
        deck_counts=Counter({MAIN: 2, EVOLUTION: 1, ENERGY: 4, SEARCH: 1, DRAW: 1, FILLER: 51}),
        known_prized=Counter(),
        unknown_prize_pool=Counter({MAIN: 2, EVOLUTION: 1, ENERGY: 4, SEARCH: 1, DRAW: 1}),
        unknown_prize_count=2,
    )

    assert all(0.0 <= value <= 1.0 for value in snapshot.to_features().values())


def test_empty_config_groups_do_not_create_search_risk() -> None:
    snapshot = ProbabilityModel({}).evaluate(
        state=_state(deck_count=8),
        deck_counts=Counter({FILLER: 60}),
        known_prized=Counter(),
        unknown_prize_pool=Counter({FILLER: 10}),
        unknown_prize_count=2,
    )

    _assert_close(snapshot.p_search_target_unavailable, 0.0)


def main() -> None:
    test_hypergeometric_helpers_are_exact()
    test_known_deck_counts_use_exact_draw_odds()
    test_known_deck_counts_override_hidden_pool_for_search_targets()
    test_unknown_pool_estimates_deck_presence_not_just_not_prized()
    test_missing_hidden_copies_are_not_treated_as_accessible()
    test_probability_features_are_bounded()
    test_empty_config_groups_do_not_create_search_risk()
    print("probability tests passed")


if __name__ == "__main__":
    main()
