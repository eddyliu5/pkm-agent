from __future__ import annotations

from collections import Counter

from .actions import ActionView
from .card_db import attack_data, card_data, is_energy
from .config import card_ids
from .opponent_memory import OpponentIntentSnapshot
from .state import AgentState


def _ids(values: list[int] | tuple[int, ...] | set[int]) -> set[int]:
    return {int(v) for v in values}


def _has_useful_non_terminal_action(state: AgentState, action: ActionView | None = None) -> bool:
    for other in state.legal_actions:
        if action and other.index == action.index:
            continue
        if other.is_end or other.is_attack:
            continue
        if other.option_type in {"PLAY", "ATTACH", "EVOLVE", "ABILITY", "RETREAT"}:
            return True
    return False


def _attack_damage(state: AgentState, action: ActionView) -> float:
    attack = attack_data(action.attack_id)
    damage = float(getattr(attack, "damage", 0) or 0)
    if damage <= 0 or not state.my_active or not state.opp_active:
        return damage
    active_data = card_data(state.my_active.id)
    target_data = card_data(state.opp_active.id)
    if active_data and target_data:
        if target_data.weakness == active_data.energyType:
            damage *= 2.0
        if target_data.resistance == active_data.energyType:
            damage = max(0.0, damage - 30.0)
    return damage


def _hand_counts(state: AgentState) -> Counter[int]:
    return Counter(card.id for card in state.my_hand)


def _min_attack_cost(card_id: int | None) -> int:
    data = card_data(card_id)
    costs: list[int] = []
    for attack_id in getattr(data, "attacks", []) or []:
        attack = attack_data(attack_id)
        if attack is not None:
            costs.append(len(getattr(attack, "energies", []) or []))
    return min(costs) if costs else 99


def _ready_attackers(state: AgentState, attacker_ids: set[int]) -> int:
    ready = 0
    for pokemon in ([state.my_active] if state.my_active else []) + list(state.my_bench):
        if pokemon.id in attacker_ids and pokemon.energy_count >= _min_attack_cost(pokemon.id):
            ready += 1
    return ready


def _count_in_play(state: AgentState, card_ids: set[int]) -> int:
    return sum(1 for pokemon in ([state.my_active] if state.my_active else []) + list(state.my_bench) if pokemon.id in card_ids)


def extract_action_features(
    state: AgentState,
    action: ActionView,
    config: dict,
    opponent_intent: OpponentIntentSnapshot | None = None,
) -> dict[str, float]:
    ids = card_ids(config)
    main_attackers = _ids(config.get("main_attackers", []))
    main_line = _ids(config.get("main_line", []))
    backup_attackers = _ids(config.get("backup_attackers", []))
    setup_basics = _ids(config.get("setup_basics", []))
    key_evolutions = _ids(config.get("key_evolutions", []))
    search_cards = _ids(config.get("search_cards", []))
    draw_cards = _ids(config.get("draw_cards", []))
    gust_cards = _ids(config.get("gust_cards", []))
    energy_cards = _ids(config.get("energy_cards", []))
    damage_boost_cards = _ids(config.get("damage_boost_cards", []))
    tool_cards = _ids(config.get("tool_cards", []))

    source_id = action.source_card_id
    target_id = action.target_card_id
    hand_counts = _hand_counts(state)
    attacker_ids = main_attackers | backup_attackers
    ready_attackers = _ready_attackers(state, attacker_ids)
    main_ready = _ready_attackers(state, main_attackers) > 0
    main_in_play = _count_in_play(state, main_line | main_attackers)
    setup_target_ids = set(opponent_intent.setup_target_ids) if opponent_intent is not None else set()
    must_answer_target_ids = set(opponent_intent.must_answer_target_ids) if opponent_intent is not None else set()
    evolution_window = opponent_intent.evolution_window_open if opponent_intent is not None else 0.0
    selected_opponent_card = action.source_player_index == state.opponent_index or action.target_player_index == state.opponent_index
    features: dict[str, float] = {
        "bias": 1.0,
        f"context_{action.context.lower()}": 1.0,
        f"option_{action.option_type.lower()}": 1.0,
        "prize_diff": float(max(-6, min(6, state.prize_diff))) / 6.0,
        "turn_early": 1.0 if state.turn <= 4 else 0.0,
        "turn_late": 1.0 if state.my_prizes_remaining <= 2 or state.opp_prizes_remaining <= 2 else 0.0,
        "my_energy_in_play": min(10.0, float(state.my_energy_in_play)) / 10.0,
        "opp_energy_in_play": min(10.0, float(state.opp_energy_in_play)) / 10.0,
        "hand_size": min(12.0, float(len(state.my_hand))) / 12.0,
        "bench_space": max(0.0, 5.0 - float(len(state.my_bench))) / 5.0,
        "ready_attackers": min(3.0, float(ready_attackers)) / 3.0,
        "main_attacker_ready": 1.0 if main_ready else 0.0,
        "main_line_in_play": min(3.0, float(main_in_play)) / 3.0,
    }
    if opponent_intent is not None:
        features["opponent_recent_pokemon_search"] = opponent_intent.recent_pokemon_search
        features["opponent_recent_evolution_search"] = opponent_intent.recent_evolution_search
        features["opponent_likely_setting_up_attacker"] = opponent_intent.likely_setting_up_attacker
        features["opponent_evolution_window_open"] = opponent_intent.evolution_window_open
        features["opponent_has_setup_targets"] = 1.0 if opponent_intent.has_setup_targets else 0.0

    if action.option_type in {"YES", "NO", "NUMBER"}:
        features["simple_selection"] = 1.0
    if action.option_type == "YES":
        features["choose_yes"] = 1.0
    if action.option_type == "NO":
        features["choose_no"] = 1.0
    if action.option_type == "NUMBER":
        features["number_value"] = min(10.0, float(getattr(action.option, "number", 0) or 0)) / 10.0

    if action.is_end:
        features["end_turn"] = 1.0
        if _has_useful_non_terminal_action(state, action):
            features["end_with_useful_action"] = 1.0

    if action.is_attack:
        damage = _attack_damage(state, action)
        active_hp = float(state.opp_active.hp if state.opp_active else 1)
        features["attack"] = 1.0
        features["attack_damage_frac"] = min(2.0, damage / max(1.0, active_hp))
        if state.my_active and state.my_active.id in main_attackers:
            features["attack_with_main"] = 1.0
        if state.my_active and state.my_active.id in backup_attackers:
            features["attack_with_backup"] = 1.0
        if state.opp_active and damage >= state.opp_active.hp:
            features["attack_takes_ko"] = 1.0
            features["attack_prize_value"] = float(state.opp_active.prize_value)
            if state.opp_active.prize_value >= state.my_prizes_remaining:
                features["attack_wins_game"] = 1.0
            if state.opp_active.id in setup_target_ids:
                features["attack_denies_evolution"] = max(0.35, evolution_window)
            if state.opp_active.id in must_answer_target_ids:
                features["attack_removes_must_answer"] = 1.0
        if _has_useful_non_terminal_action(state, action) and "attack_wins_game" not in features:
            features["attack_before_setup"] = 1.0

    if action.is_play:
        features["play_card"] = 1.0
        if source_id in setup_basics:
            features["play_setup_basic"] = 1.0
        if source_id in search_cards:
            features["play_search"] = 1.0
        if source_id in draw_cards:
            features["play_draw"] = 1.0
        if source_id in gust_cards:
            features["play_gust"] = 1.0
            if ready_attackers <= 0:
                features["play_gust_without_ready_attacker"] = 1.0
            if state.my_prizes_remaining > 2 and state.opp_active and state.opp_active.prize_value < state.my_prizes_remaining:
                features["play_gust_before_close"] = 1.0
            if setup_target_ids and ready_attackers > 0:
                features["play_gust_to_deny_evolution"] = max(0.35, evolution_window)
        if source_id in damage_boost_cards:
            features["play_damage_boost"] = 1.0
        if source_id in tool_cards:
            features["play_tool"] = 1.0
        if source_id in main_line:
            features["play_main_line"] = 1.0
            if main_ready or main_in_play >= 3:
                features["play_main_line_after_setup"] = 1.0
        if source_id in backup_attackers:
            features["play_backup_attacker"] = 1.0
            if main_ready and len(state.my_bench) >= 3:
                features["play_backup_after_setup"] = 1.0
        if source_id in search_cards and main_ready:
            features["play_search_after_main_ready"] = 1.0
        if source_id in draw_cards and main_ready and len(state.my_hand) >= 5:
            features["play_draw_with_stable_hand"] = 1.0

    if action.is_attach:
        features["attach"] = 1.0
        if source_id in energy_cards or is_energy(source_id):
            features["attach_energy"] = 1.0
        if target_id in main_line or target_id in main_attackers:
            features["attach_to_main_line"] = 1.0
        if target_id in backup_attackers:
            features["attach_to_backup"] = 1.0
        if source_id in tool_cards:
            features["attach_tool"] = 1.0
        if target_id not in main_line and target_id not in backup_attackers and source_id in energy_cards:
            features["attach_energy_to_low_value"] = 1.0
        if state.energy_attached and source_id in energy_cards:
            features["attach_after_energy_used"] = 1.0

    if action.is_evolve:
        features["evolve"] = 1.0
        if source_id in key_evolutions:
            features["evolve_key"] = 1.0
        if source_id in main_attackers:
            features["evolve_main_attacker"] = 1.0
            if main_ready:
                features["evolve_extra_main_after_ready"] = 1.0
        if source_id in backup_attackers:
            features["evolve_backup_attacker"] = 1.0
            if main_ready and ready_attackers >= 1:
                features["evolve_backup_after_ready"] = 1.0

    if action.is_ability:
        features["ability"] = 1.0
        if source_id == ids.get("lunatone"):
            features["ability_draw"] = 1.0

    if action.option_type == "RETREAT":
        features["retreat"] = 1.0
        if state.my_active and state.my_active.id not in main_attackers:
            features["retreat_non_main_active"] = 1.0

    if action.option_type == "CARD":
        if selected_opponent_card and source_id in setup_target_ids:
            features["select_opponent_setup_target"] = 1.0
            features["select_evolution_denial_target"] = max(0.35, evolution_window)
        if selected_opponent_card and source_id in must_answer_target_ids:
            features["select_must_answer_target"] = 1.0
        if source_id in main_line:
            features["select_main_line"] = 1.0
        if source_id in backup_attackers:
            features["select_backup"] = 1.0
        if source_id in energy_cards:
            features["select_energy"] = 1.0
        if source_id in search_cards:
            features["select_search_card"] = 1.0
        if source_id in draw_cards:
            features["select_draw_card"] = 1.0
        if source_id in main_line and main_ready:
            features["select_main_line_after_ready"] = 1.0
        if source_id is not None and hand_counts[source_id] >= 2:
            features["select_duplicate_in_hand"] = 1.0

    return features


def combine_feature_dicts(feature_dicts: list[dict[str, float]]) -> dict[str, float]:
    combined: dict[str, float] = {}
    if not feature_dicts:
        return combined
    for features in feature_dicts:
        for key, value in features.items():
            combined[key] = combined.get(key, 0.0) + value
    scale = 1.0 / float(len(feature_dicts))
    return {key: value * scale for key, value in combined.items()}


def default_weights() -> dict[str, float]:
    return {
        "bias": 0.0,
        "simple_selection": 10.0,
        "choose_yes": 2.0,
        "choose_no": -1.0,
        "number_value": 20.0,
        "end_turn": -80.0,
        "end_with_useful_action": -1000.0,
        "attack": 90.0,
        "attack_damage_frac": 250.0,
        "attack_with_main": 80.0,
        "attack_with_backup": 25.0,
        "attack_takes_ko": 700.0,
        "attack_prize_value": 240.0,
        "attack_wins_game": 10000.0,
        "attack_denies_evolution": 520.0,
        "attack_removes_must_answer": 320.0,
        "attack_before_setup": -220.0,
        "play_card": 10.0,
        "play_setup_basic": 150.0,
        "play_search": 260.0,
        "play_draw": 220.0,
        "play_gust": 180.0,
        "play_gust_to_deny_evolution": 430.0,
        "play_damage_boost": 80.0,
        "play_tool": 80.0,
        "play_main_line": 160.0,
        "play_backup_attacker": 65.0,
        "attach": 25.0,
        "attach_energy": 150.0,
        "attach_to_main_line": 260.0,
        "attach_to_backup": 80.0,
        "attach_tool": 120.0,
        "attach_energy_to_low_value": -220.0,
        "attach_after_energy_used": -120.0,
        "evolve": 120.0,
        "evolve_key": 250.0,
        "evolve_main_attacker": 420.0,
        "evolve_backup_attacker": 150.0,
        "ability": 80.0,
        "ability_draw": 360.0,
        "retreat": -30.0,
        "retreat_non_main_active": 140.0,
        "select_main_line": 180.0,
        "select_backup": 80.0,
        "select_energy": 80.0,
        "select_search_card": 65.0,
        "select_draw_card": 90.0,
        "select_duplicate_in_hand": -40.0,
        "select_opponent_setup_target": 340.0,
        "select_evolution_denial_target": 520.0,
        "select_must_answer_target": 260.0,
        "turn_early": 10.0,
        "turn_late": 5.0,
        "prize_diff": 20.0,
        "my_energy_in_play": 30.0,
        "opp_energy_in_play": -10.0,
        "bench_space": 20.0,
        "hand_size": 5.0,
        "ready_attackers": 180.0,
        "main_attacker_ready": 240.0,
        "main_line_in_play": 40.0,
        "play_gust_without_ready_attacker": -600.0,
        "play_gust_before_close": -240.0,
        "play_main_line_after_setup": -180.0,
        "play_backup_after_setup": -80.0,
        "play_search_after_main_ready": -120.0,
        "play_draw_with_stable_hand": -90.0,
        "evolve_extra_main_after_ready": -220.0,
        "evolve_backup_after_ready": -100.0,
        "select_main_line_after_ready": -90.0
    }


def score_features(features: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights.get(key, 0.0) * value for key, value in features.items())
