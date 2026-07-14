from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .actions import ActionView
from .config import load_config, read_deck
from .decision_engine import DecisionEngine
from .sequence.featurizer import (
    _afterstate_features,
    _candidate_metadata_features,
    _opponent_archetype_features,
    _raw_sequence_interaction_features,
)
from .sequence.types import CandidateSequence, SelectionStep
from .state import AgentState, PokemonView
from .xgb_root_ranker import RootGroupInfo, iter_groups, load_root_ranker, self_archetype_group_weights

from .cg_loader import ensure_cg_path

ensure_cg_path()

from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_start  # noqa: E402


def _pokemon(card_id: int, *, hp: int = 100, max_hp: int = 100, energy: int = 0) -> PokemonView:
    return PokemonView(
        id=card_id,
        name=str(card_id),
        hp=hp,
        max_hp=max_hp,
        energy_count=energy,
        energy_ids=tuple(range(energy)),
        tool_ids=(),
        prize_value=1,
        is_ex=False,
        is_mega_ex=False,
        is_basic=True,
        is_stage1=False,
        is_stage2=False,
        appears_this_turn=False,
    )


def _state() -> AgentState:
    return AgentState(
        turn=3,
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
        my_active=_pokemon(677, energy=1),
        my_bench=(_pokemon(1030),),
        my_hand=(),
        my_discard=(),
        my_prizes_remaining=6,
        my_deck_count=48,
        my_status=(),
        opp_active=_pokemon(860),
        opp_bench=(),
        opp_hand_count=5,
        opp_discard=(),
        opp_prizes_remaining=6,
        opp_deck_count=47,
        opp_status=(),
        my_bench_max=5,
        opp_bench_max=5,
        select_type="MAIN",
        select_context="MAIN",
        select_min_count=1,
        select_max_count=1,
        select_option_count=4,
        legal_actions=(),
    )


def _action(index: int, option_type: str, *, source_card_id: int | None = None) -> ActionView:
    return ActionView(
        index=index,
        option=SimpleNamespace(number=None),
        select_type="MAIN",
        context="MAIN",
        option_type=option_type,
        source_player_index=0,
        source_card_id=source_card_id,
        source_name=None,
        target_player_index=None,
        target_card_id=None,
        target_name=None,
        attack_id=None,
        attack_name=None,
        label=f"{option_type.lower()} {index}",
    )


def _step(action: ActionView, features: dict[str, float] | None = None) -> SelectionStep:
    return SelectionStep(
        selected_indices=(action.index,),
        actions=(action,),
        select_type="MAIN",
        context="MAIN",
        features=features or {},
        score=0.0,
        labels=(action.label,),
    )


def test_candidate_generation_and_completion_metadata_are_exact() -> None:
    root = _state()
    turn_boundary_obs = SimpleNamespace(current=SimpleNamespace(turn=4, yourIndex=1))
    partial = CandidateSequence(
        steps=(_step(_action(0, "PLAY")),),
        observation=turn_boundary_obs,
        state=root,
        terminal=False,
        used_search=True,
        generation_tags=("simulated", "sample_0"),
        sim_result="partial",
    )
    features = _candidate_metadata_features(root, partial)
    assert features["candidate_is_simulated"] == 1.0
    assert features["candidate_is_partial"] == 1.0
    assert features["candidate_reaches_turn_boundary"] == 1.0
    assert features["candidate_is_root_coverage"] == 0.0
    assert features["candidate_is_fallback"] == 0.0

    coverage = replace(
        partial,
        observation=SimpleNamespace(current=SimpleNamespace(turn=3, yourIndex=0)),
        terminal=True,
        used_search=False,
        generation_tags=("root_coverage",),
        sim_result="coverage",
    )
    coverage_features = _candidate_metadata_features(root, coverage)
    assert coverage_features["candidate_is_root_coverage"] == 1.0
    assert coverage_features["candidate_is_simulated"] == 0.0
    assert coverage_features["candidate_is_partial"] == 0.0
    assert coverage_features["candidate_reaches_turn_boundary"] == 0.0

    fallback = replace(coverage, generation_tags=("fallback_linear",), sim_result="fallback")
    assert _candidate_metadata_features(root, fallback)["candidate_is_fallback"] == 1.0


def test_raw_sequence_interactions_are_not_step_averaged() -> None:
    root = _state()
    gust = _action(0, "PLAY", source_card_id=1001)
    retreat = _action(1, "RETREAT")
    switch = _action(2, "PLAY", source_card_id=1002)
    final = replace(root, my_active=_pokemon(1030), my_bench=(_pokemon(677, energy=1),))
    sequence = CandidateSequence(
        steps=(_step(gust, {"play_gust": 1.0}), _step(retreat), _step(switch)),
        observation=SimpleNamespace(current=SimpleNamespace(turn=3, yourIndex=0)),
        state=final,
        terminal=True,
        used_search=True,
    )
    features = _raw_sequence_interaction_features(
        root,
        sequence,
        {"gust_cards": [1001], "switch_cards": [1002]},
    )
    assert features["seq_action_count_raw"] == 3.0
    assert features["seq_play_count_raw"] == 2.0
    assert features["seq_gust_count_raw"] == 1.0
    assert features["seq_switch_count_raw"] == 2.0
    assert features["seq_gust_without_attack"] == 1.0
    assert features["seq_retreat_plus_switch_card"] == 1.0
    assert features["seq_repeated_switching"] == 1.0
    assert features["seq_resource_churn_count_raw"] == 1.0
    assert features["seq_active_changed"] == 1.0

    with_attack = replace(sequence, steps=(*sequence.steps, _step(_action(3, "ATTACK"))))
    attacked = _raw_sequence_interaction_features(root, with_attack, {"gust_cards": [1001]})
    assert attacked["seq_gust_without_attack"] == 0.0
    assert attacked["seq_gust_then_attack"] == 1.0
    assert attacked["seq_attack_count_raw"] == 1.0

    prior_retreat = _raw_sequence_interaction_features(
        replace(root, retreated=True),
        replace(sequence, steps=(_step(switch),)),
        {"switch_cards": [1002]},
    )
    assert prior_retreat["seq_switch_after_prior_retreat"] == 1.0
    assert prior_retreat["seq_repeated_switching"] == 1.0
    assert prior_retreat["seq_resource_churn_count_raw"] == 1.0


def test_afterstate_and_opponent_posterior_features() -> None:
    state = replace(_state(), supporter_played=True, energy_attached=True, retreated=True)
    after = _afterstate_features(state)
    assert after["after_my_active_hp_frac"] == 1.0
    assert after["after_my_bench_count_raw"] == 1.0
    assert after["after_my_energy_in_play_raw"] == 1.0
    assert after["after_my_ready_attackers_raw"] == 1.0
    assert after["after_opp_ready_attackers_raw"] == 0.0
    assert after["after_once_per_turn_commitments_raw"] == 3.0

    opponent = SimpleNamespace(
        archetype_probs={"Mega Lucario": 0.7, "Control/Box": 0.3},
        top_archetype="Mega Lucario",
    )
    posterior = _opponent_archetype_features(opponent)
    assert posterior["opponent_archetype_posterior_mega_lucario"] == 0.7
    assert posterior["opponent_archetype_posterior_control_box"] == 0.3
    assert posterior["opponent_top_archetype_mega_lucario"] == 1.0
    assert posterior["opponent_top_archetype_posterior"] == 0.7
    assert math.isclose(posterior["opponent_archetype_posterior_margin"], 0.4)


def test_v1_compact_xgb_ignores_v2_foundation_features() -> None:
    model = load_root_ranker(Path(__file__).resolve().parent / "models" / "xgb_root_v1.json")
    additions = {
        "candidate_is_simulated": 1.0,
        "seq_gust_without_attack": 1.0,
        "after_my_ready_attackers_raw": 2.0,
        "opponent_archetype_posterior_mega_lucario": 0.8,
    }
    assert not set(additions).intersection(model.feature_names)
    baseline, augmented = model.score_batch([{}, additions])
    assert baseline == augmented


def test_full_evaluator_emits_v2_foundations() -> None:
    config = load_config()
    deck = read_deck(config=config)
    obs_dict, _ = battle_start(deck, deck)
    try:
        engine = DecisionEngine(deck=deck, config=config)
        observed = engine.observe(to_observation_class(obs_dict))
        evaluated = engine.evaluate(engine.plan(observed))
        assert evaluated.best is not None
        features = evaluated.best.features
        assert "candidate_is_simulated" in features
        assert "candidate_is_root_coverage" in features
        assert "seq_action_count_raw" in features
        assert "seq_resource_action_count_raw" in features
        assert "after_my_ready_attackers_raw" in features
        assert "after_my_bench_space_frac" in features
        assert "opponent_top_archetype_posterior" in features
        assert any(key.startswith("opponent_archetype_posterior_") for key in features)
        assert "xgb_score" not in features
        assert "mlp_score" not in features
    finally:
        battle_finish()


def test_reused_replay_step_indices_still_form_distinct_groups(tmp_path: Path) -> None:
    candidate = {"selected": [0], "features": {"bias": 1.0}, "score": 0.0}
    alternative = {"selected": [1], "features": {"bias": 1.0}, "score": -1.0}
    records = [
        {
            "episode_id": "episode",
            "root_ranker_mode": True,
            "player_index": 0,
            "step_index": 1,
            "self_archetype": "alpha",
            "matched": True,
            "label_index": 0,
            "candidates": [candidate, alternative],
        },
        {
            "episode_id": "episode",
            "root_ranker_mode": True,
            "player_index": 0,
            "step_index": 1,
            "self_archetype": "alpha",
            "matched": True,
            "label_index": 1,
            "candidates": [candidate, alternative],
        },
    ]
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    groups = list(iter_groups([labels]))
    assert len(groups) == 2
    assert groups[0].group_id != groups[1].group_id


def test_archetype_query_weights_are_normalized_and_help_rare_groups() -> None:
    def group(index: int, archetype: str) -> RootGroupInfo:
        return RootGroupInfo("episode", str(index), archetype, 0, 2, (0.0, 0.0))

    groups = [*(group(index, "common") for index in range(9)), group(9, "rare")]
    weights = self_archetype_group_weights(groups)
    assert math.isclose(sum(weights) / len(weights), 1.0)
    assert weights[-1] > weights[0]
    assert all(0.25 <= weight <= 4.0 for weight in weights)
