from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import inspect
import json
import math
import tempfile
import tarfile
import subprocess
import sys
from pathlib import Path

from .belief import PrizeTracker
from .actions import ActionView, legal_actions, sanitize_selection
from .config import load_config, read_deck
from .acting.guardrails import FallbackGuardrail
from .decision_engine import DecisionEngine
from .decision_types import ScoreBreakdown, ScoredCandidate
from .deck_db import archetype_gameplan_profiles, deck_gameplan_profile, iter_decks, load_deck_db, rank_decks_by_observed_cards, validate_deck
from .engine import run_episode
from .eval import _opponent_deck_pool
from .features import default_weights, extract_action_features, score_features
from .mlp_ranker import MLPSequenceRanker, pretrain_mlp_ranker, torch_available
from .policy import HeuristicPolicy, RandomPolicy, SequencePolicy, make_policy
from .opponent_memory import OpponentMemory
from .probability import ProbabilityModel, hypergeometric_at_least_one, hypergeometric_pmf
from .planning.pruning import SequencePruner
from .replay_pipeline import (
    build_candidate_coverage_dataset,
    build_label_dataset,
    compact_label_dataset,
    pretrain_sequence_scorer,
    rl_self_play_mlp,
)
from .runtime import RuntimeAdapter
from .sequence import CandidateSequence, SearchTree, SelectionStep, SequenceSearchConfig
from .sequence.types import EvaluatedSequenceView
from .state import AgentState, CardView, PokemonView, build_agent_state
from .submission import build_submission

from .cg_loader import ensure_cg_path

ensure_cg_path()

from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_start  # noqa: E402


def test_action_state_reward_smoke() -> None:
    config = load_config()
    deck = read_deck(config=config)
    obs_dict, _ = battle_start(deck, deck)
    try:
        obs = to_observation_class(obs_dict)
        actions = legal_actions(obs)
        assert actions
        state = build_agent_state(obs, actions=actions)
        assert state.player_index in (0, 1)
        assert state.my_deck_count == 60
        selected, invalid = sanitize_selection(obs, [actions[0].index])
        assert not invalid
        assert len(selected) == obs.select.maxCount == 1
        features = extract_action_features(state, actions[0], config)
        assert "bias" in features
    finally:
        battle_finish()


def test_episode_smoke() -> None:
    config = load_config()
    deck = read_deck(config=config)
    result = run_episode(RandomPolicy(seed=1), RandomPolicy(seed=2), deck, deck, config=config, max_steps=500)
    assert result.steps > 0
    assert result.invalid_actions == 0
    assert result.winner in (-1, 0, 1, 2)


def test_heuristic_episode_smoke() -> None:
    config = load_config()
    deck = read_deck(config=config)
    result = run_episode(HeuristicPolicy(config=config), RandomPolicy(seed=3), deck, deck, config=config, max_steps=500)
    assert result.steps > 0
    assert result.invalid_actions == 0


def test_submission_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "submission.tar.gz"
        path = build_submission(output=str(output))
        assert path.exists()
        assert path.stat().st_size > 0
        with tarfile.open(path, "r:gz") as tar:
            names = set(tar.getnames())
        assert "data/deck_db/decks.json" in names
        assert "configs/starmie_froslass.json" in names
        extract_dir = Path(tmp) / "extract"
        extract_dir.mkdir()
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")
        check = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; sys.path.insert(0, sys.argv[1]); "
                    "import main; "
                    "from pokemon_agent.config import load_config, read_deck; "
                    "cfg = load_config(); "
                    "assert cfg['deck_path'] == 'deck.csv'; "
                    "assert len(read_deck(config=cfg)) == 60; "
                    "assert callable(main.agent)"
                ),
                str(extract_dir),
            ],
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
        )
        assert check.returncode == 0, check.stderr


def test_deck_db_smoke() -> None:
    db = load_deck_db()
    decks = list(iter_decks(db))
    assert len(decks) >= 30
    assert "mega_lucario" in {deck["archetype"] for deck in decks}
    for deck in decks:
        assert not validate_deck(deck), deck["id"]

    ranked = rank_decks_by_observed_cards([677, 678, 1142, 6], db, top_n=3)
    assert ranked
    assert ranked[0]["archetype"] == "mega_lucario"


def test_deck_gameplan_profiles_smoke() -> None:
    db = load_deck_db()
    profiles = archetype_gameplan_profiles(db)
    assert len(profiles) >= 20
    assert "mega_lucario" in profiles
    lucario = profiles["mega_lucario"]
    assert lucario["plan_type"] in {"pressure", "setup_big_attacker", "hybrid", "control"}
    assert lucario["main_attacker_ids"]
    assert lucario["average_first_attack_turn"] > 0
    assert lucario["average_game_end_turn"] >= lucario["average_first_attack_turn"]
    for deck in iter_decks(db):
        profile = deck_gameplan_profile(deck, db)
        assert profile.average_first_attack_turn > 0
        assert profile.average_game_end_turn >= profile.average_first_attack_turn
        assert profile.plan_type in {"pressure", "setup_big_attacker", "hybrid", "control"}


def test_deck_db_opponent_pool_smoke() -> None:
    pool = _opponent_deck_pool("deck_db")
    assert len(pool) >= 30
    deck_id, archetype, deck = pool[0]
    assert deck_id
    assert archetype
    assert len(deck) == 60

    top_pool = _opponent_deck_pool("top_winrate")
    assert top_pool
    top_deck_id, top_archetype, top_deck = top_pool[0]
    assert top_deck_id
    assert "wr=" in top_archetype
    assert len(top_deck) == 60


def _fake_card(card_id: int, player_index: int = 0):
    return SimpleNamespace(id=int(card_id), serial=int(card_id), playerIndex=player_index)


def _fake_pokemon(card_id: int, *, energy_count: int = 0, hp: int = 100, max_hp: int = 100) -> PokemonView:
    return PokemonView(
        id=int(card_id),
        name=f"card {card_id}",
        hp=hp,
        max_hp=max_hp,
        energy_count=energy_count,
        energy_ids=(),
        tool_ids=(),
        prize_value=1,
        is_ex=False,
        is_mega_ex=False,
        is_basic=True,
        is_stage1=False,
        is_stage2=False,
        appears_this_turn=False,
    )


def _memory_state(*, turn: int, opp_discard: tuple[CardView, ...] = (), opp_bench: tuple[PokemonView, ...] = ()) -> AgentState:
    return AgentState(
        turn=turn,
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
        my_hand=(),
        my_discard=(),
        my_prizes_remaining=6,
        my_deck_count=50,
        my_status=(),
        opp_active=None,
        opp_bench=opp_bench,
        opp_hand_count=5,
        opp_discard=opp_discard,
        opp_prizes_remaining=6,
        opp_deck_count=50,
        opp_status=(),
        my_bench_max=5,
        opp_bench_max=5,
        select_type="MAIN",
        select_context="MAIN",
        select_min_count=1,
        select_max_count=1,
        select_option_count=1,
        legal_actions=(),
    )


def test_opponent_memory_setup_tag_clears_after_evolution() -> None:
    memory = OpponentMemory()
    search_card = CardView(id=1102, name="Dusk Ball", serial=1)
    riolu = _fake_pokemon(677)
    state = _memory_state(turn=2, opp_discard=(search_card,), opp_bench=(riolu,))
    memory.update(SimpleNamespace(logs=[]), state)
    intent = memory.snapshot(state, main_attacker_ids=(678,))
    assert 677 in intent.setup_target_ids
    assert intent.evolution_window_open > 0

    lucario = _fake_pokemon(678)
    state_after_evolve = _memory_state(turn=2, opp_discard=(search_card,), opp_bench=(riolu, lucario))
    memory.update(SimpleNamespace(logs=[]), state_after_evolve)
    resolved = memory.snapshot(state_after_evolve, main_attacker_ids=(678,))
    assert 678 in resolved.played_evolution_ids
    assert 677 not in resolved.setup_target_ids
    assert 678 in resolved.must_answer_target_ids


def test_probability_model_hypergeometric_features() -> None:
    assert hypergeometric_pmf(10, 3, 2, 1) == (3 * 7) / 45
    assert hypergeometric_at_least_one(60, 4, 6) == 1.0 - (math.comb(56, 6) / math.comb(60, 6))

    config = {
        "main_attackers": [101],
        "key_evolutions": [102],
        "energy_cards": [1],
        "search_cards": [201],
        "draw_cards": [301],
    }
    state = _memory_state(turn=1)
    unknown_pool = Counter({101: 2, 102: 1, 1: 4, 201: 1, 301: 1})
    snapshot = ProbabilityModel(config).evaluate(
        state=state,
        deck_counts=Counter(unknown_pool),
        known_prized=Counter(),
        unknown_prize_pool=Counter(unknown_pool),
        unknown_prize_count=2,
    )
    features = snapshot.to_features()
    assert 0.0 < snapshot.p_main_attacker_prized < 1.0
    assert 0.0 < snapshot.p_main_attacker_unavailable < snapshot.p_main_attacker_prized
    assert snapshot.p_energy_access_next_turn >= snapshot.p_energy_access_this_turn
    assert "p_combo_line_live" in features
    assert all(0.0 <= value <= 1.0 for value in features.values())


def test_search_tree_reconstructs_leaf_sequence() -> None:
    step_a = SelectionStep((1,), (), "MAIN", "MAIN", {"a": 1.0}, 10.0, ("a",))
    step_b = SelectionStep((2,), (), "MAIN", "MAIN", {"b": 1.0}, 20.0, ("b",))
    tree = SearchTree()
    root = tree.add_node(observation="root", state=None, search_id=1, used_search=True)
    child = tree.add_node(
        observation="child",
        state=None,
        search_id=2,
        parent_id=root,
        incoming_step=step_a,
        depth=1,
        cumulative_score=10.0,
        used_search=True,
    )
    leaf = tree.add_node(
        observation="leaf",
        state=None,
        search_id=3,
        parent_id=child,
        incoming_step=step_b,
        depth=2,
        cumulative_score=30.0,
        terminal=True,
        used_search=True,
    )
    sequence = tree.to_candidate(leaf)
    assert sequence.first_selection == [1]
    assert tuple(step.labels[0] for step in sequence.steps) == ("a", "b")
    assert sequence.score == 30.0


def _fake_action_view(
    index: int,
    *,
    option_type: str = "YES",
    label: str = "choose yes",
    source_card_id: int | None = None,
    attack_id: int | None = None,
) -> ActionView:
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
        attack_id=attack_id,
        attack_name=None,
        label=label,
    )


def _fake_observed_for_warm_start(state: AgentState, actions: tuple[ActionView, ...]):
    option_count = max((action.index for action in actions), default=-1) + 1
    observation = SimpleNamespace(select=SimpleNamespace(option=[object() for _ in range(option_count)], minCount=1, maxCount=1))
    return SimpleNamespace(
        observation=observation,
        actions=actions,
        state=replace(state, legal_actions=actions, select_option_count=option_count),
        belief=SimpleNamespace(opponent=SimpleNamespace(intent=None)),
    )


def test_empty_board_safety_features_prefer_benching_basic() -> None:
    play_basic = _fake_action_view(1, option_type="PLAY", label="play Snorunt", source_card_id=860)
    attack = _fake_action_view(2, option_type="ATTACK", label="attack")
    end_turn = _fake_action_view(3, option_type="END", label="end turn")
    state = replace(
        _memory_state(turn=3),
        my_active=_fake_pokemon(1031, hp=60, max_hp=180),
        my_hand=(CardView(id=860, name="Snorunt"),),
        legal_actions=(play_basic, attack, end_turn),
    )
    config = {
        "main_attackers": [1031],
        "main_line": [860, 1030, 1031],
        "setup_basics": [860, 1030],
        "backup_attackers": [],
    }
    play_features = extract_action_features(state, play_basic, config)
    attack_features = extract_action_features(state, attack, config)
    end_features = extract_action_features(state, end_turn, config)
    assert play_features["play_pokemon_to_avoid_empty_board_loss"] == 1.0
    assert play_features["play_pokemon_behind_low_active"] == 1.0
    assert attack_features["attack_with_empty_board_risk"] == 1.0
    assert end_features["end_with_empty_board_risk"] == 1.0
    weights = default_weights()
    assert score_features(play_features, weights) > score_features(attack_features, weights)
    assert score_features(play_features, weights) > score_features(end_features, weights)


def _fake_scored_candidate(
    selected: list[int],
    *,
    score: float,
    linear: float,
    mlp: float | None = None,
    features: dict[str, float] | None = None,
) -> ScoredCandidate:
    return ScoredCandidate(
        candidate=SimpleNamespace(first_selection=selected, steps=()),
        features=features or {},
        score=score,
        components=ScoreBreakdown(linear=linear, mlp=mlp),
    )


def test_fallback_guardrail_does_not_apply_linear_margin_to_mlp_scores() -> None:
    guardrail = FallbackGuardrail()
    chosen = _fake_scored_candidate([0], score=11.0, linear=4079.0, mlp=11.0)
    fallback = _fake_scored_candidate(
        [26],
        score=6.8,
        linear=3671.0,
        mlp=6.8,
        features={"end_turn": 1.0, "end_with_useful_action": 1.0},
    )

    use_fallback, reason = guardrail.should_use_fallback(chosen, [chosen, fallback], [26])

    assert not use_fallback
    assert reason is None


def test_fallback_guardrail_can_still_use_strong_linear_fallback_with_mlp() -> None:
    guardrail = FallbackGuardrail()
    chosen = _fake_scored_candidate([0], score=11.0, linear=1200.0, mlp=11.0)
    fallback = _fake_scored_candidate([4], score=9.5, linear=1600.0, mlp=9.5)

    use_fallback, reason = guardrail.should_use_fallback(chosen, [chosen, fallback], [4])

    assert use_fallback
    assert reason == "fallback_linear_margin_guardrail"


def test_fallback_guardrail_respects_large_mlp_preference() -> None:
    guardrail = FallbackGuardrail()
    chosen = _fake_scored_candidate([0], score=11.0, linear=1200.0, mlp=11.0)
    fallback = _fake_scored_candidate(
        [26],
        score=8.0,
        linear=1600.0,
        mlp=8.0,
        features={"end_turn": 1.0, "end_with_useful_action": 1.0},
    )

    use_fallback, reason = guardrail.should_use_fallback(chosen, [chosen, fallback], [26])

    assert not use_fallback
    assert reason is None


def test_compact_label_dataset_filters_passive_end_labels() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        labels_path = tmp_path / "labels.jsonl"
        records = [
            {
                "label_index": 0,
                "matched": True,
                "candidates": [
                    {
                        "selected": [9],
                        "labels": ["end turn"],
                        "features": {"end_turn": 1.0, "end_with_useful_action": 1.0},
                    },
                    {
                        "selected": [2],
                        "labels": ["attach Basic Energy to active"],
                        "features": {"attach_energy": 1.0, "attach_to_main_line": 1.0},
                    },
                ],
            },
            {
                "label_index": 0,
                "matched": True,
                "candidates": [
                    {
                        "selected": [2],
                        "labels": ["play Duraludon", "end turn"],
                        "features": {
                            "play_pokemon": 1.0,
                            "end_turn": 1.0,
                            "seq_uses_end": 1.0,
                            "end_with_useful_action": 1.0,
                        },
                    },
                    {
                        "selected": [9],
                        "labels": ["end turn"],
                        "features": {"end_turn": 1.0},
                    },
                ],
            },
            {
                "label_index": 0,
                "matched": True,
                "candidates": [
                    {
                        "selected": [2],
                        "features": {
                            "play_pokemon": 1.0,
                            "end_turn": 1.0,
                            "seq_uses_end": 1.0,
                            "end_with_useful_action": 1.0,
                        },
                    },
                    {
                        "selected": [9],
                        "features": {"end_turn": 1.0},
                    },
                ],
            },
        ]
        labels_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
        kept_path = tmp_path / "kept.jsonl"
        kept = compact_label_dataset(str(labels_path), output=str(kept_path))
        assert kept["output_records"] == 3

        filtered_path = tmp_path / "filtered.jsonl"
        filtered = compact_label_dataset(str(labels_path), output=str(filtered_path), exclude_passive_end_labels=True)
        assert filtered["output_records"] == 2
        assert filtered["skipped_passive_end_label"] == 1


def test_decision_engine_warm_start_reuses_valid_remainder() -> None:
    config = load_config()
    deck = read_deck(config=config)
    engine = DecisionEngine(deck=deck, config=config)
    state = _memory_state(turn=3)
    first_action = _fake_action_view(0, label="choose yes")
    next_action = _fake_action_view(1, option_type="NO", label="choose no")
    observed = _fake_observed_for_warm_start(state, (first_action, next_action))
    sequence = CandidateSequence(
        steps=(
            SelectionStep((0,), (first_action,), "MAIN", "MAIN", {"first": 1.0}, 3.0, ("first",)),
            SelectionStep((1,), (next_action,), "MAIN", "MAIN", {"second": 1.0}, 5.0, ("second",)),
        ),
        observation=SimpleNamespace(current="predicted-final"),
        state=observed.state,
        planner_score=8.0,
        terminal=True,
        used_search=True,
        generation_tags=("simulated", "sample_0"),
        sim_result="terminal",
    )
    engine._cache_warm_start(observed, SimpleNamespace(candidate=sequence), [0])

    next_observed = _fake_observed_for_warm_start(state, (next_action,))
    sequences = engine._ensure_warm_start_candidate(next_observed, [])

    assert len(sequences) == 1
    assert sequences[0].first_selection == [1]
    assert sequences[0].steps[0].actions == (next_action,)
    assert "warm_start" in sequences[0].generation_tags
    assert engine.last_warm_start_status == "used"


def test_decision_engine_warm_start_clears_on_action_mismatch() -> None:
    config = load_config()
    deck = read_deck(config=config)
    engine = DecisionEngine(deck=deck, config=config)
    state = _memory_state(turn=3)
    first_action = _fake_action_view(0, label="choose yes")
    cached_next_action = _fake_action_view(1, option_type="NO", label="choose no")
    observed = _fake_observed_for_warm_start(state, (first_action, cached_next_action))
    sequence = CandidateSequence(
        steps=(
            SelectionStep((0,), (first_action,), "MAIN", "MAIN", {"first": 1.0}, 3.0, ("first",)),
            SelectionStep((1,), (cached_next_action,), "MAIN", "MAIN", {"second": 1.0}, 5.0, ("second",)),
        ),
        observation=SimpleNamespace(current="predicted-final"),
        state=observed.state,
        planner_score=8.0,
        terminal=True,
        used_search=True,
    )
    engine._cache_warm_start(observed, SimpleNamespace(candidate=sequence), [0])

    different_current_action = _fake_action_view(1, option_type="YES", label="choose yes")
    next_observed = _fake_observed_for_warm_start(state, (different_current_action,))

    assert engine._ensure_warm_start_candidate(next_observed, []) == []
    assert engine.pending_warm_start is None
    assert engine.last_warm_start_status == "cleared:action_mismatch"


def test_prize_tracker_deck_reveal_infers_known_prizes() -> None:
    config = load_config()
    deck = read_deck(config=config)
    prize_cards = deck[7:13]
    hand_cards = deck[:7]
    discard_cards = deck[13:15]
    deck_cards = deck[15:]
    player = SimpleNamespace(
        hand=[_fake_card(card_id) for card_id in hand_cards],
        discard=[_fake_card(card_id) for card_id in discard_cards],
        active=[],
        bench=[],
        prize=[None for _ in prize_cards],
    )
    opponent = SimpleNamespace(hand=None, discard=[], active=[], bench=[], prize=[])
    state = SimpleNamespace(player_index=0, my_prizes_remaining=len(prize_cards), my_deck_count=len(deck_cards))

    no_reveal_obs = SimpleNamespace(current=SimpleNamespace(players=[player, opponent]), select=SimpleNamespace(deck=None))
    no_reveal = PrizeTracker(config).update(no_reveal_obs, state, deck)
    assert not no_reveal.known_prized
    assert no_reveal.unknown_prize_count == len(prize_cards)
    assert sum(no_reveal.unknown_prize_pool.values()) == len(deck) - len(hand_cards) - len(discard_cards)

    reveal_obs = SimpleNamespace(
        current=SimpleNamespace(players=[player, opponent]),
        select=SimpleNamespace(deck=[_fake_card(card_id) for card_id in deck_cards], effect=None, contextCard=None),
    )
    revealed = PrizeTracker(config).update(reveal_obs, state, deck)
    assert revealed.known_prized == Counter(prize_cards)
    assert revealed.unknown_prize_count == 0
    assert not revealed.unknown_prize_pool
    assert sum(revealed.known_prized.values()) == len(prize_cards)

    effect_card = 1142
    hand_without_effect = [card_id for card_id in hand_cards if card_id != effect_card]
    effect_player = SimpleNamespace(
        hand=[_fake_card(card_id) for card_id in hand_without_effect],
        discard=[_fake_card(card_id) for card_id in discard_cards],
        active=[],
        bench=[],
        prize=[None for _ in prize_cards],
    )
    effect_reveal_obs = SimpleNamespace(
        current=SimpleNamespace(players=[effect_player, opponent], looking=[]),
        select=SimpleNamespace(deck=[_fake_card(card_id) for card_id in deck_cards], effect=_fake_card(effect_card), contextCard=None),
    )
    effect_revealed = PrizeTracker(config).update(effect_reveal_obs, state, deck)
    assert effect_revealed.known_prized == Counter(prize_cards)
    assert sum(effect_revealed.known_prized.values()) == state.my_prizes_remaining
    assert effect_revealed.unknown_prize_count == 0
    if effect_card not in prize_cards:
        assert effect_revealed.known_prized[effect_card] == 0

    mixed_owner_obs = SimpleNamespace(
        current=SimpleNamespace(players=[player, opponent], looking=[]),
        select=SimpleNamespace(
            deck=[_fake_card(card_id, 0) for card_id in deck_cards] + [_fake_card(deck[0], 1)],
            effect=None,
            contextCard=None,
        ),
    )
    mixed_owner = PrizeTracker(config).update(mixed_owner_obs, state, deck)
    assert mixed_owner.known_prized == Counter(prize_cards)


def test_sequence_decision_engine_smoke() -> None:
    config = load_config()
    deck = read_deck(config=config)
    obs_dict, _ = battle_start(deck, deck)
    try:
        obs = to_observation_class(obs_dict)
        adapter = RuntimeAdapter()
        assert adapter.parse(obs_dict).select is not None
        assert adapter.clamp(obs, [-1, 999], deck)
        engine = DecisionEngine(
            deck=deck,
            config=config,
            search_config=SequenceSearchConfig(beam_width=4, max_depth=3, max_nodes=24, time_budget_ms=250),
        )
        observed = engine.observe(obs)
        assert engine.fallback.__class__.__name__ == "LinearQPolicy"
        assert observed.state.legal_actions
        assert observed.probabilities == observed.belief.probabilities
        assert observed.game_plan.turn_plan.primary_goal
        planning = engine.plan(observed)
        assert planning.candidates
        assert planning.coverage is not None
        assert planning.coverage.num_candidates == len(planning.candidates)
        evaluated = engine.evaluate(planning)
        assert evaluated.evaluations
        assert evaluated.best is not None
        assert all(not candidate.features for candidate in planning.candidates)
        assert evaluated.best.components.risk != 0.0 or "risk_model_score" in evaluated.best.features
        assert isinstance(evaluated.best.to_sequence_view(), EvaluatedSequenceView)
        decision = engine.act(evaluated)
        assert decision.trace.sequence_count == len(evaluated.evaluations)
        sequences = engine.score_sequences(obs)
        assert sequences
        assert "p_combo_line_live" in sequences[0].features
        assert any(key.startswith("turn_goal_") for key in sequences[0].features)
        selected = engine.choose(obs)
        clean, invalid = sanitize_selection(obs, selected)
        assert not invalid
        assert clean == selected
        assert obs.select.minCount <= len(selected) <= obs.select.maxCount
        assert engine.last_trace is not None
        if engine.last_trace.used_search:
            assert engine.simulator.last_tree is not None
            assert len(engine.simulator.last_tree.nodes) > 0
    finally:
        battle_finish()


def test_clean_architecture_sequence_boundaries() -> None:
    import pokemon_agent.sequence as sequence_pkg

    assert hasattr(sequence_pkg, "__path__")
    pruner_source = inspect.getsource(SequencePruner)
    assert "score_features" not in pruner_source
    assert "default_weights" not in pruner_source


def test_sequence_ablation_switches() -> None:
    config = load_config()
    deck = read_deck(config=config)
    obs_dict, _ = battle_start(deck, deck)
    try:
        obs = to_observation_class(obs_dict)
        search_config = SequenceSearchConfig(beam_width=4, max_depth=3, max_nodes=24, time_budget_ms=250)
        engine_with_probability = DecisionEngine(deck=deck, config=config, search_config=search_config)
        planning_with_probability = engine_with_probability.plan(engine_with_probability.observe(obs))
        engine = DecisionEngine(
            deck=deck,
            config=config,
            probability_features_enabled=False,
            value_proxy_enabled=False,
            fallback_guardrail_enabled=False,
            search_config=search_config,
        )
        planning_without_probability = engine.plan(engine.observe(obs))
        assert planning_with_probability.coverage is not None
        assert planning_without_probability.coverage is not None
        assert planning_with_probability.coverage.missing_first_actions == planning_without_probability.coverage.missing_first_actions
        evaluated = engine.evaluate(planning_without_probability)
        assert evaluated.best is not None
        assert "p_combo_line_live" not in evaluated.best.features
        assert evaluated.best.components.value_proxy == 0.0
        assert engine.act(evaluated).selected
    finally:
        battle_finish()


def test_hidden_sample_planning_diagnostics() -> None:
    config = load_config()
    deck = read_deck(config=config)
    obs_dict, _ = battle_start(deck, deck)
    try:
        obs = to_observation_class(obs_dict)
        engine = DecisionEngine(
            deck=deck,
            config=config,
            search_config=SequenceSearchConfig(
                beam_width=4,
                max_depth=3,
                max_nodes=24,
                time_budget_ms=250,
                num_hidden_samples=2,
            ),
        )
        observed = engine.observe(obs)
        assert len(observed.belief.hidden_samples) == 2
        planning = engine.plan(observed)
        assert planning.hidden_samples_used
        assert planning.diagnostics["sample_count"] >= 1
        evaluated = engine.evaluate(planning)
        assert evaluated.best is not None
        assert "hidden_sample_support" in evaluated.best.features
    finally:
        battle_finish()


def test_sequence_policy_episode_smoke() -> None:
    config = load_config()
    deck = read_deck(config=config)
    result = run_episode(
        SequencePolicy(config=config, deck=deck, seed=5),
        RandomPolicy(seed=6),
        deck,
        deck,
        config=config,
        max_steps=200,
    )
    assert result.steps > 0
    assert result.invalid_actions == 0


def test_replay_pipeline_smoke() -> None:
    config = load_config()
    deck = read_deck(config=config)
    obs_dict, _ = battle_start(deck, deck)
    try:
        obs = to_observation_class(obs_dict)
        selected = HeuristicPolicy(config=config).choose(obs)
    finally:
        battle_finish()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        replay_path = tmp_path / "episode-1-replay.json"
        labels_path = tmp_path / "labels.jsonl"
        replay_path.write_text(
            json.dumps(
                {
                    "id": "synthetic",
                    "info": {"publicScore": 1800},
                    "steps": [
                        [{"action": deck, "reward": 0, "status": "ACTIVE"}, {"action": deck, "reward": 0, "status": "ACTIVE"}],
                        [{"observation": obs_dict, "action": selected, "reward": 0, "status": "ACTIVE"}],
                    ],
                }
            ),
            encoding="utf-8",
        )
        label_summary = build_label_dataset(
            [str(replay_path)],
            output=str(labels_path),
            max_examples=1,
            beam_width=2,
            max_depth=2,
            max_nodes=16,
            time_budget_ms=250,
        )
        assert label_summary["examples"] == 1
        assert label_summary["matched"] == 1
        assert label_summary["replay_decks_matched_db"] + label_summary["replay_decks_unknown"] == 1

        coverage_labels_path = tmp_path / "coverage_labels.jsonl"
        coverage_summary = build_candidate_coverage_dataset(
            [str(replay_path)],
            output=str(coverage_labels_path),
            max_examples=1,
            beam_width=2,
            max_depth=2,
            max_nodes=16,
            time_budget_ms=250,
            num_hidden_samples=2,
        )
        assert coverage_summary["examples"] == 1
        assert coverage_summary["expert_action_in_candidates"] == 1
        assert coverage_summary["expert_action_in_candidates_rate"] == 1.0
        coverage_record = json.loads(coverage_labels_path.read_text(encoding="utf-8").splitlines()[0])
        assert coverage_record["hidden_samples_used"]

        lagged_replay_path = tmp_path / "episode-2-replay.json"
        lagged_replay_path.write_text(
            json.dumps(
                {
                    "id": "synthetic-lagged",
                    "info": {"publicScore": 1800},
                    "steps": [
                        [{"observation": obs_dict, "action": [], "reward": 0, "status": "ACTIVE"}],
                        [{"observation": obs_dict, "action": selected, "reward": 0, "status": "ACTIVE"}],
                    ],
                }
            ),
            encoding="utf-8",
        )
        lagged_coverage_path = tmp_path / "lagged_coverage_labels.jsonl"
        lagged_summary = build_candidate_coverage_dataset(
            [str(lagged_replay_path)],
            output=str(lagged_coverage_path),
            max_examples=1,
            beam_width=2,
            max_depth=2,
            max_nodes=16,
            time_budget_ms=250,
        )
        assert lagged_summary["examples"] == 1
        assert lagged_summary["expert_action_in_candidates_rate"] == 1.0
        lagged_record = json.loads(lagged_coverage_path.read_text(encoding="utf-8").splitlines()[0])
        assert lagged_record["step_index"] == 0

        clean_labels_path = tmp_path / "clean_labels.jsonl"
        clean_label_summary = build_label_dataset(
            [str(replay_path)],
            output=str(clean_labels_path),
            max_examples=1,
            beam_width=2,
            max_depth=2,
            max_nodes=16,
            time_budget_ms=250,
            clean_evaluator=True,
        )
        assert clean_label_summary["examples"] == 1
        assert clean_label_summary["matched"] == 1
        clean_record = json.loads(clean_labels_path.read_text(encoding="utf-8").splitlines()[0])
        clean_features = clean_record["candidates"][0]["features"]
        assert "value_model_score" not in clean_features
        assert not any(key.startswith("p_") for key in clean_features)

        compact_path = tmp_path / "compact_labels.jsonl"
        compact_summary = compact_label_dataset(str(labels_path), output=str(compact_path))
        assert compact_summary["output_records"] == 1
        assert compact_summary["candidate_count"] > 0
        compact_record = json.loads(compact_path.read_text(encoding="utf-8").splitlines()[0])
        assert {"candidates", "label_index", "matched", "source", "episode_id", "self_deck_id", "self_archetype"} <= set(compact_record)

        train_summary = pretrain_sequence_scorer(str(labels_path), out_dir=str(tmp_path / "pretrain"), epochs=1, lr=0.001)
        assert train_summary["examples"] == 1
        assert Path(train_summary["latest_checkpoint"]).exists()

        mlp_summary = pretrain_mlp_ranker(
            str(labels_path),
            out_dir=str(tmp_path / "mlp_pretrain"),
            epochs=1,
            lr=0.001,
            hidden_dim=4,
            hidden_dim2=2,
            max_features=32,
            max_examples=1,
            seed=7,
            validation_fraction=0.0,
        )
        assert mlp_summary["examples"] == 1
        assert mlp_summary["feature_count"] > 0
        assert mlp_summary["hidden_dims"] == [4, 2]
        assert mlp_summary["validation_fraction"] == 0.0
        assert mlp_summary["excluded_features"] == ["linear_score", "mlp_score"]
        mlp_path = Path(mlp_summary["latest_checkpoint"])
        assert mlp_path.exists()
        model = MLPSequenceRanker.from_file(mlp_path)
        assert model.hidden_dims == [4, 2]
        assert model.feature_schema_version == "full"
        assert "linear_score" not in model.feature_index
        assert "mlp_score" not in model.feature_index
        record = json.loads(labels_path.read_text(encoding="utf-8").splitlines()[0])
        scores = model.score_batch([candidate["features"] for candidate in record["candidates"]])
        assert len(scores) == len(record["candidates"])
        assert all(isinstance(score, float) for score in scores)
        packaged_mlp_path = mlp_path
        expected_mlp_name = "mlp_model.json"
        if torch_available():
            pt_path = tmp_path / "mlp_pretrain" / "latest.pt"
            model.save_pt(pt_path, summary={"test": "smoke"})
            pt_model = MLPSequenceRanker.from_file(pt_path)
            assert pt_model.hidden_dims == [4, 2]
            packaged_mlp_path = pt_path
            expected_mlp_name = "mlp_model.pt"

        if torch_available():
            rl_summary = rl_self_play_mlp(
                1,
                model_path=str(packaged_mlp_path),
                out_dir=str(tmp_path / "mlp_self_play"),
                lr=0.0001,
                epsilon_start=0.0,
                epsilon_end=0.0,
                checkpoint_every=1,
            )
            assert rl_summary["games"] == 1
            assert rl_summary["invalid_actions"] == 0
            assert Path(rl_summary["latest_checkpoint"]).exists()

        combined = make_policy(f"sequence-mlp:{packaged_mlp_path},{train_summary['latest_checkpoint']}", config=config, seed=9)
        assert isinstance(combined, SequencePolicy)
        assert combined.engine.scorer.linear_weight == 0.0
        assert combined.engine.scorer.mlp_weight == 1.0

        bundled = build_submission(
            weights_path=train_summary["latest_checkpoint"],
            mlp_model_path=str(packaged_mlp_path),
            output=str(tmp_path / "mlp_submission.tar.gz"),
        )
        with tarfile.open(bundled, "r:gz") as tar:
            names = set(tar.getnames())
            runtime_config = json.loads(tar.extractfile("runtime_config.json").read().decode("utf-8"))  # type: ignore[union-attr]
        assert expected_mlp_name in names
        assert "weights.json" in names
        assert runtime_config["linear_score_weight"] == 0.0
        assert runtime_config["mlp_score_weight"] == 1.0
        assert runtime_config["mlp_model_file"] == expected_mlp_name


def test_top_replay_planner_candidate_coverage() -> None:
    replay_dir = Path("artifacts/replays/kaggle/pokemon-tcg-ai-battle")
    if not replay_dir.exists() or not any(replay_dir.glob("episode-*-replay.json")):
        return
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "top_replay_coverage.jsonl"
        summary = build_candidate_coverage_dataset(
            [str(replay_dir)],
            output=str(output),
            max_examples=320,
            beam_width=8,
            max_depth=8,
            max_nodes=128,
            time_budget_ms=500,
        )
        assert summary["examples"] > 0
        assert summary["expert_action_in_candidates"] == summary["examples"]
        assert summary["expert_action_in_candidates_rate"] == 1.0
        multi_select_seen = False
        for line in output.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            assert record["matched"], record["action_type"]
            assert record["coverage"]["missing_first_actions"] == []
            multi_select_seen = multi_select_seen or len(record["selected"]) > 1
        assert multi_select_seen


def main() -> None:
    test_action_state_reward_smoke()
    test_episode_smoke()
    test_heuristic_episode_smoke()
    test_submission_smoke()
    test_deck_db_smoke()
    test_deck_gameplan_profiles_smoke()
    test_deck_db_opponent_pool_smoke()
    test_opponent_memory_setup_tag_clears_after_evolution()
    test_probability_model_hypergeometric_features()
    test_search_tree_reconstructs_leaf_sequence()
    test_empty_board_safety_features_prefer_benching_basic()
    test_fallback_guardrail_does_not_apply_linear_margin_to_mlp_scores()
    test_fallback_guardrail_can_still_use_strong_linear_fallback_with_mlp()
    test_fallback_guardrail_respects_large_mlp_preference()
    test_compact_label_dataset_filters_passive_end_labels()
    test_decision_engine_warm_start_reuses_valid_remainder()
    test_decision_engine_warm_start_clears_on_action_mismatch()
    test_prize_tracker_deck_reveal_infers_known_prizes()
    test_sequence_decision_engine_smoke()
    test_clean_architecture_sequence_boundaries()
    test_sequence_ablation_switches()
    test_hidden_sample_planning_diagnostics()
    test_sequence_policy_episode_smoke()
    test_replay_pipeline_smoke()
    test_top_replay_planner_candidate_coverage()
    print("smoke tests passed")


if __name__ == "__main__":
    main()
