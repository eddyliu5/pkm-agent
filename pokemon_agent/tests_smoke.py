from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
import json
import tempfile
import tarfile
import subprocess
import sys
from pathlib import Path

from .belief import PrizeTracker
from .actions import legal_actions, sanitize_selection
from .config import load_config, read_deck
from .decision_engine import DecisionEngine
from .deck_db import archetype_gameplan_profiles, deck_gameplan_profile, iter_decks, load_deck_db, rank_decks_by_observed_cards, validate_deck
from .engine import run_episode
from .eval import _opponent_deck_pool
from .features import extract_action_features
from .mlp_ranker import MLPSequenceRanker, pretrain_mlp_ranker
from .policy import HeuristicPolicy, RandomPolicy, SequencePolicy, make_policy
from .replay_pipeline import build_label_dataset, compact_label_dataset, pretrain_sequence_scorer, rl_self_play_mlp
from .sequence import SequenceSearchConfig
from .state import build_agent_state
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


def _fake_card(card_id: int, player_index: int = 0):
    return SimpleNamespace(id=int(card_id), serial=int(card_id), playerIndex=player_index)


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

    reveal_obs = SimpleNamespace(
        current=SimpleNamespace(players=[player, opponent]),
        select=SimpleNamespace(deck=[_fake_card(card_id) for card_id in deck_cards], effect=None, contextCard=None),
    )
    revealed = PrizeTracker(config).update(reveal_obs, state, deck)
    assert revealed.known_prized == Counter(prize_cards)
    assert revealed.unknown_prize_count == 0
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
        engine = DecisionEngine(
            deck=deck,
            config=config,
            search_config=SequenceSearchConfig(beam_width=4, max_depth=3, max_nodes=24, time_budget_ms=250),
        )
        selected = engine.choose(obs)
        clean, invalid = sanitize_selection(obs, selected)
        assert not invalid
        assert clean == selected
        assert obs.select.minCount <= len(selected) <= obs.select.maxCount
        assert engine.last_trace is not None
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
                    "steps": [[{"observation": obs_dict, "action": selected, "reward": 0, "status": "ACTIVE"}]],
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
        compact_path = tmp_path / "compact_labels.jsonl"
        compact_summary = compact_label_dataset(str(labels_path), output=str(compact_path))
        assert compact_summary["output_records"] == 1
        assert compact_summary["candidate_count"] > 0
        compact_record = json.loads(compact_path.read_text(encoding="utf-8").splitlines()[0])
        assert set(compact_record) == {"candidates", "label_index", "matched"}

        train_summary = pretrain_sequence_scorer(str(labels_path), out_dir=str(tmp_path / "pretrain"), epochs=1, lr=0.001)
        assert train_summary["examples"] == 1
        assert Path(train_summary["latest_checkpoint"]).exists()

        mlp_summary = pretrain_mlp_ranker(
            str(labels_path),
            out_dir=str(tmp_path / "mlp_pretrain"),
            epochs=1,
            lr=0.001,
            hidden_dim=4,
            max_features=32,
            max_examples=1,
            seed=7,
        )
        assert mlp_summary["examples"] == 1
        assert mlp_summary["feature_count"] > 0
        mlp_path = Path(mlp_summary["latest_checkpoint"])
        assert mlp_path.exists()
        model = MLPSequenceRanker.from_file(mlp_path)
        record = json.loads(labels_path.read_text(encoding="utf-8").splitlines()[0])
        scores = model.score_batch([candidate["features"] for candidate in record["candidates"]])
        assert len(scores) == len(record["candidates"])
        assert all(isinstance(score, float) for score in scores)

        rl_summary = rl_self_play_mlp(
            1,
            model_path=str(mlp_path),
            out_dir=str(tmp_path / "mlp_self_play"),
            lr=0.0001,
            epsilon_start=0.0,
            epsilon_end=0.0,
            checkpoint_every=1,
        )
        assert rl_summary["games"] == 1
        assert rl_summary["invalid_actions"] == 0
        assert Path(rl_summary["latest_checkpoint"]).exists()

        combined = make_policy(f"sequence-mlp:{mlp_path},{train_summary['latest_checkpoint']}", config=config, seed=9)
        assert isinstance(combined, SequencePolicy)

        bundled = build_submission(
            weights_path=train_summary["latest_checkpoint"],
            mlp_model_path=str(mlp_path),
            output=str(tmp_path / "mlp_submission.tar.gz"),
        )
        with tarfile.open(bundled, "r:gz") as tar:
            names = set(tar.getnames())
        assert "mlp_model.json" in names
        assert "weights.json" in names


def main() -> None:
    test_action_state_reward_smoke()
    test_episode_smoke()
    test_heuristic_episode_smoke()
    test_submission_smoke()
    test_deck_db_smoke()
    test_deck_gameplan_profiles_smoke()
    test_deck_db_opponent_pool_smoke()
    test_prize_tracker_deck_reveal_infers_known_prizes()
    test_sequence_decision_engine_smoke()
    test_sequence_policy_episode_smoke()
    test_replay_pipeline_smoke()
    print("smoke tests passed")


if __name__ == "__main__":
    main()
