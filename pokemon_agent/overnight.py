from __future__ import annotations

import argparse
import json
from pathlib import Path

from .eval import evaluate
from .submission import build_submission
from .tests_smoke import (
    test_action_state_reward_smoke,
    test_deck_db_opponent_pool_smoke,
    test_deck_db_smoke,
    test_deck_gameplan_profiles_smoke,
    test_episode_smoke,
    test_heuristic_episode_smoke,
    test_prize_tracker_deck_reveal_infers_known_prizes,
    test_sequence_decision_engine_smoke,
    test_sequence_policy_episode_smoke,
)
from .train_linear import train


def _run_smoke_suite() -> None:
    test_action_state_reward_smoke()
    test_episode_smoke()
    test_heuristic_episode_smoke()
    test_deck_db_smoke()
    test_deck_gameplan_profiles_smoke()
    test_deck_db_opponent_pool_smoke()
    test_prize_tracker_deck_reveal_infers_known_prizes()
    test_sequence_decision_engine_smoke()
    test_sequence_policy_episode_smoke()


def run_overnight(
    train_games: int,
    baseline_games: int,
    checkpoint_every: int,
    *,
    config_path: str | None = None,
    train_linear_weights: bool = False,
    out_dir: str = "artifacts/checkpoints/overnight_sequence",
) -> dict:
    _run_smoke_suite()

    sequence_random = evaluate("sequence", "random", baseline_games, config_path=config_path)
    sequence_deck_db = evaluate("sequence", "random", baseline_games, config_path=config_path, opponent_decks="deck_db")
    sequence_heuristic = evaluate("sequence", "heuristic", max(20, baseline_games // 2), config_path=config_path)

    train_summary = None
    learned_random = None
    learned_deck_db = None
    learned_heuristic = None
    best = None
    if train_linear_weights and train_games > 0:
        train_summary = train(
            train_games,
            opponents=["random", "heuristic", "self"],
            config_path=config_path,
            checkpoint_every=checkpoint_every,
            out_dir=out_dir,
            opponent_decks="deck_db",
        )
        best = train_summary["best_checkpoint"]
        learned_spec = f"sequence:{best}"
        learned_random = evaluate(learned_spec, "random", max(20, baseline_games // 2), config_path=config_path)
        learned_deck_db = evaluate(learned_spec, "random", max(20, baseline_games // 2), config_path=config_path, opponent_decks="deck_db")
        learned_heuristic = evaluate(learned_spec, "heuristic", max(20, baseline_games // 2), config_path=config_path)

    submission_path = build_submission(weights_path=best)

    summary = {
        "sequence_vs_random": sequence_random,
        "sequence_vs_random_deck_db": sequence_deck_db,
        "sequence_vs_heuristic": sequence_heuristic,
        "training": train_summary,
        "learned_vs_random": learned_random,
        "learned_vs_random_deck_db": learned_deck_db,
        "learned_vs_heuristic": learned_heuristic,
        "submission": str(submission_path),
    }
    report_path = Path("artifacts/reports/overnight_full_summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequence-agent validation, optional linear-weight training, and submission packaging.")
    parser.add_argument("--train-games", type=int, default=0)
    parser.add_argument("--baseline-games", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-linear-weights", action="store_true", help="Also train legacy linear weights for the sequence scorer fallback/features.")
    parser.add_argument("--out-dir", default="artifacts/checkpoints/overnight_sequence")
    args = parser.parse_args()
    run_overnight(
        args.train_games,
        args.baseline_games,
        args.checkpoint_every,
        config_path=args.config,
        train_linear_weights=args.train_linear_weights,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
