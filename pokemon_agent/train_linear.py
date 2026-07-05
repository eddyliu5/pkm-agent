from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config, read_deck
from .deck_db import expand_deck, iter_decks, load_deck_db, validate_deck
from .engine import EpisodeResult, run_episode
from .features import default_weights, score_features
from .policy import EpsilonGreedyPolicy, HeuristicPolicy, LinearQPolicy, RandomPolicy, load_weights


def _save_checkpoint(path: Path, weights: dict[str, float], games: int, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"games": games, "weights": weights, "summary": summary}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _update_weights(weights: dict[str, float], result: EpisodeResult, train_player: int, lr: float, gamma: float) -> None:
    returns = 1.0 * result.outcome_for(train_player)
    for transition in reversed(result.transitions):
        if transition.player != train_player:
            continue
        returns = (transition.reward / 100.0) + gamma * returns
        returns = max(-3.0, min(3.0, returns))
        current_score = score_features(transition.features, weights) / 1000.0
        advantage = max(-3.0, min(3.0, returns - current_score))
        for key, value in transition.features.items():
            weights[key] = max(-20000.0, min(20000.0, weights.get(key, 0.0) + lr * advantage * value * 100.0))


def _pick_opponent(name: str, weights: dict[str, float], config: dict, seed: int):
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "heuristic":
        return HeuristicPolicy(config=config)
    if name == "self":
        return LinearQPolicy(weights=weights, config=config)
    raise ValueError(f"Unknown opponent: {name}")


def _opponent_deck_pool(source: str | None) -> list[tuple[str, str, list[int]]]:
    if source in (None, "mirror"):
        return []
    if source != "deck_db":
        raise ValueError(f"Unknown opponent deck source: {source}")
    db = load_deck_db()
    pool: list[tuple[str, str, list[int]]] = []
    for deck in iter_decks(db):
        if validate_deck(deck):
            continue
        pool.append((str(deck["id"]), str(deck["archetype"]), expand_deck(deck)))
    if not pool:
        raise ValueError("No valid decks found in deck database")
    return pool


def train(
    games: int,
    *,
    opponents: list[str],
    config_path: str | None = None,
    checkpoint_every: int = 100,
    out_dir: str = "artifacts/checkpoints",
    lr: float = 0.003,
    gamma: float = 0.96,
    epsilon_start: float = 0.25,
    epsilon_end: float = 0.05,
    weights_path: str | None = None,
    opponent_decks: str | None = None,
) -> dict:
    config = load_config(config_path)
    deck = read_deck(config=config)
    opponent_deck_pool = _opponent_deck_pool(opponent_decks)
    weights = default_weights()
    if weights_path:
        weights.update(load_weights(weights_path))

    rng = random.Random(982451653)
    out_path = Path(out_dir)
    report_dir = Path("artifacts/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    wins = losses = draws = invalid = total_steps = 0
    best_window = -math.inf
    recent: list[float] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics_path = out_path / f"train_{stamp}.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as metrics:
        for game in range(1, games + 1):
            frac = game / max(1, games)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac
            train_side = game % 2
            opponent_name = rng.choice(opponents)
            opponent_deck_id = "mirror"
            opponent_archetype = config.get("name", "mirror")
            opponent_deck = deck
            if opponent_deck_pool:
                opponent_deck_id, opponent_archetype, opponent_deck = opponent_deck_pool[(game - 1) % len(opponent_deck_pool)]
            train_policy = EpsilonGreedyPolicy(LinearQPolicy(weights=weights, config=config), epsilon=epsilon, seed=game * 31)
            opponent_policy = _pick_opponent(opponent_name, weights, config, seed=game * 37)
            if train_side == 0:
                result = run_episode(train_policy, opponent_policy, deck, opponent_deck, config=config, collect_transitions=True)
            else:
                result = run_episode(opponent_policy, train_policy, opponent_deck, deck, config=config, collect_transitions=True)

            _update_weights(weights, result, train_side, lr=lr, gamma=gamma)
            outcome = result.outcome_for(train_side)
            recent.append(outcome)
            if len(recent) > checkpoint_every:
                recent.pop(0)
            wins += int(outcome > 0)
            losses += int(outcome < 0)
            draws += int(outcome == 0)
            invalid += result.invalid_actions
            total_steps += result.steps
            row = {
                "game": game,
                "train_side": train_side,
                "opponent": opponent_name,
                "opponent_deck_id": opponent_deck_id,
                "opponent_archetype": opponent_archetype,
                "winner": result.winner,
                "outcome": outcome,
                "steps": result.steps,
                "invalid_actions": result.invalid_actions,
                "epsilon": epsilon,
            }
            metrics.write(json.dumps(row) + "\n")

            if game % checkpoint_every == 0 or game == games:
                window_score = sum(recent) / len(recent) if recent else 0.0
                summary = {
                    "games": game,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "win_rate": wins / game,
                    "invalid_actions": invalid,
                    "average_steps": total_steps / game,
                    "recent_score": window_score,
                    "metrics": str(metrics_path),
                }
                _save_checkpoint(out_path / f"checkpoint_{game}.json", weights, game, summary)
                _save_checkpoint(out_path / "latest.json", weights, game, summary)
                if window_score > best_window:
                    best_window = window_score
                    _save_checkpoint(out_path / "best.json", weights, game, summary)

    final_summary = {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "opponent_decks": opponent_decks or "mirror",
        "invalid_actions": invalid,
        "average_steps": total_steps / games if games else 0.0,
        "metrics": str(metrics_path),
        "best_checkpoint": str(out_path / "best.json"),
        "latest_checkpoint": str(out_path / "latest.json"),
    }
    report_path = report_dir / "overnight_summary.md"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("# Overnight Training Summary\n\n")
        for key, value in final_summary.items():
            report.write(f"- {key}: {value}\n")
    print(json.dumps(final_summary, indent=2))
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a linear legal-action scorer.")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--opponents", default="random,heuristic,self")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--out-dir", default="artifacts/checkpoints")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument("--epsilon-start", type=float, default=0.25)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--opponent-decks", default=None, choices=["mirror", "deck_db"])
    args = parser.parse_args()
    opponents = [item.strip() for item in args.opponents.split(",") if item.strip()]
    train(
        args.games,
        opponents=opponents,
        config_path=args.config,
        checkpoint_every=args.checkpoint_every,
        out_dir=args.out_dir,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        weights_path=args.weights,
        opponent_decks=args.opponent_decks,
    )


if __name__ == "__main__":
    main()
