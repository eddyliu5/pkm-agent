from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config, read_deck
from .deck_db import expand_deck, iter_decks, load_deck_db, validate_deck
from .engine import run_episode
from .policy import make_policy


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)[:80]


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


def evaluate(
    agent_spec: str,
    opponent_spec: str,
    games: int,
    *,
    config_path: str | None = None,
    out_dir: str = "artifacts/eval",
    opponent_decks: str | None = None,
) -> dict:
    config = load_config(config_path)
    deck = read_deck(config=config)
    opponent_pool = _opponent_deck_pool(opponent_decks)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"{_slug(agent_spec)}_vs_{_slug(opponent_spec)}"
    jsonl_path = out_path / f"eval_{name}_{stamp}.jsonl"
    csv_path = out_path / f"eval_{name}_{stamp}.csv"
    wins = losses = draws = invalid = total_steps = 0
    prize_diffs: list[int] = []

    with jsonl_path.open("w", encoding="utf-8") as jsonl, csv_path.open("w", encoding="utf-8", newline="") as csv_fh:
        writer = csv.DictWriter(
            csv_fh,
            fieldnames=[
                "game",
                "agent_side",
                "opponent_deck_id",
                "opponent_archetype",
                "winner",
                "result",
                "steps",
                "invalid_actions",
                "agent_prizes",
                "opponent_prizes",
                "prize_diff",
            ],
        )
        writer.writeheader()
        for game in range(games):
            agent_side = game % 2
            opponent_deck_id = "mirror"
            opponent_archetype = config.get("name", "mirror")
            opponent_deck = deck
            if opponent_pool:
                opponent_deck_id, opponent_archetype, opponent_deck = opponent_pool[game % len(opponent_pool)]
            agent = make_policy(agent_spec, config=config, seed=game * 17 + 1)
            opponent = make_policy(opponent_spec, config=config, seed=game * 17 + 2)
            if agent_side == 0:
                result = run_episode(agent, opponent, deck, opponent_deck, config=config)
                agent_prizes, opponent_prizes = result.final_prizes
            else:
                result = run_episode(opponent, agent, opponent_deck, deck, config=config)
                opponent_prizes, agent_prizes = result.final_prizes

            if result.winner == agent_side:
                outcome = "win"
                wins += 1
            elif result.winner == 2:
                outcome = "draw"
                draws += 1
            else:
                outcome = "loss"
                losses += 1

            prize_diff = opponent_prizes - agent_prizes
            prize_diffs.append(prize_diff)
            invalid += result.invalid_actions
            total_steps += result.steps
            row = {
                "game": game,
                "agent_side": agent_side,
                "opponent_deck_id": opponent_deck_id,
                "opponent_archetype": opponent_archetype,
                "winner": result.winner,
                "result": outcome,
                "steps": result.steps,
                "invalid_actions": result.invalid_actions,
                "agent_prizes": agent_prizes,
                "opponent_prizes": opponent_prizes,
                "prize_diff": prize_diff,
            }
            writer.writerow(row)
            jsonl.write(json.dumps(row) + "\n")

    summary = {
        "agent": agent_spec,
        "opponent": opponent_spec,
        "opponent_decks": opponent_decks or "mirror",
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "invalid_actions": invalid,
        "average_steps": total_steps / games if games else 0.0,
        "average_prize_diff": sum(prize_diffs) / games if games else 0.0,
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pokemon TCG battle policies.")
    parser.add_argument("--agent", default="sequence")
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default="artifacts/eval")
    parser.add_argument("--opponent-decks", default=None, choices=["mirror", "deck_db"], help="Use mirror deck or rotate through valid deck_db decks.")
    args = parser.parse_args()
    evaluate(args.agent, args.opponent, args.games, config_path=args.config, out_dir=args.out_dir, opponent_decks=args.opponent_decks)


if __name__ == "__main__":
    main()
