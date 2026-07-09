from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from pokemon_agent.deck_db import load_deck_db, observed_win_rate, observed_wins, top_observed_winrate_decks


DEFAULT_AGENTS = ("sequence",)
BASELINE_SEEDS = (
    42,
    1729,
    8675309,
    314159,
    271828,
    161803,
    57721,
    104729,
    65537,
    99991,
    123457,
    424243,
    777777,
    135791,
    246802,
    987643,
    112358,
    141421,
    173205,
    223607,
    264575,
    316227,
    707107,
    811811,
    919191,
    1000003,
    1000033,
    1000037,
    1000039,
    1000081,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _csv_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("eval_*.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    games = len(rows)
    wins = sum(1 for row in rows if row.get("result") == "win")
    losses = sum(1 for row in rows if row.get("result") == "loss")
    draws = sum(1 for row in rows if row.get("result") == "draw")
    invalid = sum(int(float(row.get("invalid_actions") or 0)) for row in rows)
    steps = [float(row.get("steps") or 0) for row in rows]
    prize_diffs = [float(row.get("prize_diff") or 0) for row in rows]
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "invalid_actions": invalid,
        "average_steps": mean(steps) if steps else 0.0,
        "average_prize_diff": mean(prize_diffs) if prize_diffs else 0.0,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_deck_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "agent",
        "agent_deck_id",
        "archetype",
        "observed_win_rate",
        "observed_wins",
        "games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "invalid_actions",
        "average_steps",
        "average_prize_diff",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _aggregate_deck_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {}
    total_games = sum(int(row["games"]) for row in summaries)
    total_wins = sum(int(row["wins"]) for row in summaries)
    return {
        "deck_count": len(summaries),
        "total_games": total_games,
        "total_wins": total_wins,
        "total_losses": sum(int(row["losses"]) for row in summaries),
        "total_draws": sum(int(row["draws"]) for row in summaries),
        "pooled_win_rate": total_wins / total_games if total_games else 0.0,
        "mean_deck_win_rate": mean(float(row["win_rate"]) for row in summaries),
        "total_invalid_actions": sum(int(row["invalid_actions"]) for row in summaries),
        "mean_average_steps": mean(float(row["average_steps"]) for row in summaries),
        "mean_average_prize_diff": mean(float(row["average_prize_diff"]) for row in summaries),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sequence baselines for each top deck against top-deck opponents.")
    parser.add_argument("--run-id", default=_run_id())
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--rounds", type=int, default=5, help="Full top-deck opponent rotations per agent deck. Default 5 => 150 games with 30 top decks.")
    parser.add_argument("--games", type=int, default=None, help="Legacy total games per agent deck. Must be divisible by the selected top-deck count.")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--min-observed-wins", type=int, default=25)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--python", default="./.venv/bin/python")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--agents", default=",".join(DEFAULT_AGENTS), help="Comma-separated agent specs to evaluate. Default: sequence.")
    parser.add_argument("--seed-list", default="", help="Comma-separated per-round seeds. Defaults to the first --rounds fixed baseline seeds.")
    args = parser.parse_args()

    out_root = Path(args.out_root or f"baseline/compare/{args.run_id}")
    db = load_deck_db()
    decks = top_observed_winrate_decks(db, min_observed_wins=args.min_observed_wins, top_n=args.top_n)
    deck_records = [
        {
            "id": str(deck["id"]),
            "archetype": str(deck.get("archetype", "")),
            "observed_win_rate": observed_win_rate(deck),
            "observed_wins": observed_wins(deck),
        }
        for deck in decks
    ]
    if not deck_records:
        raise ValueError("No top decks selected")
    if args.games is not None:
        if args.games <= 0 or args.games % len(deck_records) != 0:
            raise ValueError(f"--games must be a positive multiple of selected deck count ({len(deck_records)})")
        rounds = args.games // len(deck_records)
    else:
        rounds = int(args.rounds)
    if rounds <= 0:
        raise ValueError("--rounds must be positive")
    seed_schedule = tuple(int(value) for value in args.seed_list.split(",") if value.strip()) if args.seed_list else BASELINE_SEEDS[:rounds]
    if not seed_schedule:
        raise ValueError("Seed schedule cannot be empty")
    if len(seed_schedule) > 30 or rounds > 30:
        raise ValueError("Top-deck baseline is capped at 30 rounds/seeds for now")
    if len(seed_schedule) < rounds:
        raise ValueError(f"Seed schedule has {len(seed_schedule)} seeds but {rounds} rounds were requested")
    seed_schedule = seed_schedule[:rounds]
    games = rounds * len(deck_records)
    agents = tuple(agent.strip() for agent in args.agents.split(",") if agent.strip())
    if not agents:
        raise ValueError("--agents cannot be empty")
    manifest = {
        "run_id": args.run_id,
        "rounds_per_agent_deck": rounds,
        "games_per_agent_deck": games,
        "workers": args.workers,
        "agents": list(agents),
        "opponent": "sequence",
        "opponent_decks": "top_winrate",
        "min_observed_wins": args.min_observed_wins,
        "top_n": args.top_n,
        "deck_count": len(deck_records),
        "seed_schedule": list(seed_schedule),
        "round_robin_opponents": True,
        "decks": deck_records,
    }
    _write_json(out_root / "manifest.json", manifest)

    all_deck_summaries: list[dict[str, Any]] = []
    for agent in agents:
        agent_summaries: list[dict[str, Any]] = []
        for deck in deck_records:
            deck_id = deck["id"]
            out_dir = out_root / agent / "decks" / deck_id
            existing_rows = _csv_rows(out_dir)
            if args.skip_existing and existing_rows:
                print(f"skip existing: {agent} {deck_id}", flush=True)
            else:
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                cmd = [
                    args.python,
                    "-m",
                    "pokemon_agent.eval",
                    "--agent",
                    agent,
                    "--opponent",
                    "sequence",
                    "--opponent-decks",
                    "top_winrate",
                    "--agent-deck-id",
                    deck_id,
                    "--games",
                    str(games),
                    "--workers",
                    str(args.workers),
                    "--out-dir",
                    str(out_dir),
                    "--planner-coverage-report",
                    "--disable-mlp",
                    "--round-robin-opponents",
                    "--seed-list",
                    ",".join(str(seed) for seed in seed_schedule),
                ]
                print(f"run: {agent} deck={deck_id} rounds={rounds} games={games} workers={args.workers}", flush=True)
                subprocess.run(cmd, check=True)
            rows = _csv_rows(out_dir)
            summary = {
                "agent": agent,
                "agent_deck_id": deck_id,
                "archetype": deck["archetype"],
                "observed_win_rate": deck["observed_win_rate"],
                "observed_wins": deck["observed_wins"],
                **_summarize(rows),
            }
            agent_summaries.append(summary)
            all_deck_summaries.append(summary)
            _write_json(out_dir / "summary.json", summary)

        aggregate = _aggregate_deck_summaries(agent_summaries)
        _write_json(out_root / agent / "aggregate.json", aggregate)
        _write_deck_summary_csv(out_root / agent / "deck_summaries.csv", agent_summaries)

    by_agent = {
        agent: _aggregate_deck_summaries([row for row in all_deck_summaries if row["agent"] == agent])
        for agent in agents
    }
    comparison = {
        "agents": by_agent,
        "pooled_win_rate_delta_sequence_minus_previous_best": (
            by_agent["sequence"]["pooled_win_rate"] - by_agent["sequence-starmie-froslass"]["pooled_win_rate"]
            if "sequence" in by_agent and "sequence-starmie-froslass" in by_agent and all(by_agent.values())
            else None
        ),
        "mean_deck_win_rate_delta_sequence_minus_previous_best": (
            by_agent["sequence"]["mean_deck_win_rate"] - by_agent["sequence-starmie-froslass"]["mean_deck_win_rate"]
            if "sequence" in by_agent and "sequence-starmie-froslass" in by_agent and all(by_agent.values())
            else None
        ),
    }
    _write_json(out_root / "comparison.json", comparison)
    _write_deck_summary_csv(out_root / "all_deck_summaries.csv", all_deck_summaries)
    print(json.dumps({"out_root": str(out_root), "comparison": comparison}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
