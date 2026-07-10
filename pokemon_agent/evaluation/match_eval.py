from __future__ import annotations

import argparse
import os
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from ..config import load_config, read_deck
from ..deck_db import deck_heuristic_config, expand_deck, iter_decks, load_deck_db, observed_win_rate, observed_wins, top_observed_winrate_decks, validate_deck
from ..engine import run_episode
from ..policy import make_policy


EVAL_FIELDNAMES = [
    "game",
    "agent_side",
    "agent_deck_id",
    "agent_archetype",
    "opponent_deck_id",
    "opponent_archetype",
    "winner",
    "result",
    "seed",
    "steps",
    "invalid_actions",
    "agent_prizes",
    "opponent_prizes",
    "prize_diff",
]


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)[:80]


def _resolved_deck_record(deck: dict, db: dict) -> tuple[str, str, list[int], dict]:
    archetype = str(deck["archetype"])
    return (str(deck["id"]), archetype, expand_deck(deck), deck_heuristic_config(deck, db))


def _single_deck_pool(deck_id: str | None) -> list[tuple[str, str, list[int], dict]]:
    if not deck_id:
        return []
    db = load_deck_db()
    for deck in iter_decks(db):
        if str(deck.get("id")) == deck_id:
            if validate_deck(deck):
                raise ValueError(f"Invalid deck_db deck: {deck_id}")
            return [_resolved_deck_record(deck, db)]
    raise ValueError(f"Unknown deck id: {deck_id}")


def _deck_pool(source: str | None) -> list[tuple[str, str, list[int], dict]]:
    if source in (None, "mirror"):
        return []
    if source not in {"deck_db", "top_winrate"}:
        raise ValueError(f"Unknown deck source: {source}")
    db = load_deck_db()
    pool: list[tuple[str, str, list[int], dict]] = []
    decks = top_observed_winrate_decks(db, min_observed_wins=25) if source == "top_winrate" else list(iter_decks(db))
    for deck in decks:
        if validate_deck(deck):
            continue
        archetype = str(deck["archetype"])
        if source == "top_winrate":
            archetype = f"{archetype}:wr={observed_win_rate(deck):.4f}:wins={observed_wins(deck)}"
        deck_id, _base_archetype, deck_list, role_config = _resolved_deck_record(deck, db)
        pool.append((deck_id, archetype, deck_list, role_config))
    if not pool:
        raise ValueError("No valid decks found in deck database")
    return pool


def _opponent_deck_pool(source: str | None) -> list[tuple[str, str, list[int]]]:
    return [(deck_id, archetype, deck) for deck_id, archetype, deck, _config in _deck_pool(source)]


def _run_eval_game(args: tuple[int, str, str, dict, list[int], list[tuple[str, str, list[int], dict]], list[tuple[str, str, list[int], dict]], dict[str, bool] | None, bool, tuple[int, ...], bool]) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    game, agent_spec, opponent_spec, config, deck, agent_pool, opponent_pool, sequence_options, collect_decisions, seed_schedule, round_robin_opponents = args
    rotation_size = len(opponent_pool) if opponent_pool else 0
    rotation_round = game // rotation_size if rotation_size else game
    agent_side = (game + rotation_round) % 2 if round_robin_opponents and rotation_size else game % 2
    game_seed = seed_schedule[rotation_round % len(seed_schedule)] if seed_schedule else game * 17
    current_agent_deck_id = "default"
    current_agent_archetype = str(config.get("name", "default"))
    agent_deck = deck
    agent_config = config
    if agent_pool:
        deck_id, archetype, pooled_deck, role_config = agent_pool[game % len(agent_pool)]
        current_agent_deck_id = deck_id
        current_agent_archetype = archetype
        agent_deck = pooled_deck
        agent_config = dict(config)
        agent_config.update(role_config)
    opponent_deck_id = "mirror"
    opponent_archetype = config.get("name", "mirror")
    opponent_deck = agent_deck
    opponent_config = agent_config
    if opponent_pool:
        opponent_index = game % len(opponent_pool) if round_robin_opponents else ((game * 7) + 3) % len(opponent_pool)
        opponent_deck_id, opponent_archetype, opponent_deck, opponent_role_config = opponent_pool[opponent_index]
        opponent_config = dict(config)
        opponent_config.update(opponent_role_config)
    agent = make_policy(agent_spec, config=agent_config, seed=game_seed, deck=agent_deck, sequence_options=sequence_options)
    opponent = make_policy(opponent_spec, config=opponent_config, seed=game_seed + 1, deck=opponent_deck, sequence_options=sequence_options)
    base_decision_metadata = {
        "game": int(game),
        "agent_spec": agent_spec,
        "opponent_spec": opponent_spec,
        "agent_deck_id": current_agent_deck_id,
        "agent_archetype": current_agent_archetype,
        "opponent_deck_id": opponent_deck_id,
        "opponent_archetype": opponent_archetype,
    }
    if agent_side == 0:
        result = run_episode(
            agent,
            opponent,
            agent_deck,
            opponent_deck,
            config=agent_config,
            collect_decisions=collect_decisions,
            decision_metadata=base_decision_metadata,
            player_roles={0: {"side": "agent"}, 1: {"side": "opponent"}},
        )
        agent_prizes, opponent_prizes = result.final_prizes
    else:
        result = run_episode(
            opponent,
            agent,
            opponent_deck,
            agent_deck,
            config=agent_config,
            collect_decisions=collect_decisions,
            decision_metadata=base_decision_metadata,
            player_roles={0: {"side": "opponent"}, 1: {"side": "agent"}},
        )
        opponent_prizes, agent_prizes = result.final_prizes

    if result.winner == agent_side:
        outcome = "win"
    elif result.winner == 2:
        outcome = "draw"
    else:
        outcome = "loss"

    prize_diff = opponent_prizes - agent_prizes
    row = {
        "game": game,
        "agent_side": agent_side,
        "agent_deck_id": current_agent_deck_id,
        "agent_archetype": current_agent_archetype,
        "opponent_deck_id": opponent_deck_id,
        "opponent_archetype": opponent_archetype,
        "winner": result.winner,
        "result": outcome,
        "seed": game_seed,
        "steps": result.steps,
        "invalid_actions": result.invalid_actions,
        "agent_prizes": agent_prizes,
        "opponent_prizes": opponent_prizes,
        "prize_diff": prize_diff,
    }
    coverage_rows: list[dict[str, object]] = []
    if sequence_options is not None:
        agent_coverage = _policy_coverage(agent)
        opponent_coverage = _policy_coverage(opponent)
        if agent_coverage:
            coverage_rows.append({"game": game, "side": "agent", **agent_coverage})
        if opponent_coverage:
            coverage_rows.append({"game": game, "side": "opponent", **opponent_coverage})
    return row, coverage_rows, result.decisions


def _worker_count(workers: int | None) -> int:
    if workers is None or workers <= 1:
        return 1
    return max(1, min(int(workers), os.cpu_count() or 1))


def evaluate(
    agent_spec: str,
    opponent_spec: str,
    games: int,
    *,
    config_path: str | None = None,
    out_dir: str = "artifacts/eval",
    agent_deck_id: str | None = None,
    agent_decks: str | None = None,
    opponent_decks: str | None = None,
    sequence_options: dict[str, bool] | None = None,
    workers: int = 1,
    decision_log: bool = False,
    seed_schedule: tuple[int, ...] = (),
    round_robin_opponents: bool = False,
) -> dict:
    config = load_config(config_path)
    deck = read_deck(config=config)
    agent_pool = _single_deck_pool(agent_deck_id) or _deck_pool(agent_decks)
    opponent_pool = _deck_pool(opponent_decks)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"{_slug(agent_spec)}_vs_{_slug(opponent_spec)}"
    jsonl_path = out_path / f"eval_{name}_{stamp}.jsonl"
    csv_path = out_path / f"eval_{name}_{stamp}.csv"
    decision_jsonl_path = out_path / f"decisions_{name}_{stamp}.jsonl" if decision_log else None
    wins = losses = draws = invalid = total_steps = 0
    prize_diffs: list[int] = []
    coverage_rows: list[dict[str, object]] = []
    worker_count = _worker_count(workers)
    game_args = [
        (
            game,
            agent_spec,
            opponent_spec,
            config,
            deck,
            agent_pool,
            opponent_pool,
            sequence_options,
            decision_log,
            tuple(seed_schedule),
            bool(round_robin_opponents),
        )
        for game in range(games)
    ]
    if worker_count > 1 and games > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(_run_eval_game, game_args))
    else:
        results = [_run_eval_game(args) for args in game_args]
    results.sort(key=lambda item: int(item[0]["game"]))

    with jsonl_path.open("w", encoding="utf-8") as jsonl, csv_path.open("w", encoding="utf-8", newline="") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=EVAL_FIELDNAMES)
        writer.writeheader()
        for row, game_coverage_rows, _game_decision_rows in results:
            if row["result"] == "win":
                wins += 1
            elif row["result"] == "draw":
                draws += 1
            else:
                losses += 1

            prize_diffs.append(int(row["prize_diff"]))
            invalid += int(row["invalid_actions"])
            total_steps += int(row["steps"])
            coverage_rows.extend(game_coverage_rows)
            writer.writerow(row)
            jsonl.write(json.dumps(row) + "\n")
    if decision_jsonl_path is not None:
        with decision_jsonl_path.open("w", encoding="utf-8") as decision_fh:
            for _row, _game_coverage_rows, game_decision_rows in results:
                for decision_row in game_decision_rows:
                    decision_fh.write(json.dumps(decision_row, sort_keys=True) + "\n")

    summary = {
        "agent": agent_spec,
        "opponent": opponent_spec,
        "agent_deck_id": agent_deck_id,
        "agent_decks": agent_decks or ("single" if agent_deck_id else "default"),
        "opponent_decks": opponent_decks or "mirror",
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "invalid_actions": invalid,
        "average_steps": total_steps / games if games else 0.0,
        "average_prize_diff": sum(prize_diffs) / games if games else 0.0,
        "sequence_options": sequence_options or {},
        "seed_schedule": list(seed_schedule),
        "round_robin_opponents": bool(round_robin_opponents),
        "workers": worker_count,
        "planner_coverage": _coverage_summary(coverage_rows),
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "decision_jsonl": str(decision_jsonl_path) if decision_jsonl_path is not None else None,
    }
    print(json.dumps(summary, indent=2))
    return summary


def _policy_coverage(policy: object) -> dict[str, object]:
    engine = getattr(policy, "engine", None)
    coverage = getattr(engine, "last_planning_coverage", None)
    if coverage is None:
        return {}
    return {
        "num_legal_first_actions": coverage.num_legal_first_actions,
        "num_first_actions_represented": coverage.num_first_actions_represented,
        "missing_first_action_count": len(coverage.missing_first_actions),
        "num_candidates": coverage.num_candidates,
        "node_budget_hit": coverage.node_budget_hit,
        "time_budget_hit": coverage.time_budget_hit,
    }


def _coverage_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    represented = [float(row["num_first_actions_represented"]) for row in rows]
    legal = [max(1.0, float(row["num_legal_first_actions"])) for row in rows]
    missing = [float(row["missing_first_action_count"]) for row in rows]
    return {
        "records": len(rows),
        "average_first_action_coverage": sum(value / denom for value, denom in zip(represented, legal)) / len(rows),
        "average_missing_first_actions": sum(missing) / len(missing),
        "node_budget_hits": sum(1 for row in rows if row["node_budget_hit"]),
        "time_budget_hits": sum(1 for row in rows if row["time_budget_hit"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pokemon TCG battle policies.")
    parser.add_argument("--agent", default="sequence")
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default="artifacts/eval")
    parser.add_argument("--agent-deck-id", default=None, help="Use one deck_db deck id for the agent deck.")
    parser.add_argument("--agent-decks", default=None, choices=["mirror", "deck_db", "top_winrate"], help="Use default deck, all deck_db decks, or replay-observed top winrate decks for the agent.")
    parser.add_argument("--opponent-decks", default=None, choices=["mirror", "deck_db", "top_winrate"], help="Use mirror deck, all deck_db decks, or replay-observed top winrate decks.")
    parser.add_argument("--disable-probability", action="store_true")
    parser.add_argument("--disable-mlp", action="store_true")
    parser.add_argument("--disable-value-proxy", action="store_true")
    parser.add_argument("--disable-fallback-guardrail", action="store_true")
    parser.add_argument("--planner-coverage-report", action="store_true")
    parser.add_argument("--decision-log", action="store_true", help="Write structured per-decision JSONL diagnostics.")
    parser.add_argument("--num-hidden-samples", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="Run games across this many worker processes.")
    parser.add_argument("--seed-list", default="", help="Comma-separated per-round policy seeds. The first seed is used for the first opponent rotation; the bundled simulator has no game-RNG seed API.")
    parser.add_argument("--round-robin-opponents", action="store_true", help="Cycle opponent decks in exact pool order and alternate side by full opponent rotation.")
    args = parser.parse_args()
    sequence_options = {
        "probability_features_enabled": not args.disable_probability,
        "mlp_enabled": not args.disable_mlp,
        "value_proxy_enabled": not args.disable_value_proxy,
        "fallback_guardrail_enabled": not args.disable_fallback_guardrail,
        "num_hidden_samples": max(1, int(args.num_hidden_samples)),
    }
    if not any((args.disable_probability, args.disable_mlp, args.disable_value_proxy, args.disable_fallback_guardrail, args.planner_coverage_report, args.num_hidden_samples != 1)):
        sequence_options = {}
    summary = evaluate(
        args.agent,
        args.opponent,
        args.games,
        config_path=args.config,
        out_dir=args.out_dir,
        agent_deck_id=args.agent_deck_id,
        agent_decks=args.agent_decks,
        opponent_decks=args.opponent_decks,
        sequence_options=sequence_options,
        workers=args.workers,
        decision_log=args.decision_log,
        seed_schedule=tuple(int(value) for value in args.seed_list.split(",") if value.strip()) if args.seed_list else (),
        round_robin_opponents=args.round_robin_opponents,
    )
    if args.planner_coverage_report:
        print(json.dumps({"planner_coverage_report": summary}, indent=2))


if __name__ == "__main__":
    main()
