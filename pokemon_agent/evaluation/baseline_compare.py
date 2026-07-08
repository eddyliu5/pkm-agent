from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SUITE_WEIGHTS = {
    "random_mirror": 0.10,
    "random_deck_db": 0.20,
    "heuristic_mirror": 0.15,
    "deck_heuristic_top_winrate": 0.15,
    "sequence_starmie_froslass": 0.15,
    "sequence_mirror": 0.10,
    "sequence_deck_db": 0.15,
}

CRITICAL_SUITES = {
    "random_deck_db",
    "deck_heuristic_top_winrate",
    "sequence_starmie_froslass",
    "sequence_deck_db",
}


@dataclass(frozen=True)
class SuiteRows:
    suite_id: str
    rows: list[dict[str, str]]


def _suite_id(root: Path, csv_path: Path) -> str:
    parent = csv_path.parent
    rel = parent.relative_to(root)
    if str(rel) != ".":
        return str(rel).replace("/", "_")

    name = csv_path.name
    if "_vs_deck-heuristic_" in name:
        return "deck_heuristic_top_winrate"
    if "_vs_sequence-starmie-froslass_" in name:
        return "sequence_starmie_froslass"
    if "_vs_sequence_" in name:
        return "sequence_mirror"
    if "_vs_heuristic_" in name:
        return "heuristic_mirror"
    if "_vs_random_" in name:
        return "random_mirror"
    return csv_path.stem


def _read_suites(root: Path) -> dict[str, SuiteRows]:
    suites: dict[str, list[dict[str, str]]] = {}
    for path in sorted(root.rglob("eval_*.csv")):
        suite = _suite_id(root, path)
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        suites.setdefault(suite, []).extend(rows)
    return {suite: SuiteRows(suite, rows) for suite, rows in sorted(suites.items())}


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def _wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return (0.0, 0.0)
    p = wins / games
    denom = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * games)) / games) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    mean = sum(values) / len(values)
    if len(values) == 1:
        return (mean, mean, mean)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = z * math.sqrt(variance / len(values))
    return (mean, mean - margin, mean + margin)


def _summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    games = len(rows)
    wins = sum(1 for row in rows if row.get("result") == "win")
    losses = sum(1 for row in rows if row.get("result") == "loss")
    draws = sum(1 for row in rows if row.get("result") == "draw")
    invalid_actions = sum(int(_float(row, "invalid_actions")) for row in rows)
    steps = [_float(row, "steps") for row in rows]
    prize_diffs = [_float(row, "prize_diff") for row in rows]
    low, high = _wilson_interval(wins, games)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "win_rate_ci95": [low, high],
        "invalid_actions": invalid_actions,
        "average_steps": sum(steps) / games if games else 0.0,
        "average_prize_diff": sum(prize_diffs) / games if games else 0.0,
    }


def _by_game(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        game_raw = row.get("game")
        if game_raw is None:
            continue
        result[int(game_raw)] = row
    return result


def _paired_delta(control_rows: list[dict[str, str]], candidate_rows: list[dict[str, str]]) -> dict[str, Any]:
    control = _by_game(control_rows)
    candidate = _by_game(candidate_rows)
    game_ids = sorted(set(control) & set(candidate))
    win_deltas: list[float] = []
    score_deltas: list[float] = []
    prize_deltas: list[float] = []
    step_deltas: list[float] = []

    def win_value(row: dict[str, str]) -> float:
        return 1.0 if row.get("result") == "win" else 0.0

    def score_value(row: dict[str, str]) -> float:
        if row.get("result") == "win":
            return 1.0
        if row.get("result") == "draw":
            return 0.5
        return 0.0

    for game_id in game_ids:
        c0 = control[game_id]
        c1 = candidate[game_id]
        win_deltas.append(win_value(c1) - win_value(c0))
        score_deltas.append(score_value(c1) - score_value(c0))
        prize_deltas.append(_float(c1, "prize_diff") - _float(c0, "prize_diff"))
        step_deltas.append(_float(c1, "steps") - _float(c0, "steps"))

    win_mean, win_low, win_high = _mean_ci(win_deltas)
    score_mean, score_low, score_high = _mean_ci(score_deltas)
    prize_mean, prize_low, prize_high = _mean_ci(prize_deltas)
    step_mean, step_low, step_high = _mean_ci(step_deltas)
    return {
        "matched_games": len(game_ids),
        "win_rate_delta": win_mean,
        "win_rate_delta_ci95": [win_low, win_high],
        "game_score_delta": score_mean,
        "game_score_delta_ci95": [score_low, score_high],
        "average_prize_diff_delta": prize_mean,
        "average_prize_diff_delta_ci95": [prize_low, prize_high],
        "average_steps_delta": step_mean,
        "average_steps_delta_ci95": [step_low, step_high],
    }


def compare(control_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    control_suites = _read_suites(control_dir)
    candidate_suites = _read_suites(candidate_dir)
    suite_ids = sorted(set(control_suites) | set(candidate_suites))
    suites: dict[str, Any] = {}
    weighted_control = 0.0
    weighted_candidate = 0.0
    used_weight = 0.0
    blockers: list[str] = []
    regressions: list[str] = []

    for suite_id in suite_ids:
        control_rows = control_suites.get(suite_id, SuiteRows(suite_id, [])).rows
        candidate_rows = candidate_suites.get(suite_id, SuiteRows(suite_id, [])).rows
        control_summary = _summarize(control_rows)
        candidate_summary = _summarize(candidate_rows)
        paired = _paired_delta(control_rows, candidate_rows)
        aggregate_delta = candidate_summary["win_rate"] - control_summary["win_rate"]
        weight = DEFAULT_SUITE_WEIGHTS.get(suite_id, 0.0)
        if weight and control_summary["games"] and candidate_summary["games"]:
            weighted_control += weight * control_summary["win_rate"]
            weighted_candidate += weight * candidate_summary["win_rate"]
            used_weight += weight

        if candidate_summary["games"] == 0:
            blockers.append(f"{suite_id}: missing candidate rows")
        if control_summary["games"] == 0:
            blockers.append(f"{suite_id}: missing control rows")
        if candidate_summary["invalid_actions"] > 0:
            blockers.append(f"{suite_id}: candidate invalid_actions={candidate_summary['invalid_actions']}")
        if suite_id in CRITICAL_SUITES and paired["matched_games"] > 1 and paired["win_rate_delta_ci95"][0] < -0.02:
            regressions.append(
                f"{suite_id}: paired win-rate delta lower bound {paired['win_rate_delta_ci95'][0]:.3f}"
            )

        suites[suite_id] = {
            "weight": weight,
            "critical": suite_id in CRITICAL_SUITES,
            "control": control_summary,
            "candidate": candidate_summary,
            "aggregate_win_rate_delta": aggregate_delta,
            "paired": paired,
        }

    if used_weight:
        weighted_control = weighted_control / used_weight * 100.0
        weighted_candidate = weighted_candidate / used_weight * 100.0

    return {
        "control_dir": str(control_dir),
        "candidate_dir": str(candidate_dir),
        "weighted_score": {
            "control": weighted_control,
            "candidate": weighted_candidate,
            "delta": weighted_candidate - weighted_control,
            "used_weight": used_weight,
        },
        "blockers": blockers,
        "critical_regressions": regressions,
        "suites": suites,
    }


def _print_text(report: dict[str, Any]) -> None:
    weighted = report["weighted_score"]
    print(
        "weighted_score: "
        f"control={weighted['control']:.2f} "
        f"candidate={weighted['candidate']:.2f} "
        f"delta={weighted['delta']:+.2f}"
    )
    if report["blockers"]:
        print("blockers:")
        for item in report["blockers"]:
            print(f"  - {item}")
    if report["critical_regressions"]:
        print("critical_regressions:")
        for item in report["critical_regressions"]:
            print(f"  - {item}")
    print()
    print("suite,control_wr,candidate_wr,delta,paired_delta,paired_ci95,invalid,matched")
    for suite_id, suite in report["suites"].items():
        paired = suite["paired"]
        print(
            f"{suite_id},"
            f"{suite['control']['win_rate']:.3f},"
            f"{suite['candidate']['win_rate']:.3f},"
            f"{suite['aggregate_win_rate_delta']:+.3f},"
            f"{paired['win_rate_delta']:+.3f},"
            f"[{paired['win_rate_delta_ci95'][0]:+.3f},{paired['win_rate_delta_ci95'][1]:+.3f}],"
            f"{suite['candidate']['invalid_actions']},"
            f"{paired['matched_games']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired Pokemon-Agent baseline artifacts.")
    parser.add_argument("--control-dir", required=True, help="Directory containing control eval_*.csv artifacts.")
    parser.add_argument("--candidate-dir", required=True, help="Directory containing candidate eval_*.csv artifacts.")
    parser.add_argument("--out-json", default=None, help="Optional path for the machine-readable comparison report.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    report = compare(Path(args.control_dir), Path(args.candidate_dir))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)


if __name__ == "__main__":
    main()
