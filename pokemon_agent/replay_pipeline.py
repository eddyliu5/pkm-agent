from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .actions import legal_actions, sanitize_selection
from .cg_loader import ensure_cg_path
from .config import load_config, read_deck
from .deck_db import deck_counts, deck_heuristic_config, expand_deck, iter_decks, load_deck_db
from .decision_engine import DecisionEngine
from .engine import EpisodeResult
from .features import combine_feature_dicts, extract_action_features, score_features
from .mlp_ranker import MLPSequenceRanker, pretrain_mlp_ranker
from .policy import RandomPolicy, load_weights
from .reward import transition_reward
from .sequence import SequenceSearchConfig, default_clean_sequence_weights, default_sequence_weights
from .state import build_agent_state

ensure_cg_path()

from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402


@dataclass(frozen=True)
class ReplayDecision:
    source: str
    episode_id: str
    step_index: int
    player_index: int
    observation: dict[str, Any]
    selected: list[int]
    rating: float | None = None
    self_deck: list[int] | None = None
    opponent_deck: list[int] | None = None


@dataclass(frozen=True)
class ResolvedReplayDeck:
    deck: list[int]
    config: dict[str, Any]
    deck_id: str
    archetype: str
    archetype_name: str
    matched_deck_db: bool


@dataclass
class SequenceTrainingTransition:
    player: int
    reward: float
    features: dict[str, float]
    candidate_features: list[dict[str, float]] | None = None
    label_index: int | None = None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _run_command(command: list[str], *, dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        return {"command": command, "returncode": None, "stdout": "", "stderr": "", "dry_run": True}
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "dry_run": False,
    }


def _manifest_result(result: dict[str, Any], *, limit: int = 2000) -> dict[str, Any]:
    compact = dict(result)
    for key in ("stdout", "stderr"):
        value = str(compact.get(key, ""))
        compact[f"{key}_bytes"] = len(value.encode("utf-8"))
        if len(value) > limit:
            compact[key] = value[:limit] + "\n...[truncated]"
            compact[f"{key}_truncated"] = True
        else:
            compact[f"{key}_truncated"] = False
    return compact


def _kaggle_command(*args: str) -> list[str]:
    executable = shutil.which("kaggle")
    if executable is None:
        venv_kaggle = Path(sys.executable).with_name("kaggle")
        if venv_kaggle.exists():
            executable = str(venv_kaggle)
    if executable is None:
        executable = "kaggle"
    return [executable, *args]


def _parse_table_rows(text: str) -> list[dict[str, str]]:
    text = text.strip()
    if not text:
        return []
    raw_lines = [line for line in text.splitlines() if line.strip()]
    lines = [line for line in raw_lines if not line.lower().startswith("next page token")]
    if not lines:
        return []
    if "," in lines[0]:
        return list(csv.DictReader(lines))
    lines = [line for line in lines if not _is_separator_line(line)]
    if len(lines) < 2:
        return []
    header_index = 0
    for index, line in enumerate(lines):
        lowered = line.lower()
        if "teamid" in lowered or "submission" in lowered or "episode" in lowered or line.lower().startswith("id "):
            header_index = index
            break
    header = lines[header_index].split()
    rows: list[dict[str, str]] = []
    for line in lines[header_index + 1 :]:
        if _is_separator_line(line):
            continue
        parts = line.split(maxsplit=len(header) - 1)
        if len(parts) >= len(header):
            rows.append({key: parts[index] for index, key in enumerate(header)})
    return rows


def _is_separator_line(line: str) -> bool:
    compact = line.replace(" ", "").strip()
    return bool(compact) and set(compact) == {"-"}


def _row_float(row: dict[str, str], *keys: str) -> float:
    lowered = {key.lower(): value for key, value in row.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is None:
            continue
        try:
            return float(str(value).split()[-1])
        except (TypeError, ValueError):
            continue
    return -math.inf


def _row_id(row: dict[str, str], *keys: str) -> str | None:
    lowered = {key.lower(): value for key, value in row.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value:
            return str(value)
    return None


def fetch_kaggle_replays(
    competition: str,
    *,
    out_dir: str = "artifacts/replays/kaggle",
    top_teams: int = 10,
    submissions_per_team: int = 1,
    episodes_per_submission: int = 20,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Best-effort Kaggle CLI fetcher for simulation episode replays."""

    replay_dir = Path(out_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "competition": competition,
        "out_dir": str(replay_dir),
        "top_teams": top_teams,
        "submissions_per_team": submissions_per_team,
        "episodes_per_submission": episodes_per_submission,
        "dry_run": dry_run,
        "commands": [],
        "downloaded_episode_ids": [],
        "errors": [],
    }
    seen_episode_ids: set[str] = set()

    leaderboard = _run_command(_kaggle_command("competitions", "leaderboard", competition, "-s"), dry_run=dry_run)
    manifest["commands"].append(_manifest_result(leaderboard))
    team_rows = _parse_table_rows(leaderboard.get("stdout", ""))
    team_ids = []
    for row in team_rows:
        team_id = _row_id(row, "teamId", "team_id", "id")
        if team_id:
            team_ids.append(team_id)
        if len(team_ids) >= top_teams:
            break

    for team_id in team_ids:
        submissions = _run_command(_kaggle_command("competitions", "team-submissions", team_id), dry_run=dry_run)
        manifest["commands"].append(_manifest_result(submissions))
        submission_rows = _parse_table_rows(submissions.get("stdout", ""))
        submission_rows.sort(key=lambda row: _row_float(row, "publicScore", "score"), reverse=True)
        submission_ids = []
        for row in submission_rows:
            submission_id = _row_id(row, "id", "submissionId", "submission_id")
            if submission_id:
                submission_ids.append(submission_id)
            if len(submission_ids) >= submissions_per_team:
                break

        for submission_id in submission_ids:
            episodes = _run_command(_kaggle_command("competitions", "episodes", submission_id, "-v"), dry_run=dry_run)
            manifest["commands"].append(_manifest_result(episodes))
            episode_rows = _parse_table_rows(episodes.get("stdout", ""))
            episode_ids = []
            for row in episode_rows:
                episode_id = _row_id(row, "id", "episodeId", "episode_id")
                if episode_id:
                    episode_ids.append(episode_id)
                if len(episode_ids) >= episodes_per_submission:
                    break

            for episode_id in episode_ids:
                if episode_id in seen_episode_ids:
                    continue
                seen_episode_ids.add(episode_id)
                replay = _run_command(_kaggle_command("competitions", "replay", episode_id, "-p", str(replay_dir)), dry_run=dry_run)
                manifest["commands"].append(_manifest_result(replay))
                if replay.get("returncode") in (0, None):
                    manifest["downloaded_episode_ids"].append(episode_id)
                else:
                    manifest["errors"].append({"episode_id": episode_id, "stderr": replay.get("stderr", "")})

    manifest_path = replay_dir / f"fetch_manifest_{_utc_stamp()}.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    manifest["manifest_path"] = str(manifest_path)
    print(json.dumps(manifest, indent=2))
    return manifest


def _iter_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in {".json", ".jsonl"}:
                    yield child
        elif path.suffix.lower() in {".json", ".jsonl"}:
            yield path


def _json_records(path: Path) -> Iterable[Any]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, list):
        yield from payload
    else:
        yield payload


def _float_values(value: Any) -> Iterable[float]:
    if isinstance(value, dict):
        for key, item in value.items():
            key_l = str(key).lower()
            if key_l in {"score", "publicscore", "rating", "rankscore", "updatedscore", "elo"}:
                try:
                    yield float(item)
                except (TypeError, ValueError):
                    pass
            yield from _float_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _float_values(item)


def episode_rating(payload: Any) -> float | None:
    values = [value for value in _float_values(payload) if math.isfinite(value)]
    return max(values) if values else None


def _episode_id(payload: Any, path: Path) -> str:
    if isinstance(payload, dict):
        info = payload.get("info")
        if isinstance(info, dict):
            for key in ("EpisodeId", "episodeId", "episode_id"):
                value = info.get(key)
                if value is not None:
                    return str(value)
        for key in ("episodeId", "episode_id", "id"):
            value = payload.get(key)
            if value is not None:
                return str(value)
    stem = path.stem
    if stem.startswith("episode-") and stem.endswith("-replay"):
        return stem.removeprefix("episode-").removesuffix("-replay")
    return stem


def _selected_action(action: Any) -> list[int]:
    if action is None:
        return []
    if isinstance(action, int):
        return [int(action)]
    if isinstance(action, list):
        return [int(value) for value in action if isinstance(value, int)]
    if isinstance(action, dict):
        for key in ("selected", "selection", "action", "indices"):
            if key in action:
                return _selected_action(action[key])
    return []


def _deck_submission_from_action(action: Any) -> list[int] | None:
    if not isinstance(action, list) or len(action) != 60:
        return None
    if not all(isinstance(value, int) for value in action):
        return None
    return [int(value) for value in action]


def _initial_replay_decks(payload: dict[str, Any]) -> tuple[list[int] | None, list[int] | None]:
    decks: list[list[int] | None] = [None, None]
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return (None, None)
    for step in steps:
        if not isinstance(step, list):
            continue
        for player_index, agent_step in enumerate(step[:2]):
            if decks[player_index] is not None or not isinstance(agent_step, dict):
                continue
            deck = _deck_submission_from_action(agent_step.get("action"))
            if deck is not None:
                decks[player_index] = deck
        if decks[0] is not None and decks[1] is not None:
            break
    return (decks[0], decks[1])


def _deck_count_key(cards: Iterable[int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(card_id), int(count)) for card_id, count in Counter(int(card_id) for card_id in cards).items()))


def _deck_record_count_key(deck: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(card_id), int(count)) for card_id, count in deck_counts(deck).items()))


def _synthetic_replay_deck(cards: list[int]) -> dict[str, Any]:
    counts = Counter(int(card_id) for card_id in cards)
    digest = hashlib.sha1(",".join(str(card_id) for card_id in sorted(cards)).encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"replay_unknown_{digest}",
        "name": f"Replay unknown deck {digest}",
        "archetype": "unknown_replay",
        "archetype_name": "Unknown replay deck",
        "counts": [{"id": card_id, "count": count} for card_id, count in sorted(counts.items())],
        "source": {"type": "replay_submission"},
    }


class ReplayDeckResolver:
    def __init__(self, base_config: dict[str, Any]):
        self.base_config = dict(base_config)
        self.db = load_deck_db()
        self.records_by_counts: dict[tuple[tuple[int, int], ...], dict[str, Any]] = {}
        for deck in iter_decks(self.db):
            self.records_by_counts.setdefault(_deck_record_count_key(deck), deck)
        self.cache: dict[tuple[tuple[int, int], ...], ResolvedReplayDeck] = {}

    def resolve(self, cards: list[int] | None) -> ResolvedReplayDeck | None:
        if not cards:
            return None
        key = _deck_count_key(cards)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        matched = self.records_by_counts.get(key)
        deck_record = matched or _synthetic_replay_deck(cards)
        derived_config = deck_heuristic_config(deck_record, self.db)
        config = dict(self.base_config)
        config.update(derived_config)
        config["replay_deck_matched_db"] = matched is not None
        resolved = ResolvedReplayDeck(
            deck=expand_deck(deck_record) if matched is not None else [int(card_id) for card_id in cards],
            config=config,
            deck_id=str(deck_record.get("id", "")),
            archetype=str(deck_record.get("archetype", "")),
            archetype_name=str(deck_record.get("archetype_name", deck_record.get("archetype", ""))),
            matched_deck_db=matched is not None,
        )
        self.cache[key] = resolved
        return resolved


def _deck_metadata(prefix: str, resolved: ResolvedReplayDeck | None) -> dict[str, Any]:
    if resolved is None:
        return {}
    return {
        f"{prefix}_deck_id": resolved.deck_id,
        f"{prefix}_archetype": resolved.archetype,
        f"{prefix}_archetype_name": resolved.archetype_name,
        f"{prefix}_deck_matched_db": resolved.matched_deck_db,
    }


def _selection_match_key(selected: list[int]) -> tuple[int, ...]:
    return tuple(sorted(int(index) for index in selected))


def _observation_dict(step: dict[str, Any]) -> dict[str, Any] | None:
    obs = step.get("observation") or step.get("obs")
    if isinstance(obs, dict) and isinstance(obs.get("observation"), dict):
        obs = obs["observation"]
    return obs if isinstance(obs, dict) else None


def iter_replay_decisions(paths: Iterable[str | Path], *, min_rating: float | None = None) -> Iterable[ReplayDecision]:
    for path in _iter_files(paths):
        for payload in _json_records(path):
            rating = episode_rating(payload)
            if min_rating is not None and rating is not None and rating < min_rating:
                continue
            episode_id = _episode_id(payload, path)
            if not isinstance(payload, dict):
                continue
            steps = payload.get("steps")
            if not isinstance(steps, list):
                continue
            replay_decks = _initial_replay_decks(payload)
            pending_observations: dict[int, tuple[int, dict[str, Any]]] = {}
            for step_index, step in enumerate(steps):
                if isinstance(step, dict):
                    agents = [step]
                elif isinstance(step, list):
                    agents = step
                else:
                    continue
                for player_index, agent_step in enumerate(agents):
                    if not isinstance(agent_step, dict):
                        continue
                    obs = _observation_dict(agent_step)
                    selected = _selected_action(agent_step.get("action"))
                    if selected:
                        pending = pending_observations.pop(player_index, None)
                        decision_step = pending[0] if pending is not None else step_index
                        decision_obs = pending[1] if pending is not None else obs
                        if decision_obs is not None:
                            yield ReplayDecision(
                                source=str(path),
                                episode_id=episode_id,
                                step_index=decision_step,
                                player_index=player_index,
                                observation=decision_obs,
                                selected=selected,
                                rating=rating,
                                self_deck=replay_decks[player_index] if player_index < len(replay_decks) else None,
                                opponent_deck=replay_decks[1 - player_index] if player_index in (0, 1) else None,
                            )
                    if isinstance(obs, dict) and obs.get("select") is not None:
                        pending_observations[player_index] = (step_index, obs)


def _compact_features(features: dict[str, float]) -> dict[str, float]:
    excluded = {"linear_score", "mlp_score"}
    return {str(key): float(value) for key, value in features.items() if value and str(key) not in excluded}


def _candidate_record(sequence: Any) -> dict[str, Any]:
    return {
        "selected": list(sequence.first_selection),
        "score": float(sequence.score),
        "labels": [label for step in sequence.steps for label in step.labels],
        "features": _compact_features(sequence.features),
    }


def _candidate_coverage_record(sequence: Any) -> dict[str, Any]:
    labels = [label for step in sequence.steps for label in step.labels]
    return {
        "selected": list(sequence.first_selection),
        "labels": labels,
        "tags": list(getattr(sequence, "generation_tags", ())),
        "used_search": bool(getattr(sequence, "used_search", False)),
    }


def build_candidate_coverage_dataset(
    inputs: list[str],
    *,
    output: str = "artifacts/replay_labels/candidate_coverage.jsonl",
    config_path: str | None = None,
    weights_path: str | None = None,
    min_rating: float | None = None,
    max_examples: int | None = None,
    beam_width: int = 16,
    max_depth: int = 10,
    max_nodes: int = 256,
    time_budget_ms: int = 1000,
    num_hidden_samples: int = 1,
    multi_self_decks: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    default_deck = read_deck(config=config)
    weights = load_weights(weights_path) if weights_path else None
    search_config = SequenceSearchConfig(
        beam_width=beam_width,
        max_depth=max_depth,
        max_nodes=max_nodes,
        time_budget_ms=time_budget_ms,
        num_hidden_samples=max(1, int(num_hidden_samples)),
    )
    resolver = ReplayDeckResolver(config) if multi_self_decks else None
    engine_cache: dict[tuple[int, str], DecisionEngine] = {}
    active_episode: tuple[str, str] | None = None

    def engine_for(decision: ReplayDecision) -> tuple[DecisionEngine, ResolvedReplayDeck | None, ResolvedReplayDeck | None]:
        nonlocal active_episode
        episode_key = (decision.source, decision.episode_id)
        if active_episode != episode_key:
            engine_cache.clear()
            active_episode = episode_key
        resolved = resolver.resolve(decision.self_deck) if resolver is not None else None
        opponent_resolved = resolver.resolve(decision.opponent_deck) if resolver is not None else None
        cache_key = (decision.player_index, resolved.deck_id if resolved is not None else "__default__")
        engine = engine_cache.get(cache_key)
        if engine is None:
            engine = DecisionEngine(
                deck=resolved.deck if resolved is not None else default_deck,
                config=resolved.config if resolved is not None else config,
                weights=weights,
                search_config=search_config,
            )
            engine_cache[cache_key] = engine
        return engine, resolved, opponent_resolved

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "inputs": inputs,
        "output": str(output_path),
        "examples": 0,
        "expert_action_in_candidates": 0,
        "expert_action_missing": 0,
        "parse_errors": 0,
        "candidate_errors": 0,
        "expert_action_in_candidates_rate": 0.0,
        "candidate_coverage_by_action_type": {},
        "multi_self_decks": multi_self_decks,
        "replay_decks_matched_db": 0,
        "replay_decks_unknown": 0,
        "replay_decks_missing": 0,
    }
    action_type_totals: dict[str, int] = {}
    action_type_hits: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as fh:
        for decision in iter_replay_decisions(inputs, min_rating=min_rating):
            if max_examples is not None and stats["examples"] >= max_examples:
                break
            try:
                obs = to_observation_class(decision.observation)
            except Exception:
                stats["parse_errors"] += 1
                continue
            if obs.select is None:
                continue
            selected, invalid = sanitize_selection(obs, decision.selected)
            if invalid:
                stats["expert_action_missing"] += 1
                continue
            engine, resolved_deck, opponent_resolved = engine_for(decision)
            if resolved_deck is None:
                stats["replay_decks_missing"] += 1
            elif resolved_deck.matched_deck_db:
                stats["replay_decks_matched_db"] += 1
            else:
                stats["replay_decks_unknown"] += 1
            try:
                observed = engine.observe(obs)
                planning = engine.plan(observed)
            except Exception:
                stats["candidate_errors"] += 1
                continue
            candidates = list(planning.candidates)
            if not candidates:
                stats["candidate_errors"] += 1
                continue
            selected_key = _selection_match_key(selected)
            matched = any(_selection_match_key(sequence.first_selection) == selected_key for sequence in candidates)
            selected_actions = [action for action in observed.actions if action.index in selected]
            action_type = "+".join(action.label for action in selected_actions) or "unknown"
            action_type_totals[action_type] = action_type_totals.get(action_type, 0) + 1
            if matched:
                action_type_hits[action_type] = action_type_hits.get(action_type, 0) + 1
            coverage = planning.coverage
            record = {
                "source": decision.source,
                "episode_id": decision.episode_id,
                "step_index": decision.step_index,
                "player_index": decision.player_index,
                "rating": decision.rating,
                "selected": selected,
                "matched": matched,
                "action_type": action_type,
                "coverage": {
                    "num_legal_first_actions": coverage.num_legal_first_actions if coverage else 0,
                    "num_first_actions_represented": coverage.num_first_actions_represented if coverage else 0,
                    "missing_first_actions": [list(item) for item in coverage.missing_first_actions] if coverage else [],
                    "num_candidates": coverage.num_candidates if coverage else len(candidates),
                    "node_budget_hit": coverage.node_budget_hit if coverage else False,
                    "time_budget_hit": coverage.time_budget_hit if coverage else False,
                    "candidate_tags": list(coverage.candidate_tags) if coverage else [],
                },
                "hidden_samples_used": list(planning.hidden_samples_used),
                "candidates": [_candidate_coverage_record(sequence) for sequence in candidates],
            }
            record.update(_deck_metadata("self", resolved_deck))
            record.update(_deck_metadata("opponent", opponent_resolved))
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            stats["examples"] += 1
            stats["expert_action_in_candidates" if matched else "expert_action_missing"] += 1
    if stats["examples"]:
        stats["expert_action_in_candidates_rate"] = stats["expert_action_in_candidates"] / stats["examples"]
    stats["candidate_coverage_by_action_type"] = {
        key: {
            "examples": total,
            "matched": action_type_hits.get(key, 0),
            "rate": action_type_hits.get(key, 0) / total if total else 0.0,
        }
        for key, total in sorted(action_type_totals.items())
    }
    print(json.dumps(stats, indent=2))
    return stats


def build_ranking_dataset(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return build_label_dataset(*args, **kwargs)


def build_label_dataset(
    inputs: list[str],
    *,
    output: str = "artifacts/replay_labels/sequence_labels.jsonl",
    config_path: str | None = None,
    weights_path: str | None = None,
    min_rating: float | None = None,
    max_examples: int | None = None,
    max_candidates: int = 24,
    beam_width: int = 16,
    max_depth: int = 10,
    max_nodes: int = 256,
    time_budget_ms: int = 1000,
    clean_evaluator: bool = False,
    multi_self_decks: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    if workers > 1:
        return _build_label_dataset_parallel(
            inputs,
            output=output,
            config_path=config_path,
            weights_path=weights_path,
            min_rating=min_rating,
            max_examples=max_examples,
            max_candidates=max_candidates,
            beam_width=beam_width,
            max_depth=max_depth,
            max_nodes=max_nodes,
            time_budget_ms=time_budget_ms,
            clean_evaluator=clean_evaluator,
            multi_self_decks=multi_self_decks,
            workers=workers,
        )
    return _build_label_dataset_serial(
        inputs,
        output=output,
        config_path=config_path,
        weights_path=weights_path,
        min_rating=min_rating,
        max_examples=max_examples,
        max_candidates=max_candidates,
        beam_width=beam_width,
        max_depth=max_depth,
        max_nodes=max_nodes,
        time_budget_ms=time_budget_ms,
        clean_evaluator=clean_evaluator,
        multi_self_decks=multi_self_decks,
        print_summary=True,
    )


def _label_stats(
    inputs: list[str],
    output: str | Path,
    *,
    clean_evaluator: bool,
    multi_self_decks: bool,
) -> dict[str, Any]:
    return {
        "inputs": inputs,
        "output": str(output),
        "examples": 0,
        "matched": 0,
        "unmatched": 0,
        "parse_errors": 0,
        "candidate_errors": 0,
        "clean_evaluator": clean_evaluator,
        "multi_self_decks": multi_self_decks,
        "replay_decks_matched_db": 0,
        "replay_decks_unknown": 0,
        "replay_decks_missing": 0,
    }


def _build_label_dataset_serial(
    inputs: list[str],
    *,
    output: str = "artifacts/replay_labels/sequence_labels.jsonl",
    config_path: str | None = None,
    weights_path: str | None = None,
    min_rating: float | None = None,
    max_examples: int | None = None,
    max_candidates: int = 24,
    beam_width: int = 16,
    max_depth: int = 10,
    max_nodes: int = 256,
    time_budget_ms: int = 1000,
    clean_evaluator: bool = False,
    multi_self_decks: bool = True,
    print_summary: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    default_deck = read_deck(config=config)
    weights = load_weights(weights_path) if weights_path else None
    search_config = SequenceSearchConfig(beam_width=beam_width, max_depth=max_depth, max_nodes=max_nodes, time_budget_ms=time_budget_ms)
    resolver = ReplayDeckResolver(config) if multi_self_decks else None
    engine_cache: dict[tuple[int, str], DecisionEngine] = {}
    active_episode: tuple[str, str] | None = None

    def engine_for(decision: ReplayDecision) -> tuple[DecisionEngine, ResolvedReplayDeck | None, ResolvedReplayDeck | None]:
        nonlocal active_episode
        episode_key = (decision.source, decision.episode_id)
        if active_episode != episode_key:
            engine_cache.clear()
            active_episode = episode_key
        resolved = resolver.resolve(decision.self_deck) if resolver is not None else None
        opponent_resolved = resolver.resolve(decision.opponent_deck) if resolver is not None else None
        cache_key = (decision.player_index, resolved.deck_id if resolved is not None else "__default__")
        engine = engine_cache.get(cache_key)
        if engine is None:
            engine = DecisionEngine(
                deck=resolved.deck if resolved is not None else default_deck,
                config=resolved.config if resolved is not None else config,
                weights=weights,
                search_config=search_config,
                clean_evaluator=clean_evaluator,
            )
            engine_cache[cache_key] = engine
        return engine, resolved, opponent_resolved

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = _label_stats(inputs, output_path, clean_evaluator=clean_evaluator, multi_self_decks=multi_self_decks)
    with output_path.open("w", encoding="utf-8") as fh:
        for decision in iter_replay_decisions(inputs, min_rating=min_rating):
            if max_examples is not None and stats["examples"] >= max_examples:
                break
            try:
                obs = to_observation_class(decision.observation)
            except Exception:
                stats["parse_errors"] += 1
                continue
            if obs.select is None:
                continue
            selected, invalid = sanitize_selection(obs, decision.selected)
            if invalid:
                stats["unmatched"] += 1
                continue
            engine, resolved_deck, opponent_resolved = engine_for(decision)
            if resolved_deck is None:
                stats["replay_decks_missing"] += 1
            elif resolved_deck.matched_deck_db:
                stats["replay_decks_matched_db"] += 1
            else:
                stats["replay_decks_unknown"] += 1
            try:
                sequences = engine.score_sequences(obs)
            except Exception:
                stats["candidate_errors"] += 1
                continue
            if not sequences:
                stats["candidate_errors"] += 1
                continue
            sequences.sort(key=lambda sequence: sequence.score, reverse=True)
            selected_key = _selection_match_key(selected)
            label_index = next((index for index, sequence in enumerate(sequences) if _selection_match_key(sequence.first_selection) == selected_key), -1)
            kept = sequences[:max_candidates]
            if label_index >= max_candidates:
                kept = kept[:-1] + [sequences[label_index]]
                label_index = len(kept) - 1
            elif label_index >= 0:
                kept = sequences[:max_candidates]
            record = {
                "source": decision.source,
                "episode_id": decision.episode_id,
                "step_index": decision.step_index,
                "player_index": decision.player_index,
                "rating": decision.rating,
                "selected": selected,
                "label_index": label_index,
                "matched": label_index >= 0,
                "candidates": [_candidate_record(sequence) for sequence in kept],
            }
            record.update(_deck_metadata("self", resolved_deck))
            record.update(_deck_metadata("opponent", opponent_resolved))
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            stats["examples"] += 1
            stats["matched" if label_index >= 0 else "unmatched"] += 1
    if print_summary:
        print(json.dumps(stats, indent=2))
    return stats


def _merge_label_stats(
    base: dict[str, Any],
    shard: dict[str, Any],
) -> None:
    for key in (
        "examples",
        "matched",
        "unmatched",
        "parse_errors",
        "candidate_errors",
        "replay_decks_matched_db",
        "replay_decks_unknown",
        "replay_decks_missing",
    ):
        base[key] = int(base.get(key, 0)) + int(shard.get(key, 0))


def _build_label_shard(args: dict[str, Any]) -> dict[str, Any]:
    stats = _build_label_dataset_serial(
        [args["input"]],
        output=args["output"],
        config_path=args["config_path"],
        weights_path=args["weights_path"],
        min_rating=args["min_rating"],
        max_examples=None,
        max_candidates=args["max_candidates"],
        beam_width=args["beam_width"],
        max_depth=args["max_depth"],
        max_nodes=args["max_nodes"],
        time_budget_ms=args["time_budget_ms"],
        clean_evaluator=args["clean_evaluator"],
        multi_self_decks=args["multi_self_decks"],
        print_summary=False,
    )
    stats["shard_index"] = args["shard_index"]
    return stats


def _build_label_dataset_parallel(
    inputs: list[str],
    *,
    output: str,
    config_path: str | None,
    weights_path: str | None,
    min_rating: float | None,
    max_examples: int | None,
    max_candidates: int,
    beam_width: int,
    max_depth: int,
    max_nodes: int,
    time_budget_ms: int,
    clean_evaluator: bool,
    multi_self_decks: bool,
    workers: int,
) -> dict[str, Any]:
    if max_examples is not None:
        raise ValueError("--workers cannot be combined with --max-examples because examples are split by replay file.")
    files = [str(path) for path in _iter_files(inputs)]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = _label_stats(inputs, output_path, clean_evaluator=clean_evaluator, multi_self_decks=multi_self_decks)
    stats["workers"] = workers
    stats["input_files"] = len(files)
    if not files:
        output_path.write_text("", encoding="utf-8")
        print(json.dumps(stats, indent=2))
        return stats

    with tempfile.TemporaryDirectory(prefix="label_shards_", dir=str(output_path.parent)) as tmp_dir:
        shard_args = []
        for index, file_path in enumerate(files):
            shard_args.append(
                {
                    "shard_index": index,
                    "input": file_path,
                    "output": str(Path(tmp_dir) / f"shard_{index:06d}.jsonl"),
                    "config_path": config_path,
                    "weights_path": weights_path,
                    "min_rating": min_rating,
                    "max_candidates": max_candidates,
                    "beam_width": beam_width,
                    "max_depth": max_depth,
                    "max_nodes": max_nodes,
                    "time_budget_ms": time_budget_ms,
                    "clean_evaluator": clean_evaluator,
                    "multi_self_decks": multi_self_decks,
                }
            )
        completed: dict[int, dict[str, Any]] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_build_label_shard, args): args["shard_index"] for args in shard_args}
            for future in concurrent.futures.as_completed(futures):
                shard_index = futures[future]
                shard_stats = future.result()
                completed[shard_index] = shard_stats
                _merge_label_stats(stats, shard_stats)
                print(
                    json.dumps(
                        {
                            "completed_shards": len(completed),
                            "input_files": len(files),
                            "examples": stats["examples"],
                            "matched": stats["matched"],
                            "unmatched": stats["unmatched"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        with output_path.open("w", encoding="utf-8") as out_fh:
            for index in range(len(files)):
                shard_path = Path(tmp_dir) / f"shard_{index:06d}.jsonl"
                with shard_path.open("r", encoding="utf-8") as shard_fh:
                    shutil.copyfileobj(shard_fh, out_fh)
                shard_path.unlink()
    print(json.dumps(stats, indent=2))
    return stats


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    best = max(scores)
    values = [math.exp(max(-40.0, min(40.0, score - best))) for score in scores]
    total = sum(values) or 1.0
    return [value / total for value in values]


def _clip_weight(value: float) -> float:
    return max(-25000.0, min(25000.0, value))


def _label_records(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def compact_label_dataset(
    labels_path: str | Path,
    *,
    output: str | Path,
    matched_only: bool = True,
    keep_candidate_debug: bool = False,
    max_candidates: int | None = None,
    exclude_label_index_zero: bool = False,
) -> dict[str, Any]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "input": str(labels_path),
        "output": str(output_path),
        "input_records": 0,
        "output_records": 0,
        "matched_records": 0,
        "skipped_unmatched": 0,
        "skipped_label_index_zero": 0,
        "candidate_count": 0,
        "max_candidates": max_candidates,
        "exclude_label_index_zero": exclude_label_index_zero,
    }
    with output_path.open("w", encoding="utf-8") as fh:
        for record in _label_records(labels_path):
            stats["input_records"] += 1
            label_index = int(record.get("label_index", -1))
            candidates = record.get("candidates") or []
            matched = bool(record.get("matched", label_index >= 0))
            if matched:
                stats["matched_records"] += 1
            if matched_only and not matched:
                stats["skipped_unmatched"] += 1
                continue
            if label_index < 0 or label_index >= len(candidates):
                stats["skipped_unmatched"] += 1
                continue
            if exclude_label_index_zero and label_index == 0:
                stats["skipped_label_index_zero"] += 1
                continue
            if max_candidates is not None and max_candidates > 0 and len(candidates) > max_candidates:
                if label_index < max_candidates:
                    candidates = candidates[:max_candidates]
                else:
                    candidates = candidates[: max_candidates - 1] + [candidates[label_index]]
                    label_index = len(candidates) - 1
            compact_candidates: list[dict[str, Any]] = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                compact: dict[str, Any] = {"features": _compact_features(candidate.get("features") or {})}
                if keep_candidate_debug:
                    for key in ("selected", "score", "labels"):
                        if key in candidate:
                            compact[key] = candidate[key]
                compact_candidates.append(compact)
            if not compact_candidates:
                continue
            compact_record: dict[str, Any] = {
                "label_index": label_index,
                "matched": matched,
                "candidates": compact_candidates,
            }
            for key in (
                "source",
                "episode_id",
                "step_index",
                "player_index",
                "rating",
                "self_deck_id",
                "self_archetype",
                "self_archetype_name",
                "self_deck_matched_db",
                "opponent_deck_id",
                "opponent_archetype",
                "opponent_archetype_name",
                "opponent_deck_matched_db",
            ):
                if key in record:
                    compact_record[key] = record[key]
            fh.write(json.dumps(compact_record, sort_keys=True, separators=(",", ":")) + "\n")
            stats["output_records"] += 1
            stats["candidate_count"] += len(compact_candidates)
    print(json.dumps(stats, indent=2))
    return stats


def _save_weights(path: Path, weights: dict[str, float], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"weights": weights, "summary": summary}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def pretrain_sequence_scorer(
    labels_path: str,
    *,
    out_dir: str = "artifacts/checkpoints/replay_pretrain",
    epochs: int = 3,
    lr: float = 0.02,
    temperature: float = 350.0,
    weights_path: str | None = None,
    clean_evaluator: bool = False,
) -> dict[str, Any]:
    weights = default_clean_sequence_weights() if clean_evaluator else default_sequence_weights()
    if weights_path:
        weights.update(load_weights(weights_path))
    out_path = Path(out_dir)
    latest_path = out_path / "latest.json"
    best_path = out_path / "best.json"
    best_acc = -1.0
    summary: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        examples = matched = top1 = 0
        total_loss = 0.0
        for record in _label_records(labels_path):
            label_index = int(record.get("label_index", -1))
            candidates = record.get("candidates") or []
            if label_index < 0 or label_index >= len(candidates):
                continue
            candidate_features = [candidate.get("features") or {} for candidate in candidates]
            raw_scores = [score_features(features, weights) / max(1.0, temperature) for features in candidate_features]
            probs = _softmax(raw_scores)
            if not probs:
                continue
            examples += 1
            matched += int(record.get("matched", False))
            top1 += int(max(range(len(raw_scores)), key=lambda index: raw_scores[index]) == label_index)
            total_loss -= math.log(max(1e-9, probs[label_index]))
            for index, features in enumerate(candidate_features):
                coeff = (1.0 if index == label_index else 0.0) - probs[index]
                for key, value in features.items():
                    weights[key] = _clip_weight(weights.get(key, 0.0) + lr * coeff * float(value) * 100.0)

        summary = {
            "epoch": epoch,
            "labels": labels_path,
            "examples": examples,
            "matched": matched,
            "top1_accuracy": top1 / examples if examples else 0.0,
            "average_loss": total_loss / examples if examples else 0.0,
            "out_dir": str(out_path),
            "clean_evaluator": clean_evaluator,
        }
        _save_weights(out_path / f"checkpoint_epoch_{epoch}.json", weights, summary)
        _save_weights(latest_path, weights, summary)
        if summary["top1_accuracy"] >= best_acc:
            best_acc = summary["top1_accuracy"]
            _save_weights(best_path, weights, summary)

    summary["latest_checkpoint"] = str(latest_path)
    summary["best_checkpoint"] = str(best_path)
    print(json.dumps(summary, indent=2))
    return summary


class TrainableSequencePolicy:
    def __init__(
        self,
        *,
        deck: list[int],
        config: dict[str, Any],
        weights: dict[str, float],
        epsilon: float,
        seed: int,
        clean_evaluator: bool = False,
    ):
        self.random_policy = RandomPolicy(seed=seed)
        self.rng = random.Random(seed)
        self.epsilon = epsilon
        self.engine = DecisionEngine(
            deck=deck,
            config=config,
            weights=weights,
            search_config=SequenceSearchConfig(beam_width=8, max_depth=8, max_nodes=128, time_budget_ms=500),
            clean_evaluator=clean_evaluator,
            seed=seed,
        )
        self.last_features: dict[str, float] | None = None
        self.last_candidate_features: list[dict[str, float]] | None = None
        self.last_label_index: int | None = None

    def choose(self, obs: Any) -> list[int]:
        self.last_features = None
        self.last_candidate_features = None
        self.last_label_index = None
        if self.rng.random() < self.epsilon:
            return self.random_policy.choose(obs)
        selected = self.engine.choose(obs, mode="train")
        if self.engine.last_chosen_sequence is not None:
            self.last_features = dict(self.engine.last_chosen_sequence.features)
        return selected


class TrainableMLPSequencePolicy:
    def __init__(
        self,
        *,
        deck: list[int],
        config: dict[str, Any],
        model: MLPSequenceRanker,
        weights: dict[str, float] | None,
        epsilon: float,
        seed: int,
        linear_score_weight: float,
        mlp_score_weight: float,
        clean_evaluator: bool = False,
    ):
        self.random_policy = RandomPolicy(seed=seed)
        self.rng = random.Random(seed)
        self.epsilon = epsilon
        self.engine = DecisionEngine(
            deck=deck,
            config=config,
            weights=weights,
            mlp_ranker=model,
            linear_score_weight=linear_score_weight,
            mlp_score_weight=mlp_score_weight,
            search_config=SequenceSearchConfig(beam_width=8, max_depth=8, max_nodes=128, time_budget_ms=500),
            clean_evaluator=clean_evaluator,
            seed=seed,
        )
        self.last_features: dict[str, float] | None = None
        self.last_candidate_features: list[dict[str, float]] | None = None
        self.last_label_index: int | None = None

    def choose(self, obs: Any) -> list[int]:
        self.last_features = None
        self.last_candidate_features = None
        self.last_label_index = None
        if self.rng.random() < self.epsilon:
            return self.random_policy.choose(obs)
        sequences = self.engine.score_sequences(obs)
        if not sequences:
            return self.random_policy.choose(obs)
        chosen = self.engine.selector.select(sequences, mode="train")
        if chosen is None:
            return self.random_policy.choose(obs)
        selected, invalid = sanitize_selection(obs, chosen.first_selection)
        if invalid:
            return self.random_policy.choose(obs)
        self.last_features = dict(chosen.features)
        self.last_candidate_features = [dict(sequence.features) for sequence in sequences]
        self.last_label_index = next(
            (index for index, sequence in enumerate(sequences) if tuple(sequence.first_selection) == tuple(chosen.first_selection)),
            None,
        )
        return selected


def _sequence_episode(
    policy0: Any,
    policy1: Any,
    deck0: list[int],
    deck1: list[int],
    *,
    config: dict[str, Any],
    max_steps: int = 2000,
) -> tuple[EpisodeResult, list[SequenceTrainingTransition]]:
    obs_dict, start_data = battle_start(deck0, deck1)
    if obs_dict is None:
        raise RuntimeError(f"battle_start failed: player={start_data.errorPlayer} type={start_data.errorType}")
    obs = to_observation_class(obs_dict)
    transitions: list[SequenceTrainingTransition] = []
    invalid_actions = 0
    steps = 0
    try:
        while obs.current is not None and obs.current.result == -1 and steps < max_steps:
            player = obs.current.yourIndex
            policy = policy0 if player == 0 else policy1
            action_views = legal_actions(obs)
            prev_state = build_agent_state(obs, perspective=player, actions=action_views)
            selected_raw = policy.choose(obs)
            selected, invalid = sanitize_selection(obs, selected_raw)
            invalid_actions += int(invalid)
            selected_views = [action for action in action_views if action.index in selected]
            next_obs = to_observation_class(battle_select(selected))
            reward = transition_reward(prev_state, next_obs, player, selected_views)
            features = policy.last_features
            if features is None:
                feature_dicts = [extract_action_features(prev_state, action, config) for action in selected_views]
                features = combine_feature_dicts(feature_dicts)
            transitions.append(SequenceTrainingTransition(player=player, reward=reward, features=dict(features)))
            transitions[-1].candidate_features = getattr(policy, "last_candidate_features", None)
            transitions[-1].label_index = getattr(policy, "last_label_index", None)
            obs = next_obs
            steps += 1
    finally:
        battle_finish()

    final = obs.current
    if final is None:
        winner = -1
        final_prizes = (0, 0)
    else:
        winner = final.result
        final_prizes = (len(final.players[0].prize), len(final.players[1].prize))
    if steps >= max_steps and winner == -1:
        winner = 2
    return EpisodeResult(winner=winner, reason=None, steps=steps, invalid_actions=invalid_actions, final_prizes=final_prizes), transitions


def _update_rl_weights(
    weights: dict[str, float],
    transitions: list[SequenceTrainingTransition],
    *,
    train_player: int,
    outcome: float,
    lr: float,
    gamma: float,
) -> None:
    returns = float(outcome)
    for transition in reversed(transitions):
        if transition.player != train_player:
            continue
        returns = (transition.reward / 100.0) + gamma * returns
        returns = max(-3.0, min(3.0, returns))
        current = score_features(transition.features, weights) / 1000.0
        advantage = max(-3.0, min(3.0, returns - current))
        for key, value in transition.features.items():
            weights[key] = _clip_weight(weights.get(key, 0.0) + lr * advantage * float(value) * 100.0)


def _update_mlp_rl_model(
    model: MLPSequenceRanker,
    transitions: list[SequenceTrainingTransition],
    *,
    train_player: int,
    outcome: float,
    lr: float,
    gamma: float,
    temperature: float,
    l2: float,
    grad_clip: float,
) -> dict[str, float]:
    returns = float(outcome)
    updates = 0
    total_loss = 0.0
    total_advantage = 0.0
    for transition in reversed(transitions):
        if transition.player != train_player:
            continue
        returns = (transition.reward / 100.0) + gamma * returns
        returns = max(-3.0, min(3.0, returns))
        if transition.candidate_features is None or transition.label_index is None:
            continue
        if transition.label_index < 0 or transition.label_index >= len(transition.candidate_features):
            continue
        advantage = max(-3.0, min(3.0, returns))
        if abs(advantage) < 1e-9:
            continue
        loss, _top1, _top3 = model.train_step(
            transition.candidate_features,
            transition.label_index,
            lr=lr,
            temperature=temperature,
            l2=l2,
            grad_clip=grad_clip,
            loss_scale=advantage,
        )
        updates += 1
        total_loss += loss
        total_advantage += advantage
    return {
        "updates": float(updates),
        "average_loss": total_loss / updates if updates else 0.0,
        "average_advantage": total_advantage / updates if updates else 0.0,
    }


def rl_self_play(
    games: int,
    *,
    out_dir: str = "artifacts/checkpoints/sequence_self_play",
    config_path: str | None = None,
    weights_path: str | None = None,
    lr: float = 0.003,
    gamma: float = 0.96,
    epsilon_start: float = 0.15,
    epsilon_end: float = 0.03,
    checkpoint_every: int = 100,
    clean_evaluator: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    deck = read_deck(config=config)
    weights = default_clean_sequence_weights() if clean_evaluator else default_sequence_weights()
    if weights_path:
        weights.update(load_weights(weights_path))
    out_path = Path(out_dir)
    metrics_path = out_path / f"sequence_self_play_{_utc_stamp()}.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    wins = losses = draws = invalid = total_steps = 0
    best_recent = -math.inf
    recent: list[float] = []

    with metrics_path.open("w", encoding="utf-8") as metrics:
        for game in range(1, games + 1):
            frac = game / max(1, games)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac
            train_player = game % 2
            policy0 = TrainableSequencePolicy(
                deck=deck,
                config=config,
                weights=weights,
                epsilon=epsilon,
                seed=game * 101 + 1,
                clean_evaluator=clean_evaluator,
            )
            policy1 = TrainableSequencePolicy(
                deck=deck,
                config=config,
                weights=weights,
                epsilon=epsilon,
                seed=game * 101 + 2,
                clean_evaluator=clean_evaluator,
            )
            result, transitions = _sequence_episode(policy0, policy1, deck, deck, config=config)
            outcome = result.outcome_for(train_player)
            _update_rl_weights(weights, transitions, train_player=train_player, outcome=outcome, lr=lr, gamma=gamma)
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
                "train_player": train_player,
                "winner": result.winner,
                "outcome": outcome,
                "steps": result.steps,
                "invalid_actions": result.invalid_actions,
                "epsilon": epsilon,
            }
            metrics.write(json.dumps(row, sort_keys=True) + "\n")

            if game % checkpoint_every == 0 or game == games:
                recent_score = sum(recent) / len(recent) if recent else 0.0
                summary = {
                    "games": game,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "win_rate": wins / game,
                    "invalid_actions": invalid,
                    "average_steps": total_steps / game,
                    "recent_score": recent_score,
                    "metrics": str(metrics_path),
                    "clean_evaluator": clean_evaluator,
                }
                _save_weights(out_path / f"checkpoint_{game}.json", weights, summary)
                _save_weights(out_path / "latest.json", weights, summary)
                if recent_score >= best_recent:
                    best_recent = recent_score
                    _save_weights(out_path / "best.json", weights, summary)

    summary = {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "invalid_actions": invalid,
        "average_steps": total_steps / games if games else 0.0,
        "metrics": str(metrics_path),
        "best_checkpoint": str(out_path / "best.json"),
        "latest_checkpoint": str(out_path / "latest.json"),
        "clean_evaluator": clean_evaluator,
    }
    print(json.dumps(summary, indent=2))
    return summary


def rl_self_play_mlp(
    games: int,
    *,
    model_path: str,
    out_dir: str = "artifacts/checkpoints/mlp_self_play",
    config_path: str | None = None,
    weights_path: str | None = None,
    lr: float = 0.0005,
    gamma: float = 0.96,
    epsilon_start: float = 0.10,
    epsilon_end: float = 0.02,
    checkpoint_every: int = 100,
    temperature: float = 1.0,
    l2: float = 1e-5,
    grad_clip: float = 5.0,
    linear_score_weight: float = 0.0,
    mlp_score_weight: float = 1.0,
    clean_evaluator: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    deck = read_deck(config=config)
    weights = default_clean_sequence_weights() if clean_evaluator else default_sequence_weights()
    if weights_path:
        weights.update(load_weights(weights_path))
    model = MLPSequenceRanker.from_file(model_path)
    out_path = Path(out_dir)
    metrics_path = out_path / f"mlp_self_play_{_utc_stamp()}.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    wins = losses = draws = invalid = total_steps = 0
    total_updates = 0
    best_recent = -math.inf
    recent: list[float] = []

    with metrics_path.open("w", encoding="utf-8") as metrics:
        for game in range(1, games + 1):
            frac = game / max(1, games)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac
            train_player = game % 2
            policy0 = TrainableMLPSequencePolicy(
                deck=deck,
                config=config,
                model=model,
                weights=weights,
                epsilon=epsilon,
                seed=game * 101 + 1,
                linear_score_weight=linear_score_weight,
                mlp_score_weight=mlp_score_weight,
                clean_evaluator=clean_evaluator,
            )
            policy1 = TrainableMLPSequencePolicy(
                deck=deck,
                config=config,
                model=model,
                weights=weights,
                epsilon=epsilon,
                seed=game * 101 + 2,
                linear_score_weight=linear_score_weight,
                mlp_score_weight=mlp_score_weight,
                clean_evaluator=clean_evaluator,
            )
            result, transitions = _sequence_episode(policy0, policy1, deck, deck, config=config)
            outcome = result.outcome_for(train_player)
            update_stats = _update_mlp_rl_model(
                model,
                transitions,
                train_player=train_player,
                outcome=outcome,
                lr=lr,
                gamma=gamma,
                temperature=temperature,
                l2=l2,
                grad_clip=grad_clip,
            )
            total_updates += int(update_stats["updates"])
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
                "train_player": train_player,
                "winner": result.winner,
                "outcome": outcome,
                "steps": result.steps,
                "invalid_actions": result.invalid_actions,
                "epsilon": epsilon,
                "updates": int(update_stats["updates"]),
                "average_update_loss": update_stats["average_loss"],
                "average_advantage": update_stats["average_advantage"],
            }
            metrics.write(json.dumps(row, sort_keys=True) + "\n")

            if game % checkpoint_every == 0 or game == games:
                recent_score = sum(recent) / len(recent) if recent else 0.0
                summary = {
                    "games": game,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "win_rate": wins / game,
                    "invalid_actions": invalid,
                    "average_steps": total_steps / game,
                    "recent_score": recent_score,
                    "total_updates": total_updates,
                    "metrics": str(metrics_path),
                    "source_model": model_path,
                    "linear_score_weight": linear_score_weight,
                    "mlp_score_weight": mlp_score_weight,
                    "clean_evaluator": clean_evaluator,
                }
                model.save(out_path / f"checkpoint_{game}.json", summary=summary)
                model.save(out_path / "latest.json", summary=summary)
                if recent_score >= best_recent:
                    best_recent = recent_score
                    model.save(out_path / "best.json", summary=summary)

    summary = {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "invalid_actions": invalid,
        "average_steps": total_steps / games if games else 0.0,
        "total_updates": total_updates,
        "metrics": str(metrics_path),
        "best_checkpoint": str(out_path / "best.json"),
        "latest_checkpoint": str(out_path / "latest.json"),
        "clean_evaluator": clean_evaluator,
    }
    print(json.dumps(summary, indent=2))
    return summary


def rl_against_sequence_mlp(
    games: int,
    *,
    model_path: str,
    out_dir: str = "artifacts/checkpoints/mlp_vs_sequence",
    config_path: str | None = None,
    weights_path: str | None = None,
    lr: float = 0.00005,
    gamma: float = 0.96,
    epsilon_start: float = 0.06,
    epsilon_end: float = 0.01,
    checkpoint_every: int = 100,
    temperature: float = 1.0,
    l2: float = 1e-5,
    grad_clip: float = 5.0,
    linear_score_weight: float = 0.0,
    mlp_score_weight: float = 1.0,
) -> dict[str, Any]:
    config = load_config(config_path)
    deck = read_deck(config=config)
    weights = default_sequence_weights()
    if weights_path:
        weights.update(load_weights(weights_path))
    model = MLPSequenceRanker.from_file(model_path)
    out_path = Path(out_dir)
    metrics_path = out_path / f"mlp_vs_sequence_{_utc_stamp()}.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    wins = losses = draws = invalid = total_steps = 0
    total_updates = 0
    best_recent = -math.inf
    recent: list[float] = []

    with metrics_path.open("w", encoding="utf-8") as metrics:
        for game in range(1, games + 1):
            frac = game / max(1, games)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac
            train_player = game % 2
            mlp_policy = TrainableMLPSequencePolicy(
                deck=deck,
                config=config,
                model=model,
                weights=weights,
                epsilon=epsilon,
                seed=game * 101 + 1,
                linear_score_weight=linear_score_weight,
                mlp_score_weight=mlp_score_weight,
            )
            sequence_policy = TrainableSequencePolicy(
                deck=deck,
                config=config,
                weights=weights,
                epsilon=0.0,
                seed=game * 101 + 2,
            )
            if train_player == 0:
                result, transitions = _sequence_episode(mlp_policy, sequence_policy, deck, deck, config=config)
            else:
                result, transitions = _sequence_episode(sequence_policy, mlp_policy, deck, deck, config=config)
            outcome = result.outcome_for(train_player)
            update_stats = _update_mlp_rl_model(
                model,
                transitions,
                train_player=train_player,
                outcome=outcome,
                lr=lr,
                gamma=gamma,
                temperature=temperature,
                l2=l2,
                grad_clip=grad_clip,
            )
            total_updates += int(update_stats["updates"])
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
                "train_player": train_player,
                "winner": result.winner,
                "outcome": outcome,
                "steps": result.steps,
                "invalid_actions": result.invalid_actions,
                "epsilon": epsilon,
                "updates": int(update_stats["updates"]),
                "average_update_loss": update_stats["average_loss"],
                "average_advantage": update_stats["average_advantage"],
            }
            metrics.write(json.dumps(row, sort_keys=True) + "\n")

            if game % checkpoint_every == 0 or game == games:
                recent_score = sum(recent) / len(recent) if recent else 0.0
                summary = {
                    "games": game,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "win_rate": wins / game,
                    "invalid_actions": invalid,
                    "average_steps": total_steps / game,
                    "recent_score": recent_score,
                    "total_updates": total_updates,
                    "metrics": str(metrics_path),
                    "source_model": model_path,
                    "opponent": "sequence",
                    "linear_score_weight": linear_score_weight,
                    "mlp_score_weight": mlp_score_weight,
                }
                model.save(out_path / f"checkpoint_{game}.json", summary=summary)
                model.save(out_path / "latest.json", summary=summary)
                if recent_score >= best_recent:
                    best_recent = recent_score
                    model.save(out_path / "best.json", summary=summary)

    summary = {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "invalid_actions": invalid,
        "average_steps": total_steps / games if games else 0.0,
        "total_updates": total_updates,
        "metrics": str(metrics_path),
        "best_checkpoint": str(out_path / "best.json"),
        "latest_checkpoint": str(out_path / "latest.json"),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay imitation and sequence self-play pipeline.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fetch = sub.add_parser("fetch", help="Download top-team Kaggle replays with the Kaggle CLI.")
    fetch.add_argument("--competition", required=True)
    fetch.add_argument("--out-dir", default="artifacts/replays/kaggle")
    fetch.add_argument("--top-teams", type=int, default=10)
    fetch.add_argument("--submissions-per-team", type=int, default=1)
    fetch.add_argument("--episodes-per-submission", type=int, default=20)
    fetch.add_argument("--dry-run", action="store_true")

    coverage_labels = sub.add_parser("coverage-labels", help="Build planner-only candidate coverage labels from replay decisions.")
    coverage_labels.add_argument("inputs", nargs="+")
    coverage_labels.add_argument("--output", default="artifacts/replay_labels/candidate_coverage.jsonl")
    coverage_labels.add_argument("--config", default=None)
    coverage_labels.add_argument("--weights", default=None)
    coverage_labels.add_argument("--min-rating", type=float, default=None)
    coverage_labels.add_argument("--max-examples", type=int, default=None)
    coverage_labels.add_argument("--beam-width", type=int, default=16)
    coverage_labels.add_argument("--max-depth", type=int, default=10)
    coverage_labels.add_argument("--max-nodes", type=int, default=256)
    coverage_labels.add_argument("--time-budget-ms", type=int, default=1000)
    coverage_labels.add_argument("--num-hidden-samples", type=int, default=1)
    coverage_labels.add_argument("--single-config", action="store_true", help="Ignore replay-submitted decks and use only --config.")

    labels = sub.add_parser("labels", help="Convert replay JSON into sequence ranking labels.")
    labels.add_argument("inputs", nargs="+")
    labels.add_argument("--output", default="artifacts/replay_labels/sequence_labels.jsonl")
    labels.add_argument("--config", default=None)
    labels.add_argument("--weights", default=None)
    labels.add_argument("--min-rating", type=float, default=None)
    labels.add_argument("--max-examples", type=int, default=None)
    labels.add_argument("--max-candidates", type=int, default=24)
    labels.add_argument("--beam-width", type=int, default=16)
    labels.add_argument("--max-depth", type=int, default=10)
    labels.add_argument("--max-nodes", type=int, default=256)
    labels.add_argument("--time-budget-ms", type=int, default=1000)
    labels.add_argument("--clean", action="store_true", help="Generate labels with the clean sequence evaluator feature set.")
    labels.add_argument("--single-config", action="store_true", help="Ignore replay-submitted decks and use only --config.")
    labels.add_argument("--workers", type=int, default=1, help="Generate labels across replay files with this many worker processes.")

    ranking_labels = sub.add_parser("ranking-labels", help="Alias for labels; builds scored ranking labels from replay decisions.")
    ranking_labels.add_argument("inputs", nargs="+")
    ranking_labels.add_argument("--output", default="artifacts/replay_labels/ranking_labels.jsonl")
    ranking_labels.add_argument("--config", default=None)
    ranking_labels.add_argument("--weights", default=None)
    ranking_labels.add_argument("--min-rating", type=float, default=None)
    ranking_labels.add_argument("--max-examples", type=int, default=None)
    ranking_labels.add_argument("--max-candidates", type=int, default=24)
    ranking_labels.add_argument("--beam-width", type=int, default=16)
    ranking_labels.add_argument("--max-depth", type=int, default=10)
    ranking_labels.add_argument("--max-nodes", type=int, default=256)
    ranking_labels.add_argument("--time-budget-ms", type=int, default=1000)
    ranking_labels.add_argument("--clean", action="store_true", help="Generate labels with the clean sequence evaluator feature set.")
    ranking_labels.add_argument("--single-config", action="store_true", help="Ignore replay-submitted decks and use only --config.")
    ranking_labels.add_argument("--workers", type=int, default=1, help="Generate labels across replay files with this many worker processes.")

    pretrain = sub.add_parser("pretrain", help="Supervised pretrain sequence weights from labels.")
    pretrain.add_argument("--labels", required=True)
    pretrain.add_argument("--out-dir", default="artifacts/checkpoints/replay_pretrain")
    pretrain.add_argument("--epochs", type=int, default=3)
    pretrain.add_argument("--lr", type=float, default=0.02)
    pretrain.add_argument("--temperature", type=float, default=350.0)
    pretrain.add_argument("--weights", default=None)
    pretrain.add_argument("--clean", action="store_true", help="Train clean sequence linear weights.")

    compact_labels = sub.add_parser("compact-labels", help="Write a compact MLP-friendly label JSONL.")
    compact_labels.add_argument("--labels", required=True)
    compact_labels.add_argument("--output", required=True)
    compact_labels.add_argument("--include-unmatched", action="store_true")
    compact_labels.add_argument("--keep-candidate-debug", action="store_true")
    compact_labels.add_argument("--max-candidates", type=int, default=None)
    compact_labels.add_argument("--exclude-label-index-zero", action="store_true")

    pretrain_mlp = sub.add_parser("pretrain-mlp", help="Supervised pretrain a tiny MLP sequence ranker from labels.")
    pretrain_mlp.add_argument("--labels", required=True)
    pretrain_mlp.add_argument("--out-dir", default="artifacts/checkpoints/mlp_ranker")
    pretrain_mlp.add_argument("--epochs", type=int, default=3)
    pretrain_mlp.add_argument("--lr", type=float, default=0.001)
    pretrain_mlp.add_argument("--hidden-dim", type=int, default=0, help="First hidden layer width; 0 derives from input feature count.")
    pretrain_mlp.add_argument("--hidden-dim2", type=int, default=0, help="Second hidden layer width; 0 derives from hidden-dim.")
    pretrain_mlp.add_argument("--max-features", type=int, default=256)
    pretrain_mlp.add_argument("--min-feature-count", type=int, default=1)
    pretrain_mlp.add_argument("--max-examples", type=int, default=None)
    pretrain_mlp.add_argument("--seed", type=int, default=13)
    pretrain_mlp.add_argument("--temperature", type=float, default=1.0)
    pretrain_mlp.add_argument("--l2", type=float, default=1e-5)
    pretrain_mlp.add_argument("--input-clip", type=float, default=8.0)
    pretrain_mlp.add_argument("--grad-clip", type=float, default=5.0)
    pretrain_mlp.add_argument("--normalization", choices=("sparse", "standard"), default="sparse")
    pretrain_mlp.add_argument(
        "--feature-schema-version",
        choices=("full", "clean", "sequence_v3_full", "sequence_v3_clean"),
        default="full",
    )
    pretrain_mlp.add_argument("--validation-fraction", type=float, default=0.10)
    pretrain_mlp.add_argument("--include-linear-score", action="store_true", help="Allow linear_score as an MLP input feature. Default excludes it for MLP-only ranking.")

    rl = sub.add_parser("self-play", help="Fine-tune sequence weights with self-play returns.")
    rl.add_argument("--games", type=int, default=500)
    rl.add_argument("--out-dir", default="artifacts/checkpoints/sequence_self_play")
    rl.add_argument("--config", default=None)
    rl.add_argument("--weights", default=None)
    rl.add_argument("--lr", type=float, default=0.003)
    rl.add_argument("--gamma", type=float, default=0.96)
    rl.add_argument("--epsilon-start", type=float, default=0.15)
    rl.add_argument("--epsilon-end", type=float, default=0.03)
    rl.add_argument("--checkpoint-every", type=int, default=100)
    rl.add_argument("--clean", action="store_true", help="Fine-tune clean sequence linear weights.")

    rl_mlp = sub.add_parser("self-play-mlp", help="Fine-tune an MLP sequence ranker with self-play returns.")
    rl_mlp.add_argument("--games", type=int, default=500)
    rl_mlp.add_argument("--model", required=True)
    rl_mlp.add_argument("--out-dir", default="artifacts/checkpoints/mlp_self_play")
    rl_mlp.add_argument("--config", default=None)
    rl_mlp.add_argument("--weights", default=None)
    rl_mlp.add_argument("--lr", type=float, default=0.0005)
    rl_mlp.add_argument("--gamma", type=float, default=0.96)
    rl_mlp.add_argument("--epsilon-start", type=float, default=0.10)
    rl_mlp.add_argument("--epsilon-end", type=float, default=0.02)
    rl_mlp.add_argument("--checkpoint-every", type=int, default=100)
    rl_mlp.add_argument("--temperature", type=float, default=1.0)
    rl_mlp.add_argument("--l2", type=float, default=1e-5)
    rl_mlp.add_argument("--grad-clip", type=float, default=5.0)
    rl_mlp.add_argument("--linear-score-weight", type=float, default=0.0)
    rl_mlp.add_argument("--mlp-score-weight", type=float, default=1.0)
    rl_mlp.add_argument("--clean", action="store_true", help="Fine-tune the MLP inside the clean sequence evaluator.")

    rl_vs_sequence = sub.add_parser("train-mlp-vs-sequence", help="Fine-tune an MLP ranker against a frozen sequence baseline.")
    rl_vs_sequence.add_argument("--games", type=int, default=500)
    rl_vs_sequence.add_argument("--model", required=True)
    rl_vs_sequence.add_argument("--out-dir", default="artifacts/checkpoints/mlp_vs_sequence")
    rl_vs_sequence.add_argument("--config", default=None)
    rl_vs_sequence.add_argument("--weights", default=None)
    rl_vs_sequence.add_argument("--lr", type=float, default=0.00005)
    rl_vs_sequence.add_argument("--gamma", type=float, default=0.96)
    rl_vs_sequence.add_argument("--epsilon-start", type=float, default=0.06)
    rl_vs_sequence.add_argument("--epsilon-end", type=float, default=0.01)
    rl_vs_sequence.add_argument("--checkpoint-every", type=int, default=100)
    rl_vs_sequence.add_argument("--temperature", type=float, default=1.0)
    rl_vs_sequence.add_argument("--l2", type=float, default=1e-5)
    rl_vs_sequence.add_argument("--grad-clip", type=float, default=5.0)
    rl_vs_sequence.add_argument("--linear-score-weight", type=float, default=0.0)
    rl_vs_sequence.add_argument("--mlp-score-weight", type=float, default=1.0)

    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch_kaggle_replays(
            args.competition,
            out_dir=args.out_dir,
            top_teams=args.top_teams,
            submissions_per_team=args.submissions_per_team,
            episodes_per_submission=args.episodes_per_submission,
            dry_run=args.dry_run,
        )
    elif args.cmd == "coverage-labels":
        build_candidate_coverage_dataset(
            args.inputs,
            output=args.output,
            config_path=args.config,
            weights_path=args.weights,
            min_rating=args.min_rating,
            max_examples=args.max_examples,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            max_nodes=args.max_nodes,
            time_budget_ms=args.time_budget_ms,
            num_hidden_samples=args.num_hidden_samples,
            multi_self_decks=not args.single_config,
        )
    elif args.cmd in {"labels", "ranking-labels"}:
        build_label_dataset(
            args.inputs,
            output=args.output,
            config_path=args.config,
            weights_path=args.weights,
            min_rating=args.min_rating,
            max_examples=args.max_examples,
            max_candidates=args.max_candidates,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            max_nodes=args.max_nodes,
            time_budget_ms=args.time_budget_ms,
            clean_evaluator=args.clean,
            multi_self_decks=not args.single_config,
            workers=args.workers,
        )
    elif args.cmd == "pretrain":
        pretrain_sequence_scorer(
            args.labels,
            out_dir=args.out_dir,
            epochs=args.epochs,
            lr=args.lr,
            temperature=args.temperature,
            weights_path=args.weights,
            clean_evaluator=args.clean,
        )
    elif args.cmd == "compact-labels":
        compact_label_dataset(
            args.labels,
            output=args.output,
            matched_only=not args.include_unmatched,
            keep_candidate_debug=args.keep_candidate_debug,
            max_candidates=args.max_candidates,
            exclude_label_index_zero=args.exclude_label_index_zero,
        )
    elif args.cmd == "pretrain-mlp":
        summary = pretrain_mlp_ranker(
            args.labels,
            out_dir=args.out_dir,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            hidden_dim2=args.hidden_dim2,
            max_features=args.max_features,
            min_feature_count=args.min_feature_count,
            max_examples=args.max_examples,
            seed=args.seed,
            temperature=args.temperature,
            l2=args.l2,
            input_clip=args.input_clip,
            grad_clip=args.grad_clip,
            normalization=args.normalization,
            feature_schema_version=args.feature_schema_version,
            validation_fraction=args.validation_fraction,
            exclude_features=() if args.include_linear_score else ("linear_score",),
        )
        print(json.dumps(summary, indent=2))
    elif args.cmd == "self-play":
        rl_self_play(
            args.games,
            out_dir=args.out_dir,
            config_path=args.config,
            weights_path=args.weights,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            checkpoint_every=args.checkpoint_every,
            clean_evaluator=args.clean,
        )
    elif args.cmd == "self-play-mlp":
        rl_self_play_mlp(
            args.games,
            model_path=args.model,
            out_dir=args.out_dir,
            config_path=args.config,
            weights_path=args.weights,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            checkpoint_every=args.checkpoint_every,
            temperature=args.temperature,
            l2=args.l2,
            grad_clip=args.grad_clip,
            linear_score_weight=args.linear_score_weight,
            mlp_score_weight=args.mlp_score_weight,
            clean_evaluator=args.clean,
        )
    elif args.cmd == "train-mlp-vs-sequence":
        rl_against_sequence_mlp(
            args.games,
            model_path=args.model,
            out_dir=args.out_dir,
            config_path=args.config,
            weights_path=args.weights,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            checkpoint_every=args.checkpoint_every,
            temperature=args.temperature,
            l2=args.l2,
            grad_clip=args.grad_clip,
            linear_score_weight=args.linear_score_weight,
            mlp_score_weight=args.mlp_score_weight,
        )


if __name__ == "__main__":
    main()
