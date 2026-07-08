#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-./.venv/bin/python}"
CONTROL_AGENT="${CONTROL_AGENT:-sequence}"
CANDIDATE_AGENT="${CANDIDATE_AGENT:-${CURRENT_AGENT:-}}"
TIER="${TIER:-quick}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-artifacts/baseline/harness_${RUN_ID}}"
CONFIG="${CONFIG:-}"
AGENT_DECK_ID="${AGENT_DECK_ID:-}"

case "$TIER" in
  smoke)
    DEFAULT_GAMES=5
    ;;
  quick)
    DEFAULT_GAMES=32
    ;;
  standard)
    DEFAULT_GAMES=128
    ;;
  confirmation)
    DEFAULT_GAMES=384
    ;;
  *)
    printf 'Unknown TIER=%s. Use smoke, quick, standard, or confirmation.\n' "$TIER" >&2
    exit 2
    ;;
esac

GAMES="${GAMES:-$DEFAULT_GAMES}"

if [[ -z "$CANDIDATE_AGENT" ]]; then
  printf 'CANDIDATE_AGENT is required.\n' >&2
  exit 2
fi

run_eval() {
  local root="$1"
  local suite="$2"
  local agent="$3"
  local opponent="$4"
  local opponent_decks="$5"
  local out_dir="$root/$suite"

  local cmd=(
    "$PYTHON" -m pokemon_agent.eval
    --agent "$agent"
    --opponent "$opponent"
    --games "$GAMES"
    --workers "$EVAL_WORKERS"
    --out-dir "$out_dir"
  )

  if [[ -n "$CONFIG" ]]; then
    cmd+=(--config "$CONFIG")
  fi
  if [[ -n "$AGENT_DECK_ID" ]]; then
    cmd+=(--agent-deck-id "$AGENT_DECK_ID")
  fi
  if [[ -n "$opponent_decks" ]]; then
    cmd+=(--opponent-decks "$opponent_decks")
  fi

  printf '\n==> [%s] %s vs %s %s (%s games)\n' "$suite" "$agent" "$opponent" "$opponent_decks" "$GAMES"
  "${cmd[@]}"
}

write_manifest() {
  local git_head git_dirty
  git_head="$(git rev-parse HEAD 2>/dev/null || true)"
  if [[ -n "$(git status --short 2>/dev/null || true)" ]]; then
    git_dirty=true
  else
    git_dirty=false
  fi

  "$PYTHON" - "$OUT_ROOT/manifest.json" "$RUN_ID" "$TIER" "$CONTROL_AGENT" "$CANDIDATE_AGENT" "$CONFIG" "$AGENT_DECK_ID" "$GAMES" "$EVAL_WORKERS" "$git_head" "$git_dirty" <<'PY'
import json
import sys

(
    out_path,
    run_id,
    tier,
    control_agent,
    candidate_agent,
    config,
    agent_deck_id,
    games,
    workers,
    git_head,
    git_dirty,
) = sys.argv[1:]

manifest = {
    "run_id": run_id,
    "tier": tier,
    "control_agent": control_agent,
    "candidate_agent": candidate_agent,
    "config": config or None,
    "agent_deck_id": agent_deck_id or None,
    "games_per_suite": int(games),
    "workers": int(workers),
    "suite_ids": [
        "random_mirror",
        "random_deck_db",
        "heuristic_mirror",
        "deck_heuristic_top_winrate",
        "sequence_starmie_froslass",
        "sequence_mirror",
        "sequence_deck_db",
    ],
    "git_head": git_head or None,
    "git_dirty": git_dirty == "true",
}
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\\n")
PY
}

mkdir -p "$OUT_ROOT/control" "$OUT_ROOT/candidate"

printf 'Output root:   %s\n' "$OUT_ROOT"
printf 'Tier:          %s\n' "$TIER"
printf 'Control:       %s\n' "$CONTROL_AGENT"
printf 'Candidate:     %s\n' "$CANDIDATE_AGENT"
printf 'Games/suite:   %s\n' "$GAMES"
printf 'Workers:       %s\n' "$EVAL_WORKERS"

"$PYTHON" -m pokemon_agent.tests_smoke
write_manifest

for label_agent in \
  "control::$CONTROL_AGENT" \
  "candidate::$CANDIDATE_AGENT"
do
  label="${label_agent%%::*}"
  agent="${label_agent#*::}"
  root="$OUT_ROOT/$label"

  run_eval "$root" random_mirror "$agent" random ""
  run_eval "$root" random_deck_db "$agent" random deck_db
  run_eval "$root" heuristic_mirror "$agent" heuristic ""
  run_eval "$root" deck_heuristic_top_winrate "$agent" deck-heuristic top_winrate
  run_eval "$root" sequence_starmie_froslass "$agent" sequence-starmie-froslass ""
  run_eval "$root" sequence_mirror "$agent" sequence ""
  run_eval "$root" sequence_deck_db "$agent" sequence deck_db
done

"$PYTHON" -m pokemon_agent.evaluation.baseline_compare \
  --control-dir "$OUT_ROOT/control" \
  --candidate-dir "$OUT_ROOT/candidate" \
  --out-json "$OUT_ROOT/comparison.json"

printf '\nHarness complete.\n'
printf 'Artifacts: %s\n' "$OUT_ROOT"
