#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-./.venv/bin/python}"
GAMES="${GAMES:-25}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

PREVIOUS_AGENT="${PREVIOUS_AGENT:-sequence}"
CURRENT_AGENT="${CURRENT_AGENT:-sequence-mlp:artifacts/checkpoints/top_replay_mlp_diagnostic_e10/best.json}"

OUT_ROOT="${OUT_ROOT:-artifacts/baseline/compare_25_${RUN_ID}}"

run_eval() {
  local label="$1"
  local agent="$2"
  local opponent="$3"
  local deck_pool="$4"
  local out_dir="$OUT_ROOT/$label"

  local cmd=(
    "$PYTHON" -m pokemon_agent.eval
    --agent "$agent"
    --opponent "$opponent"
    --games "$GAMES"
    --out-dir "$out_dir"
  )

  if [[ -n "$deck_pool" ]]; then
    cmd+=(--opponent-decks "$deck_pool")
  fi

  printf '\n==> [%s] %s vs %s %s (%s games)\n' "$label" "$agent" "$opponent" "$deck_pool" "$GAMES"
  "${cmd[@]}"
}

mkdir -p "$OUT_ROOT"

printf 'Output root: %s\n' "$OUT_ROOT"
printf 'Previous:    %s\n' "$PREVIOUS_AGENT"
printf 'Current:     %s\n' "$CURRENT_AGENT"
printf 'Games:       %s per matchup\n' "$GAMES"

for label_agent in \
  "previous::$PREVIOUS_AGENT" \
  "current::$CURRENT_AGENT"
do
  label="${label_agent%%::*}"
  agent="${label_agent#*::}"

  run_eval "$label/random_mirror" "$agent" random ""
  run_eval "$label/random_deck_db" "$agent" random deck_db
  run_eval "$label/heuristic_mirror" "$agent" heuristic ""
  run_eval "$label/deck_heuristic_top_winrate" "$agent" deck-heuristic top_winrate
  run_eval "$label/sequence_starmie_froslass" "$agent" sequence-starmie-froslass ""
  run_eval "$label/sequence_mirror" "$agent" sequence ""
  run_eval "$label/sequence_deck_db" "$agent" sequence deck_db
done

printf '\nCompare run complete.\n'
printf 'Artifacts: %s\n' "$OUT_ROOT"
