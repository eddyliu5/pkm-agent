#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Simple clean-architecture train, package, and Kaggle-submit flow.
#
# Default run:
#   ./scripts/simple_train_package_submit.sh
#
# Safer local smoke without submitting:
#   TRAIN_GAMES=1 SKIP_SUBMIT=1 ./scripts/simple_train_package_submit.sh

PYTHON="${PYTHON:-./.venv/bin/python}"
KAGGLE_COMPETITION="${KAGGLE_COMPETITION:-pokemon-tcg-ai-battle}"
KAGGLE_MESSAGE="${KAGGLE_MESSAGE:-refactor}"

TRAIN_GAMES="${TRAIN_GAMES:-8}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-8}"
TRAIN_OUT="${TRAIN_OUT:-artifacts/checkpoints/refactor_simple_clean_self_play}"
SUBMISSION="${SUBMISSION:-artifacts/submission/refactor_submission.tar.gz}"
WEIGHTS="${WEIGHTS:-${TRAIN_OUT}/best.json}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_SUBMIT="${SKIP_SUBMIT:-0}"

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

find_kaggle() {
  if [[ -n "${KAGGLE:-}" ]]; then
    printf '%s\n' "$KAGGLE"
    return
  fi

  local python_dir
  python_dir="$(dirname "$PYTHON")"
  if [[ -x "${python_dir}/kaggle" ]]; then
    printf '%s\n' "${python_dir}/kaggle"
    return
  fi

  printf '%s\n' "kaggle"
}

mkdir -p "$TRAIN_OUT" "$(dirname "$SUBMISSION")"

run "$PYTHON" -m compileall pokemon_agent

if [[ "$SKIP_TRAIN" != "1" ]]; then
  run "$PYTHON" -m pokemon_agent.replay_pipeline self-play \
    --clean \
    --games "$TRAIN_GAMES" \
    --out-dir "$TRAIN_OUT" \
    --checkpoint-every "$CHECKPOINT_EVERY"
fi

if [[ ! -f "$WEIGHTS" ]]; then
  printf 'Missing weights checkpoint: %s\n' "$WEIGHTS" >&2
  printf 'Run training first, or pass WEIGHTS=/path/to/best.json.\n' >&2
  exit 1
fi

run "$PYTHON" -m pokemon_agent.submission \
  --clean \
  --weights "$WEIGHTS" \
  --output "$SUBMISSION"

if [[ "$SKIP_SUBMIT" != "1" ]]; then
  KAGGLE_CMD="$(find_kaggle)"
  run "$KAGGLE_CMD" competitions submit \
    -c "$KAGGLE_COMPETITION" \
    -f "$SUBMISSION" \
    -m "$KAGGLE_MESSAGE"
fi

printf '\nSimple train/package/submit flow complete.\n'
printf 'Weights:    %s\n' "$WEIGHTS"
printf 'Submission: %s\n' "$SUBMISSION"
if [[ "$SKIP_SUBMIT" != "1" ]]; then
  printf 'Kaggle:     %s (-m "%s")\n' "$KAGGLE_COMPETITION" "$KAGGLE_MESSAGE"
fi
