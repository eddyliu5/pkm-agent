#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Clean architecture training pipeline.
#
# Default run:
#   ./scripts/train_clean_models.sh
#
# Pilot run:
#   MAX_EXAMPLES=1000 SEQ_EPOCHS=2 MLP_EPOCHS=2 ./scripts/train_clean_models.sh
#
# Custom replay source:
#   INPUTS="artifacts/replays/kaggle data/kaggle_replays" ./scripts/train_clean_models.sh

PYTHON="${PYTHON:-./.venv/bin/python}"
INPUTS="${INPUTS:-artifacts/replays/kaggle}"

LABELS="${LABELS:-artifacts/replay_labels/clean_sequence_labels.jsonl}"
COMPACT_LABELS="${COMPACT_LABELS:-artifacts/replay_labels/clean_sequence_labels.top16.compact.jsonl}"

SEQ_OUT="${SEQ_OUT:-artifacts/checkpoints/clean_sequence_linear}"
MLP_OUT="${MLP_OUT:-artifacts/checkpoints/clean_mlp_ranker}"
SEQ_SELF_PLAY_OUT="${SEQ_SELF_PLAY_OUT:-artifacts/checkpoints/clean_sequence_self_play}"
MLP_SELF_PLAY_OUT="${MLP_SELF_PLAY_OUT:-artifacts/checkpoints/clean_mlp_self_play}"

MIN_RATING="${MIN_RATING:-1400}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
MAX_CANDIDATES="${MAX_CANDIDATES:-24}"
COMPACT_MAX_CANDIDATES="${COMPACT_MAX_CANDIDATES:-16}"
BEAM_WIDTH="${BEAM_WIDTH:-32}"
MAX_DEPTH="${MAX_DEPTH:-12}"
MAX_NODES="${MAX_NODES:-512}"
TIME_BUDGET_MS="${TIME_BUDGET_MS:-1000}"

SEQ_EPOCHS="${SEQ_EPOCHS:-5}"
MLP_EPOCHS="${MLP_EPOCHS:-5}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
MLP_MAX_FEATURES="${MLP_MAX_FEATURES:-256}"
RUN_SELF_PLAY="${RUN_SELF_PLAY:-0}"
SELF_PLAY_GAMES="${SELF_PLAY_GAMES:-500}"
SELF_PLAY_CHECKPOINT_EVERY="${SELF_PLAY_CHECKPOINT_EVERY:-100}"

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

split_inputs() {
  # shellcheck disable=SC2206
  INPUT_ARRAY=($INPUTS)
}

mkdir -p \
  "$(dirname "$LABELS")" \
  "$(dirname "$COMPACT_LABELS")" \
  "$SEQ_OUT" \
  "$MLP_OUT" \
  "$SEQ_SELF_PLAY_OUT" \
  "$MLP_SELF_PLAY_OUT"

run "$PYTHON" -m compileall pokemon_agent

split_inputs

label_cmd=(
  "$PYTHON" -m pokemon_agent.replay_pipeline labels
  "${INPUT_ARRAY[@]}"
  --output "$LABELS"
  --min-rating "$MIN_RATING"
  --max-candidates "$MAX_CANDIDATES"
  --beam-width "$BEAM_WIDTH"
  --max-depth "$MAX_DEPTH"
  --max-nodes "$MAX_NODES"
  --time-budget-ms "$TIME_BUDGET_MS"
  --clean
)

if [[ -n "$MAX_EXAMPLES" ]]; then
  label_cmd+=(--max-examples "$MAX_EXAMPLES")
fi

run "${label_cmd[@]}"

run "$PYTHON" -m pokemon_agent.replay_pipeline compact-labels \
  --labels "$LABELS" \
  --output "$COMPACT_LABELS" \
  --max-candidates "$COMPACT_MAX_CANDIDATES"

run "$PYTHON" -m pokemon_agent.replay_pipeline pretrain \
  --labels "$LABELS" \
  --out-dir "$SEQ_OUT" \
  --epochs "$SEQ_EPOCHS" \
  --clean

run "$PYTHON" -m pokemon_agent.replay_pipeline pretrain-mlp \
  --labels "$COMPACT_LABELS" \
  --out-dir "$MLP_OUT" \
  --epochs "$MLP_EPOCHS" \
  --hidden-dim "$HIDDEN_DIM" \
  --max-features "$MLP_MAX_FEATURES"

if [[ "$RUN_SELF_PLAY" == "1" ]]; then
  run "$PYTHON" -m pokemon_agent.replay_pipeline self-play \
    --clean \
    --weights "${SEQ_OUT}/best.json" \
    --out-dir "$SEQ_SELF_PLAY_OUT" \
    --games "$SELF_PLAY_GAMES" \
    --checkpoint-every "$SELF_PLAY_CHECKPOINT_EVERY"

  run "$PYTHON" -m pokemon_agent.replay_pipeline self-play-mlp \
    --clean \
    --model "${MLP_OUT}/best.json" \
    --weights "${SEQ_SELF_PLAY_OUT}/best.json" \
    --out-dir "$MLP_SELF_PLAY_OUT" \
    --games "$SELF_PLAY_GAMES" \
    --checkpoint-every "$SELF_PLAY_CHECKPOINT_EVERY" \
    --linear-score-weight 0.05 \
    --mlp-score-weight 1.0
fi

printf '\nClean training pipeline complete.\n'
printf 'Clean sequence weights: %s\n' "${SEQ_OUT}/best.json"
printf 'Clean MLP ranker:       %s\n' "${MLP_OUT}/best.json"
printf 'Eval agent:             sequence-clean-mlp:%s,%s\n' "${MLP_OUT}/best.json" "${SEQ_OUT}/best.json"
if [[ "$RUN_SELF_PLAY" == "1" ]]; then
  printf 'Self-play sequence:     %s\n' "${SEQ_SELF_PLAY_OUT}/best.json"
  printf 'Self-play MLP:          %s\n' "${MLP_SELF_PLAY_OUT}/best.json"
  printf 'Self-play eval agent:   sequence-clean-mlp:%s,%s\n' "${MLP_SELF_PLAY_OUT}/best.json" "${SEQ_SELF_PLAY_OUT}/best.json"
fi
