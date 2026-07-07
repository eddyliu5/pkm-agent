#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Full probability-feature training pipeline.
#
# Default run:
#   ./scripts/train_probability_pipeline.sh
#
# Short pilot run:
#   MAX_EXAMPLES=1000 SEQ_EPOCHS=3 MLP_EPOCHS=3 EVAL_GAMES=50 ./scripts/train_probability_pipeline.sh
#
# Skip slower end steps:
#   SKIP_EVAL=1 SKIP_SUBMISSION=1 ./scripts/train_probability_pipeline.sh

PYTHON="${PYTHON:-./.venv/bin/python}"
INPUTS="${INPUTS:-artifacts/replays/kaggle}"

LABELS="${LABELS:-artifacts/replay_labels/high_elo_sequence_labels_probability.jsonl}"
COMPACT_LABELS="${COMPACT_LABELS:-artifacts/replay_labels/high_elo_sequence_labels_probability.top16.compact.jsonl}"

SEQ_OUT="${SEQ_OUT:-artifacts/checkpoints/replay_pretrain_probability}"
MLP_OUT="${MLP_OUT:-artifacts/checkpoints/mlp_ranker_probability}"

SUBMISSION="${SUBMISSION:-artifacts/submission/mlp_probability_sequence_submission.tar.gz}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-artifacts/eval/probability_pipeline}"

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
MAX_FEATURES="${MAX_FEATURES:-320}"

EVAL_GAMES="${EVAL_GAMES:-100}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_SUBMISSION="${SKIP_SUBMISSION:-0}"

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

mkdir -p \
  "$(dirname "$LABELS")" \
  "$(dirname "$COMPACT_LABELS")" \
  "$SEQ_OUT" \
  "$MLP_OUT" \
  "$EVAL_OUT_DIR" \
  "$(dirname "$SUBMISSION")"

run "$PYTHON" -m compileall pokemon_agent

label_cmd=(
  "$PYTHON" -m pokemon_agent.replay_pipeline labels
  "$INPUTS"
  --output "$LABELS"
  --min-rating "$MIN_RATING"
  --max-candidates "$MAX_CANDIDATES"
  --beam-width "$BEAM_WIDTH"
  --max-depth "$MAX_DEPTH"
  --max-nodes "$MAX_NODES"
  --time-budget-ms "$TIME_BUDGET_MS"
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
  --epochs "$SEQ_EPOCHS"

run "$PYTHON" -m pokemon_agent.replay_pipeline pretrain-mlp \
  --labels "$COMPACT_LABELS" \
  --out-dir "$MLP_OUT" \
  --epochs "$MLP_EPOCHS" \
  --hidden-dim "$HIDDEN_DIM" \
  --max-features "$MAX_FEATURES"

AGENT="sequence-mlp:${MLP_OUT}/best.json,${SEQ_OUT}/best.json"

if [[ "$SKIP_EVAL" != "1" ]]; then
  run "$PYTHON" -m pokemon_agent.eval \
    --agent "$AGENT" \
    --opponent heuristic \
    --games "$EVAL_GAMES" \
    --out-dir "$EVAL_OUT_DIR"

  run "$PYTHON" -m pokemon_agent.eval \
    --agent "$AGENT" \
    --opponent random \
    --opponent-decks deck_db \
    --games "$EVAL_GAMES" \
    --out-dir "$EVAL_OUT_DIR"
fi

if [[ "$SKIP_SUBMISSION" != "1" ]]; then
  run "$PYTHON" -m pokemon_agent.submission \
    --weights "${SEQ_OUT}/best.json" \
    --mlp-model "${MLP_OUT}/best.json" \
    --output "$SUBMISSION"
fi

printf '\nPipeline complete.\n'
printf 'Sequence checkpoint: %s\n' "${SEQ_OUT}/best.json"
printf 'MLP checkpoint:      %s\n' "${MLP_OUT}/best.json"
printf 'Agent:               %s\n' "$AGENT"
if [[ "$SKIP_SUBMISSION" != "1" ]]; then
  printf 'Submission:          %s\n' "$SUBMISSION"
fi
