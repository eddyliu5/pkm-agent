#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Full-from-scratch general training, trained-agent deck selection, package, submit, and baseline flow.
#
# Default run:
#   ./scripts/train_clean_models.sh
#
# Pilot run without Kaggle submit:
#   MAX_EXAMPLES=1000 SEQ_EPOCHS=2 MLP_EPOCHS=2 FAST_DECK_SWEEP_GAMES=90 SKIP_SUBMIT=1 ./scripts/train_clean_models.sh

PYTHON="${PYTHON:-./.venv/bin/python}"
INPUTS="${INPUTS:-artifacts/replays/kaggle}"
RUN_NAME="${RUN_NAME:-full_from_scratch_general}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

LABELS="${LABELS:-artifacts/replay_labels/${RUN_NAME}.sequence_labels.jsonl}"
COMPACT_LABELS="${COMPACT_LABELS:-artifacts/replay_labels/${RUN_NAME}.top24.compact.jsonl}"
SEQ_OUT="${SEQ_OUT:-artifacts/checkpoints/${RUN_NAME}_sequence_linear}"
MLP_OUT="${MLP_OUT:-artifacts/checkpoints/${RUN_NAME}_mlp_ranker}"

MIN_RATING="${MIN_RATING:-1400}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
MAX_CANDIDATES="${MAX_CANDIDATES:-24}"
COMPACT_MAX_CANDIDATES="${COMPACT_MAX_CANDIDATES:-16}"
BEAM_WIDTH="${BEAM_WIDTH:-32}"
MAX_DEPTH="${MAX_DEPTH:-12}"
MAX_NODES="${MAX_NODES:-512}"
TIME_BUDGET_MS="${TIME_BUDGET_MS:-1000}"
LABEL_WORKERS="${LABEL_WORKERS:-1}"

SEQ_EPOCHS="${SEQ_EPOCHS:-5}"
MLP_EPOCHS="${MLP_EPOCHS:-5}"
HIDDEN_DIM="${HIDDEN_DIM:-0}"
HIDDEN_DIM2="${HIDDEN_DIM2:-0}"
MLP_MAX_FEATURES="${MLP_MAX_FEATURES:-512}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.10}"

FAST_DECK_SWEEP_GAMES="${FAST_DECK_SWEEP_GAMES:-500}"
FAST_DECK_SWEEP_AGENT_DECKS="${FAST_DECK_SWEEP_AGENT_DECKS:-top_winrate}"
FAST_DECK_SWEEP_OPPONENT_DECKS="${FAST_DECK_SWEEP_OPPONENT_DECKS:-top_winrate}"
FAST_DECK_SWEEP_OUT="${FAST_DECK_SWEEP_OUT:-artifacts/baseline/${RUN_NAME}_${RUN_ID}_fast_deck_sweep}"
BASELINE_EVAL_GAMES="${BASELINE_EVAL_GAMES:-100}"
BASELINE_EVAL_OUT="${BASELINE_EVAL_OUT:-artifacts/baseline/${RUN_NAME}_${RUN_ID}_selected_deck}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"

SUBMISSION_DIR="${SUBMISSION_DIR:-artifacts/submission}"
KAGGLE_COMPETITION="${KAGGLE_COMPETITION:-pokemon-tcg-ai-battle}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_DECK_SWEEP="${SKIP_DECK_SWEEP:-0}"
SKIP_PACKAGE="${SKIP_PACKAGE:-0}"
SKIP_SUBMIT="${SKIP_SUBMIT:-0}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
BEST_DECK_ID="${BEST_DECK_ID:-}"
MIN_HEURISTIC_WIN_RATE_FOR_SUBMIT="${MIN_HEURISTIC_WIN_RATE_FOR_SUBMIT:-0.50}"

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

split_inputs() {
  # shellcheck disable=SC2206
  INPUT_ARRAY=($INPUTS)
}

select_best_deck() {
  "$PYTHON" -c '
import csv
import glob
import re
import sys
from collections import defaultdict

out_dir = sys.argv[1]
rows = []
for path in glob.glob(f"{out_dir}/eval_*.csv"):
    with open(path, newline="", encoding="utf-8") as fh:
        rows.extend(csv.DictReader(fh))
if not rows:
    raise SystemExit(f"No eval CSV rows found under {out_dir}")

stats = defaultdict(lambda: {"games": 0, "wins": 0, "draws": 0, "prize_diff": 0.0, "archetype": ""})
for row in rows:
    deck_id = row.get("agent_deck_id") or ""
    if not deck_id or deck_id == "default":
        continue
    item = stats[deck_id]
    item["games"] += 1
    item["wins"] += 1 if row.get("result") == "win" else 0
    item["draws"] += 1 if row.get("result") == "draw" else 0
    item["prize_diff"] += float(row.get("prize_diff") or 0.0)
    item["archetype"] = row.get("agent_archetype") or ""
if not stats:
    raise SystemExit("No agent deck stats found in fast sweep output")

deck_id, item = max(
    stats.items(),
    key=lambda pair: (
        pair[1]["wins"] / max(1, pair[1]["games"]),
        -pair[1]["prize_diff"] / max(1, pair[1]["games"]),
        pair[1]["games"],
        pair[0],
    ),
)
slug = re.sub(r"[^A-Za-z0-9_]+", "_", deck_id).strip("_").lower()
print(f"{deck_id}\t{slug}\t{item['\''archetype'\'']}\t{item['\''wins'\'']}/{item['\''games'\'']}\t{item['\''prize_diff'\'']/max(1,item['\''games'\'']):.3f}")
' "$FAST_DECK_SWEEP_OUT"
}

latest_heuristic_win_rate() {
  "$PYTHON" -c '
import csv
import glob
import os
import sys

out_dir = sys.argv[1]
paths = glob.glob(f"{out_dir}/eval_*_vs_heuristic_*.csv")
if not paths:
    raise SystemExit(f"No heuristic baseline CSV found under {out_dir}")
path = max(paths, key=os.path.getmtime)
wins = games = 0
with open(path, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        games += 1
        wins += 1 if row.get("result") == "win" else 0
if games <= 0:
    raise SystemExit(f"No rows found in heuristic baseline CSV: {path}")
print(f"{wins / games:.6f}")
' "$BASELINE_EVAL_OUT"
}

assert_submit_baseline_gate() {
  if [[ "$SKIP_BASELINE" == "1" || "$SKIP_SUBMIT" == "1" || -z "$MIN_HEURISTIC_WIN_RATE_FOR_SUBMIT" ]]; then
    return
  fi
  local heuristic_win_rate
  heuristic_win_rate="$(latest_heuristic_win_rate)"
  "$PYTHON" -c '
import sys

actual = float(sys.argv[1])
minimum = float(sys.argv[2])
if actual < minimum:
    raise SystemExit(
        f"Selected deck heuristic mirror win_rate={actual:.3f} is below "
        f"MIN_HEURISTIC_WIN_RATE_FOR_SUBMIT={minimum:.3f}; refusing to submit. "
        "Set MIN_HEURISTIC_WIN_RATE_FOR_SUBMIT= to disable this gate."
    )
' "$heuristic_win_rate" "$MIN_HEURISTIC_WIN_RATE_FOR_SUBMIT"
}

mkdir -p \
  "$(dirname "$LABELS")" \
  "$(dirname "$COMPACT_LABELS")" \
  "$SEQ_OUT" \
  "$MLP_OUT" \
  "$FAST_DECK_SWEEP_OUT" \
  "$BASELINE_EVAL_OUT" \
  "$SUBMISSION_DIR"

run "$PYTHON" -m compileall pokemon_agent

split_inputs

if [[ "$SKIP_TRAIN" != "1" ]]; then
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
    --workers "$LABEL_WORKERS"
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

  mlp_cmd=(
    "$PYTHON" -m pokemon_agent.replay_pipeline pretrain-mlp
    --labels "$COMPACT_LABELS"
    --out-dir "$MLP_OUT"
    --epochs "$MLP_EPOCHS"
    --hidden-dim "$HIDDEN_DIM"
    --hidden-dim2 "$HIDDEN_DIM2"
    --max-features "$MLP_MAX_FEATURES"
    --feature-schema-version full
    --validation-fraction "$VALIDATION_FRACTION"
  )
  run "${mlp_cmd[@]}"
fi

EVAL_AGENT="sequence-mlp:${MLP_OUT}/best.json,${SEQ_OUT}/best.json"

if [[ "$SKIP_DECK_SWEEP" != "1" && -z "$BEST_DECK_ID" ]]; then
  run "$PYTHON" -m pokemon_agent.eval \
    --agent "$EVAL_AGENT" \
    --agent-decks "$FAST_DECK_SWEEP_AGENT_DECKS" \
    --opponent deck-heuristic \
    --opponent-decks "$FAST_DECK_SWEEP_OPPONENT_DECKS" \
    --games "$FAST_DECK_SWEEP_GAMES" \
    --workers "$EVAL_WORKERS" \
    --out-dir "$FAST_DECK_SWEEP_OUT"
fi

if [[ -z "$BEST_DECK_ID" ]]; then
  BEST_DECK_ROW="$(select_best_deck)"
  IFS=$'\t' read -r BEST_DECK_ID BEST_DECK_SLUG BEST_DECK_ARCHETYPE BEST_DECK_RECORD BEST_DECK_PRIZE_DIFF <<< "$BEST_DECK_ROW"
else
  BEST_DECK_SLUG="$("$PYTHON" -c 'import re,sys; print(re.sub(r"[^A-Za-z0-9_]+", "_", sys.argv[1]).strip("_").lower())' "$BEST_DECK_ID")"
  BEST_DECK_ARCHETYPE="manual"
  BEST_DECK_RECORD="manual"
  BEST_DECK_PRIZE_DIFF="manual"
fi

MODEL_NAME="full_from_scratch_${BEST_DECK_SLUG}"
SUBMISSION="${SUBMISSION:-${SUBMISSION_DIR}/${MODEL_NAME}.tar.gz}"
KAGGLE_MESSAGE="${KAGGLE_MESSAGE:-$MODEL_NAME}"

if [[ "$SKIP_BASELINE" != "1" ]]; then
  run "$PYTHON" -m pokemon_agent.eval \
    --agent "$EVAL_AGENT" \
    --agent-deck-id "$BEST_DECK_ID" \
    --opponent heuristic \
    --games "$BASELINE_EVAL_GAMES" \
    --workers "$EVAL_WORKERS" \
    --out-dir "$BASELINE_EVAL_OUT"

  run "$PYTHON" -m pokemon_agent.eval \
    --agent "$EVAL_AGENT" \
    --agent-deck-id "$BEST_DECK_ID" \
    --opponent random \
    --opponent-decks deck_db \
    --games "$BASELINE_EVAL_GAMES" \
    --workers "$EVAL_WORKERS" \
    --out-dir "$BASELINE_EVAL_OUT"

  run "$PYTHON" -m pokemon_agent.eval \
    --agent "$EVAL_AGENT" \
    --agent-deck-id "$BEST_DECK_ID" \
    --opponent deck-heuristic \
    --opponent-decks top_winrate \
    --games "$BASELINE_EVAL_GAMES" \
    --workers "$EVAL_WORKERS" \
    --out-dir "$BASELINE_EVAL_OUT"
fi

assert_submit_baseline_gate

if [[ "$SKIP_PACKAGE" != "1" ]]; then
  run "$PYTHON" -m pokemon_agent.submission \
    --weights "${SEQ_OUT}/best.json" \
    --mlp-model "${MLP_OUT}/best.json" \
    --deck-id "$BEST_DECK_ID" \
    --output "$SUBMISSION"
fi

if [[ "$SKIP_SUBMIT" != "1" ]]; then
  KAGGLE_CMD="$(find_kaggle)"
  run "$KAGGLE_CMD" competitions submit \
    -c "$KAGGLE_COMPETITION" \
    -f "$SUBMISSION" \
    -m "$KAGGLE_MESSAGE"
fi

printf '\nFull-from-scratch flow complete.\n'
printf 'General run:        %s\n' "$RUN_NAME"
printf 'Sequence weights:   %s\n' "${SEQ_OUT}/best.json"
printf 'MLP ranker:         %s\n' "${MLP_OUT}/best.json"
printf 'Fast trained-agent deck sweep: %s\n' "$FAST_DECK_SWEEP_OUT"
printf 'Fast sweep pools:   self=%s opponent=%s\n' "$FAST_DECK_SWEEP_AGENT_DECKS" "$FAST_DECK_SWEEP_OPPONENT_DECKS"
printf 'Eval workers:       %s\n' "$EVAL_WORKERS"
printf 'Best deck:          %s (%s, %s, avg_prize_diff=%s)\n' "$BEST_DECK_ID" "$BEST_DECK_ARCHETYPE" "$BEST_DECK_RECORD" "$BEST_DECK_PRIZE_DIFF"
printf 'Model name:         %s\n' "$MODEL_NAME"
printf 'Submission:         %s\n' "$SUBMISSION"
printf 'Kaggle message:     %s\n' "$KAGGLE_MESSAGE"
printf 'Regular baselines:  %s\n' "$BASELINE_EVAL_OUT"
printf 'Submit heuristic gate: min_win_rate=%s\n' "${MIN_HEURISTIC_WIN_RATE_FOR_SUBMIT:-disabled}"
