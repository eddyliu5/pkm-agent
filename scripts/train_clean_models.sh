#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Full-from-scratch general training, trained-agent deck selection, package, submit, and baseline flow.
# The MLP is trained with PyTorch by default and writes best.pt. Runtime defaults to
# auto policy selection: sequence for deck sweep, then sequence-vs-MLP selected-deck
# baselines before packaging.
#
# Default run:
#   ./scripts/train_clean_models.sh
#
# Pilot run without Kaggle submit:
#   MAX_EXAMPLES=1000 SEQ_EPOCHS=2 MLP_EPOCHS=2 FAST_DECK_SWEEP_GAMES=90 MIN_DECK_SWEEP_GAMES_PER_DECK=1 SKIP_SUBMIT=1 ./scripts/train_clean_models.sh
#
# Retrain only the MLP from an existing raw label file:
#   TRAIN_ONLY=1 SKIP_LABEL_GENERATION=1 SKIP_SEQUENCE_TRAIN=1 ./scripts/train_clean_models.sh
# TRAIN_ONLY exits before deck sweep, baseline evaluation, packaging, and Kaggle submission.
# This shell workflow requires MLP_BACKEND=torch so new MLP checkpoints are .pt-only.

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
FILTER_PASSIVE_END_LABELS="${FILTER_PASSIVE_END_LABELS:-1}"
PROGRESS="${PROGRESS:-1}"
COMPACT_PROGRESS="${COMPACT_PROGRESS:-$PROGRESS}"
MLP_PROGRESS="${MLP_PROGRESS:-$PROGRESS}"
MLP_BACKEND="${MLP_BACKEND:-torch}"
MLP_DEVICE="${MLP_DEVICE:-auto}"
MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-256}"
MLP_CACHE_EXAMPLES="${MLP_CACHE_EXAMPLES:-8192}"
MLP_CACHE_DIR="${MLP_CACHE_DIR:-}"
MLP_MODEL_PATH="${MLP_MODEL_PATH:-}"

FAST_DECK_SWEEP_GAMES="${FAST_DECK_SWEEP_GAMES:-600}"
FAST_DECK_SWEEP_AGENT_DECKS="${FAST_DECK_SWEEP_AGENT_DECKS:-top_winrate}"
FAST_DECK_SWEEP_OPPONENT_DECKS="${FAST_DECK_SWEEP_OPPONENT_DECKS:-top_winrate}"
FAST_DECK_SWEEP_OUT="${FAST_DECK_SWEEP_OUT:-artifacts/baseline/${RUN_NAME}_${RUN_ID}_fast_deck_sweep}"
BASELINE_EVAL_GAMES="${BASELINE_EVAL_GAMES:-100}"
BASELINE_EVAL_OUT="${BASELINE_EVAL_OUT:-artifacts/baseline/${RUN_NAME}_${RUN_ID}_selected_deck}"
EVAL_WORKERS="${EVAL_WORKERS:-1}"
MIN_DECK_SWEEP_GAMES_PER_DECK="${MIN_DECK_SWEEP_GAMES_PER_DECK:-15}"

SUBMISSION_DIR="${SUBMISSION_DIR:-artifacts/submission}"
KAGGLE_COMPETITION="${KAGGLE_COMPETITION:-pokemon-tcg-ai-battle}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_LABEL_GENERATION="${SKIP_LABEL_GENERATION:-0}"
SKIP_COMPACT_LABELS="${SKIP_COMPACT_LABELS:-0}"
SKIP_SEQUENCE_TRAIN="${SKIP_SEQUENCE_TRAIN:-0}"
SKIP_MLP_TRAIN="${SKIP_MLP_TRAIN:-0}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
SKIP_DECK_SWEEP="${SKIP_DECK_SWEEP:-0}"
SKIP_PACKAGE="${SKIP_PACKAGE:-0}"
SKIP_SUBMIT="${SKIP_SUBMIT:-0}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
BEST_DECK_ID="${BEST_DECK_ID:-}"
RUNTIME_POLICY="${RUNTIME_POLICY:-auto}"
RUNTIME_POLICY_REQUESTED="$RUNTIME_POLICY"
RUNTIME_LINEAR_WEIGHTS="${RUNTIME_LINEAR_WEIGHTS:-default}"
ALLOW_UNSAFE_RUNTIME_WEIGHTS="${ALLOW_UNSAFE_RUNTIME_WEIGHTS:-0}"
MIN_HEURISTIC_WIN_RATE_FOR_PACKAGE="${MIN_HEURISTIC_WIN_RATE_FOR_PACKAGE:-0.35}"
MIN_HEURISTIC_WIN_RATE_FOR_SUBMIT="${MIN_HEURISTIC_WIN_RATE_FOR_SUBMIT:-0.50}"

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -s "$path" ]]; then
    printf '%s is missing or empty: %s\n' "$description" "$path" >&2
    exit 2
  fi
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
min_games = int(sys.argv[2])
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

eligible = {deck_id: item for deck_id, item in stats.items() if item["games"] >= min_games}
if not eligible:
    counts = ", ".join(f"{deck_id}:{item['games']}" for deck_id, item in sorted(stats.items()))
    raise SystemExit(
        f"No agent deck has at least {min_games} fast-sweep games. "
        f"Increase FAST_DECK_SWEEP_GAMES or lower MIN_DECK_SWEEP_GAMES_PER_DECK. Counts: {counts}"
    )

deck_id, item = max(
    eligible.items(),
    key=lambda pair: (
        pair[1]["wins"] / max(1, pair[1]["games"]),
        pair[1]["prize_diff"] / max(1, pair[1]["games"]),
        pair[1]["games"],
        pair[0],
    ),
)
slug = re.sub(r"[^A-Za-z0-9_]+", "_", deck_id).strip("_").lower()
print(f"{deck_id}\t{slug}\t{item['\''archetype'\'']}\t{item['\''wins'\'']}/{item['\''games'\'']}\t{item['\''prize_diff'\'']/max(1,item['\''games'\'']):.3f}")
' "$FAST_DECK_SWEEP_OUT" "$MIN_DECK_SWEEP_GAMES_PER_DECK"
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

assert_package_baseline_gate() {
  if [[ "$SKIP_BASELINE" == "1" || -z "$MIN_HEURISTIC_WIN_RATE_FOR_PACKAGE" ]]; then
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
        f"MIN_HEURISTIC_WIN_RATE_FOR_PACKAGE={minimum:.3f}; refusing to package. "
        "Set MIN_HEURISTIC_WIN_RATE_FOR_PACKAGE= to disable this gate."
    )
' "$heuristic_win_rate" "$MIN_HEURISTIC_WIN_RATE_FOR_PACKAGE"
}

resolve_runtime_weights() {
  RUNTIME_WEIGHTS_PATH=""
  case "$RUNTIME_LINEAR_WEIGHTS" in
    ""|"default"|"none")
      return
      ;;
    "trained")
      RUNTIME_WEIGHTS_PATH="${SEQ_OUT}/best.json"
      ;;
    *)
      RUNTIME_WEIGHTS_PATH="$RUNTIME_LINEAR_WEIGHTS"
      ;;
  esac
}

resolve_mlp_model_path() {
  if [[ -n "${MLP_MODEL_PATH:-}" ]]; then
    require_file "$MLP_MODEL_PATH" "MLP .pt checkpoint"
    return
  fi
  MLP_MODEL_PATH="${MLP_OUT}/best.pt"
  require_file "$MLP_MODEL_PATH" "MLP .pt checkpoint"
}

assert_runtime_weight_safety() {
  if [[ -z "${RUNTIME_WEIGHTS_PATH:-}" || "$ALLOW_UNSAFE_RUNTIME_WEIGHTS" == "1" ]]; then
    return
  fi
  "$PYTHON" -c '
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    payload = json.load(fh)
weights = payload.get("weights", payload)
checks = {
    "attach_energy": 0.0,
    "attach_to_main_line": 0.0,
    "end_with_useful_action": 0.0,
    "play_pokemon_to_avoid_empty_board_loss": 0.0,
    "seq_energy_delta": 0.0,
    "seq_bench_delta": 0.0,
}
problems = []
for key, threshold in checks.items():
    value = float(weights.get(key, 0.0))
    if key == "end_with_useful_action":
        if value >= threshold:
            problems.append(f"{key}={value:.3f} should be negative")
    elif value <= threshold:
        problems.append(f"{key}={value:.3f} should be positive")
if problems:
    raise SystemExit(
        "Runtime linear weights failed safety checks: "
        + "; ".join(problems)
        + ". Use RUNTIME_LINEAR_WEIGHTS=default or ALLOW_UNSAFE_RUNTIME_WEIGHTS=1."
    )
' "$RUNTIME_WEIGHTS_PATH"
}

configure_runtime_policy() {
  local policy="$1"
  case "$policy" in
    "sequence")
      if [[ -n "${RUNTIME_WEIGHTS_PATH:-}" ]]; then
        EVAL_AGENT="sequence:${RUNTIME_WEIGHTS_PATH}"
      else
        EVAL_AGENT="sequence"
      fi
      ;;
    "mlp")
      resolve_mlp_model_path
      if [[ -n "${RUNTIME_WEIGHTS_PATH:-}" ]]; then
        EVAL_AGENT="sequence-mlp:${MLP_MODEL_PATH},${RUNTIME_WEIGHTS_PATH}"
      else
        EVAL_AGENT="sequence-mlp:${MLP_MODEL_PATH}"
      fi
      ;;
    *)
      printf 'Unknown runtime policy: %s. Use auto, sequence, or mlp.\n' "$policy" >&2
      exit 2
      ;;
  esac
}

run_selected_deck_baselines() {
  local out_dir="$1"
  run "$PYTHON" -m pokemon_agent.eval \
    --agent "$EVAL_AGENT" \
    --agent-deck-id "$BEST_DECK_ID" \
    --opponent heuristic \
    --games "$BASELINE_EVAL_GAMES" \
    --workers "$EVAL_WORKERS" \
    --out-dir "$out_dir"

  run "$PYTHON" -m pokemon_agent.eval \
    --agent "$EVAL_AGENT" \
    --agent-deck-id "$BEST_DECK_ID" \
    --opponent random \
    --opponent-decks deck_db \
    --games "$BASELINE_EVAL_GAMES" \
    --workers "$EVAL_WORKERS" \
    --out-dir "$out_dir"

  run "$PYTHON" -m pokemon_agent.eval \
    --agent "$EVAL_AGENT" \
    --agent-deck-id "$BEST_DECK_ID" \
    --opponent deck-heuristic \
    --opponent-decks top_winrate \
    --games "$BASELINE_EVAL_GAMES" \
    --workers "$EVAL_WORKERS" \
    --out-dir "$out_dir"
}

select_best_runtime_policy() {
  "$PYTHON" -c '
import csv
import glob
import os
import sys

choices = [("sequence", sys.argv[1]), ("mlp", sys.argv[2])]

def opponent_from_path(path: str) -> str:
    name = os.path.basename(path)
    if "_vs_deck-heuristic_" in name:
        return "deck-heuristic"
    if "_vs_heuristic_" in name:
        return "heuristic"
    if "_vs_random_" in name:
        return "random"
    return "unknown"

def summarize(out_dir: str) -> dict[str, float]:
    result: dict[str, float] = {
        "deck-heuristic_win_rate": -1.0,
        "heuristic_win_rate": -1.0,
        "random_win_rate": -1.0,
        "avg_prize_diff": -99.0,
    }
    prize_diffs = []
    for path in glob.glob(f"{out_dir}/eval_*.csv"):
        opponent = opponent_from_path(path)
        rows = []
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        wins = sum(1 for row in rows if row.get("result") == "win")
        result[f"{opponent}_win_rate"] = wins / len(rows)
        prize_diffs.extend(float(row.get("prize_diff") or 0.0) for row in rows)
    if prize_diffs:
        result["avg_prize_diff"] = sum(prize_diffs) / len(prize_diffs)
    return result

ranked = []
for policy, out_dir in choices:
    summary = summarize(out_dir)
    score = (
        summary["deck-heuristic_win_rate"],
        summary["heuristic_win_rate"],
        summary["random_win_rate"],
        summary["avg_prize_diff"],
        1.0 if policy == "sequence" else 0.0,
    )
    ranked.append((score, policy, out_dir, summary))

ranked.sort(reverse=True)
_score, policy, out_dir, summary = ranked[0]
deck_heuristic_wr = summary["deck-heuristic_win_rate"]
heuristic_wr = summary["heuristic_win_rate"]
random_wr = summary["random_win_rate"]
avg_prize_diff = summary["avg_prize_diff"]
record = (
    f"deck-heuristic={deck_heuristic_wr:.3f},"
    f"heuristic={heuristic_wr:.3f},"
    f"random={random_wr:.3f},"
    f"avg_prize_diff={avg_prize_diff:.3f}"
)
print(f"{policy}\t{record}\t{out_dir}")
' "$1" "$2"
}

mkdir -p \
  "$(dirname "$LABELS")" \
  "$(dirname "$COMPACT_LABELS")" \
  "$SEQ_OUT" \
  "$MLP_OUT" \
  "$FAST_DECK_SWEEP_OUT" \
  "$BASELINE_EVAL_OUT" \
  "$SUBMISSION_DIR"

if [[ "$MLP_BACKEND" != "torch" ]]; then
  printf 'scripts/train_clean_models.sh now requires MLP_BACKEND=torch for .pt-only MLP checkpoints; got %s\n' "$MLP_BACKEND" >&2
  exit 2
fi
run "$PYTHON" -c 'import torch; mps = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False; print("torch=%s mps=%s" % (torch.__version__, mps))'

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
  if [[ "$SKIP_LABEL_GENERATION" != "1" ]]; then
    run "${label_cmd[@]}"
  else
    require_file "$LABELS" "Existing replay label file"
    printf '\n==> Skipping replay label generation; using %s\n' "$LABELS"
  fi

  compact_cmd=(
    "$PYTHON" -m pokemon_agent.replay_pipeline compact-labels
    --labels "$LABELS"
    --output "$COMPACT_LABELS"
    --max-candidates "$COMPACT_MAX_CANDIDATES"
  )
  if [[ "$FILTER_PASSIVE_END_LABELS" == "1" ]]; then
    compact_cmd+=(--exclude-passive-end-labels)
  fi
  if [[ "$COMPACT_PROGRESS" == "1" ]]; then
    compact_cmd+=(--progress)
  fi
  if [[ "$SKIP_COMPACT_LABELS" != "1" ]]; then
    require_file "$LABELS" "Replay label file"
    run "${compact_cmd[@]}"
  else
    require_file "$COMPACT_LABELS" "Existing compact MLP label file"
    printf '\n==> Skipping compact label generation; using %s\n' "$COMPACT_LABELS"
  fi

  if [[ "$SKIP_SEQUENCE_TRAIN" != "1" ]]; then
    require_file "$LABELS" "Replay label file"
    run "$PYTHON" -m pokemon_agent.replay_pipeline pretrain \
      --labels "$LABELS" \
      --out-dir "$SEQ_OUT" \
      --epochs "$SEQ_EPOCHS"
  else
    printf '\n==> Skipping sequence linear training\n'
  fi

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
    --backend "$MLP_BACKEND"
    --device "$MLP_DEVICE"
    --batch-size "$MLP_BATCH_SIZE"
    --cache-examples "$MLP_CACHE_EXAMPLES"
  )
  if [[ -n "$MLP_CACHE_DIR" ]]; then
    mlp_cmd+=(--mlp-cache-dir "$MLP_CACHE_DIR")
  fi
  if [[ "$MLP_PROGRESS" == "1" ]]; then
    mlp_cmd+=(--progress)
  fi
  if [[ "$SKIP_MLP_TRAIN" != "1" ]]; then
    require_file "$COMPACT_LABELS" "Compact MLP label file"
    run "${mlp_cmd[@]}"
    MLP_MODEL_PATH="${MLP_OUT}/best.pt"
    require_file "$MLP_MODEL_PATH" "Trained MLP .pt checkpoint"
  else
    resolve_mlp_model_path
    printf '\n==> Skipping MLP training; using %s\n' "$MLP_MODEL_PATH"
  fi
fi

if [[ "$TRAIN_ONLY" == "1" ]]; then
  if [[ -z "${MLP_MODEL_PATH:-}" && -s "${MLP_OUT}/best.pt" ]]; then
    MLP_MODEL_PATH="${MLP_OUT}/best.pt"
  fi
  printf '\nTraining flow complete.\n'
  printf 'General run:        %s\n' "$RUN_NAME"
  printf 'Replay labels:      %s\n' "$LABELS"
  printf 'Compact labels:     %s\n' "$COMPACT_LABELS"
  printf 'Sequence weights:   %s\n' "${SEQ_OUT}/best.json"
  printf 'MLP ranker:         %s\n' "${MLP_MODEL_PATH:-${MLP_OUT}/best.pt}"
  printf 'MLP backend:         %s\n' "$MLP_BACKEND"
  printf 'MLP device:          %s\n' "$MLP_DEVICE"
  printf 'MLP batch size:      %s\n' "$MLP_BATCH_SIZE"
  printf 'MLP cache examples:  %s\n' "$MLP_CACHE_EXAMPLES"
  printf 'MLP cache dir:       %s\n' "${MLP_CACHE_DIR:-rebuild}"
  printf 'Filter passive end labels: %s\n' "$FILTER_PASSIVE_END_LABELS"
  printf 'Skipped label generation: %s\n' "$SKIP_LABEL_GENERATION"
  printf 'Skipped compact labels: %s\n' "$SKIP_COMPACT_LABELS"
  printf 'Skipped sequence training: %s\n' "$SKIP_SEQUENCE_TRAIN"
  printf 'Skipped MLP training: %s\n' "$SKIP_MLP_TRAIN"
  exit 0
fi

resolve_runtime_weights
assert_runtime_weight_safety
if [[ "$RUNTIME_POLICY_REQUESTED" == "auto" ]]; then
  RUNTIME_POLICY="sequence"
  configure_runtime_policy "$RUNTIME_POLICY"
else
  RUNTIME_POLICY="$RUNTIME_POLICY_REQUESTED"
  configure_runtime_policy "$RUNTIME_POLICY"
fi

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

RUNTIME_POLICY_RECORD="manual"
if [[ "$RUNTIME_POLICY_REQUESTED" == "auto" && "$SKIP_BASELINE" != "1" ]]; then
  AUTO_BASELINE_ROOT="$BASELINE_EVAL_OUT"
  SEQUENCE_BASELINE_OUT="${AUTO_BASELINE_ROOT}/runtime_sequence"
  MLP_BASELINE_OUT="${AUTO_BASELINE_ROOT}/runtime_mlp"

  RUNTIME_POLICY="sequence"
  configure_runtime_policy "$RUNTIME_POLICY"
  run_selected_deck_baselines "$SEQUENCE_BASELINE_OUT"

  RUNTIME_POLICY="mlp"
  configure_runtime_policy "$RUNTIME_POLICY"
  run_selected_deck_baselines "$MLP_BASELINE_OUT"

  RUNTIME_POLICY_ROW="$(select_best_runtime_policy "$SEQUENCE_BASELINE_OUT" "$MLP_BASELINE_OUT")"
  IFS=$'\t' read -r RUNTIME_POLICY RUNTIME_POLICY_RECORD BASELINE_EVAL_OUT <<< "$RUNTIME_POLICY_ROW"
  configure_runtime_policy "$RUNTIME_POLICY"
elif [[ "$RUNTIME_POLICY_REQUESTED" == "auto" ]]; then
  RUNTIME_POLICY="sequence"
  RUNTIME_POLICY_RECORD="auto_without_baselines:sequence"
  configure_runtime_policy "$RUNTIME_POLICY"
elif [[ "$SKIP_BASELINE" != "1" ]]; then
  run_selected_deck_baselines "$BASELINE_EVAL_OUT"
fi

MODEL_NAME="full_from_scratch_${RUNTIME_POLICY}_${BEST_DECK_SLUG}"
SUBMISSION="${SUBMISSION:-${SUBMISSION_DIR}/${MODEL_NAME}.tar.gz}"
KAGGLE_MESSAGE="${KAGGLE_MESSAGE:-$MODEL_NAME}"

assert_package_baseline_gate
assert_submit_baseline_gate

if [[ "$SKIP_PACKAGE" != "1" ]]; then
  submission_cmd=("$PYTHON" -m pokemon_agent.submission)
  if [[ -n "${RUNTIME_WEIGHTS_PATH:-}" ]]; then
    submission_cmd+=(--weights "$RUNTIME_WEIGHTS_PATH")
  fi
  if [[ "$RUNTIME_POLICY" == "mlp" ]]; then
    resolve_mlp_model_path
    submission_cmd+=(--mlp-model "$MLP_MODEL_PATH")
  fi
  submission_cmd+=(--deck-id "$BEST_DECK_ID" --output "$SUBMISSION")
  run "${submission_cmd[@]}"
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
printf 'MLP ranker:         %s\n' "${MLP_MODEL_PATH:-${MLP_OUT}/best.pt}"
printf 'MLP backend:         %s\n' "$MLP_BACKEND"
printf 'MLP device:          %s\n' "$MLP_DEVICE"
printf 'MLP batch size:      %s\n' "$MLP_BATCH_SIZE"
printf 'MLP cache examples:  %s\n' "$MLP_CACHE_EXAMPLES"
printf 'MLP cache dir:       %s\n' "${MLP_CACHE_DIR:-rebuild}"
printf 'Filter passive end labels: %s\n' "$FILTER_PASSIVE_END_LABELS"
printf 'Requested runtime policy: %s\n' "$RUNTIME_POLICY_REQUESTED"
printf 'Runtime policy:     %s\n' "$RUNTIME_POLICY"
printf 'Runtime policy record: %s\n' "$RUNTIME_POLICY_RECORD"
printf 'Runtime linear weights: %s\n' "${RUNTIME_WEIGHTS_PATH:-default}"
printf 'Fast trained-agent deck sweep: %s\n' "$FAST_DECK_SWEEP_OUT"
printf 'Fast sweep pools:   self=%s opponent=%s\n' "$FAST_DECK_SWEEP_AGENT_DECKS" "$FAST_DECK_SWEEP_OPPONENT_DECKS"
printf 'Fast sweep min games/deck: %s\n' "$MIN_DECK_SWEEP_GAMES_PER_DECK"
printf 'Eval workers:       %s\n' "$EVAL_WORKERS"
printf 'Best deck:          %s (%s, %s, avg_prize_diff=%s)\n' "$BEST_DECK_ID" "$BEST_DECK_ARCHETYPE" "$BEST_DECK_RECORD" "$BEST_DECK_PRIZE_DIFF"
printf 'Model name:         %s\n' "$MODEL_NAME"
printf 'Submission:         %s\n' "$SUBMISSION"
printf 'Kaggle message:     %s\n' "$KAGGLE_MESSAGE"
printf 'Regular baselines:  %s\n' "$BASELINE_EVAL_OUT"
printf 'Package heuristic gate: min_win_rate=%s\n' "${MIN_HEURISTIC_WIN_RATE_FOR_PACKAGE:-disabled}"
printf 'Submit heuristic gate: min_win_rate=%s\n' "${MIN_HEURISTIC_WIN_RATE_FOR_SUBMIT:-disabled}"
