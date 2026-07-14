# Pokemon Agent

Sequence-based Pokemon TCG battle agent for the Kaggle-style simulator expected at `data/pokemon-tcg-ai-battle`.

The runtime is organized as an explicit agent loop:

```text
observe:
  raw observation + legal selections + logs
  -> AgentState + OpponentMemory + BeliefState + ProbabilitySnapshot

plan:
  observed state and belief context
  -> simulator-backed SearchTree + CandidateSequence options

evaluate:
  candidate sequences and resulting simulated states
  -> linear sequence score + optional MLP or root-ranker score + value guardrail features

act:
  scored candidates
  -> selected candidate.first_selection returned to the simulator
```

After acting, any uncommitted suffix of the chosen sequence is cached as a warm-start candidate for the next callback in the same turn. It is reused only if the next cached step still matches the current legal option identity, and it is still evaluated against freshly generated alternatives.

## Current Defaults

- Default deck/config: `configs/starmie_froslass.json`
- Runtime policy: `sequence`
- Submission bundle target: `artifacts/submission/sequence_starmie_froslass_submission.tar.gz`
- Opponent/deck priors: `data/deck_db/decks.json`, with derived archetype gameplan profiles.

## Useful Commands

The Kaggle simulator/data bundle is intentionally ignored by Git because it contains large local files. Unpack it locally under `data/pokemon-tcg-ai-battle` before running tests, evaluations, or submission builds.

Run smoke tests:

```bash
./.venv/bin/python -m pokemon_agent.tests_smoke
```

Evaluate the current sequence agent:

```bash
./.venv/bin/python -m pokemon_agent.eval --agent sequence --opponent random --games 24
./.venv/bin/python -m pokemon_agent.eval --agent sequence --opponent random --games 24 --opponent-decks deck_db
./.venv/bin/python -m pokemon_agent.eval --agent sequence --opponent heuristic --games 24
```

Build the current submission:

```bash
./.venv/bin/python -m pokemon_agent.submission --output artifacts/submission/sequence_starmie_froslass_submission.tar.gz
```

Run the batch validation/package flow:

```bash
./scripts/train_clean_models.sh
```

## Immediate-Root XGBoost Ranker

The optional root ranker learns which immediate action root an expert selected. Replay preparation canonicalizes each `first_selection`, retains one base-scored sequence per root, and caps the selectable group at the first 24 roots. At inference, XGBoost reranks that same selectable set. Deduplicated roots outside the cap remain guardrail-only so the linear fallback can still override an unsafe learned choice; they are never made selectable by the model.

The offline experiment uses native XGBoost for training, but the shipped model is the compact JSON artifact at `pokemon_agent/models/xgb_root_v1.json`. Runtime loading and inference for that artifact do not require the `xgboost` package. Root ranking requires `num_hidden_samples=1` and is mutually exclusive with MLP ranking.

Generate deterministic replay splits, root labels, and a native training model with:

```bash
./.venv/bin/python -m pokemon_agent.xgb_root_ranker manifest \
  --high-elo-manifest <high_elo_manifest.csv> \
  --output artifacts/xgb_root_v1/replay_manifest.json

./.venv/bin/python -m pokemon_agent.xgb_root_ranker prepare-labels \
  --manifest artifacts/xgb_root_v1/replay_manifest.json \
  --output-dir artifacts/xgb_root_v1/labels \
  --workers 32

./.venv/bin/python -m pokemon_agent.xgb_root_ranker train \
  --train-labels artifacts/xgb_root_v1/labels/train.jsonl \
  --validation-labels artifacts/xgb_root_v1/labels/validation.jsonl \
  --test-labels artifacts/xgb_root_v1/labels/test.jsonl \
  --output-dir artifacts/xgb_root_v1/model_fixed_v1
```

Use and package the compact runtime model with:

```bash
./.venv/bin/python -m pokemon_agent.eval \
  --agent 'sequence-xgb:pokemon_agent/models/xgb_root_v1.json' \
  --opponent sequence --games 150

./.venv/bin/python -m pokemon_agent.submission \
  --xgb-model pokemon_agent/models/xgb_root_v1.json \
  --output artifacts/submission/xgb_root_v1_submission.tar.gz
```

The frozen 30-deck, five-rotation full matrix completed 4,500 games per agent with zero invalid actions. The compact-equivalent root policy scored 61.07% versus 50.29% for `sequence`, a paired improvement of 10.78 percentage points (paired-game 95% CI 8.89 to 12.66; deck-cluster 95% CI 6.31 to 15.24). It improved 22 of 30 agent decks.

## Full-Schema MLP Ranking Training

The default training path keeps the simulator-backed sequence tree, uses linear weights for search expansion/pruning, and trains the MLP to own final candidate ranking. Replay labels use each player's submitted replay deck by default, deriving deck-specific roles and plan context from `data/deck_db/decks.json` when the list matches and from a synthetic replay config when it does not. The MLP receives full-schema candidate context, including action, board, consequence, self-deck archetype/plan, opponent, probability, risk/counterplay, card-knowledge, and field-attention features. Scorer-generated `linear_score` and `mlp_score` diagnostics are excluded from MLP training features.

It trains:

- full-schema linear sequence weights for diagnostics and ablations
- full-schema two-hidden-layer MLP sequence ranker selected by validation top-1
- runtime policy selection that packages the locally stronger selected-deck runtime

Run the full training pipeline:

```bash
./scripts/train_clean_models.sh
```

For a pilot run:

```bash
MAX_EXAMPLES=1000 SEQ_EPOCHS=2 MLP_EPOCHS=2 FAST_DECK_SWEEP_GAMES=90 MIN_DECK_SWEEP_GAMES_PER_DECK=1 SKIP_SUBMIT=1 ./scripts/train_clean_models.sh
```

Include self-play fine-tuning after replay pretraining:

```bash
RUN_SELF_PLAY=1 SELF_PLAY_GAMES=5000 ./scripts/train_clean_models.sh
```

Evaluate checkpoints against varied decks, including replay-observed top win-rate deck rotations. The production default is `sequence`; MLP checkpoints should beat it locally before packaging:

```bash
RUN_BASELINE_EVAL=1 BASELINE_EVAL_GAMES=100 ./scripts/train_clean_models.sh

./.venv/bin/python -m pokemon_agent.eval \
  --agent sequence \
  --opponent heuristic \
  --games 100

./.venv/bin/python -m pokemon_agent.eval \
  --agent sequence \
  --opponent deck-heuristic \
  --opponent-decks top_winrate \
  --games 100
```

Package through the training script so runtime policy selection and baseline gates are applied:

```bash
SKIP_TRAIN=1 BEST_DECK_ID=<deck_id> SKIP_SUBMIT=1 ./scripts/train_clean_models.sh
```

For clean-schema ablations, run `TRAIN_SCHEMA=clean ./scripts/train_clean_models.sh`.

## Replay Imitation And Self-Play

Fetch top-team replay JSON with the Kaggle CLI after joining the competition and configuring Kaggle credentials:

```bash
./.venv/bin/python -m pokemon_agent.replay_pipeline fetch \
  --competition <competition-slug> \
  --out-dir artifacts/replays/kaggle \
  --top-teams 20 \
  --episodes-per-submission 50
```

Convert replay decisions into sequence labels. This extracts the submitted 60-card decks from the replay, scores each player's decisions with that player's deck-specific config, matches each high-Elo selected action against simulator-generated candidate sequences, and stores the candidate feature set plus the expert label:

```bash
./.venv/bin/python -m pokemon_agent.replay_pipeline labels \
  artifacts/replays/kaggle \
  --output artifacts/replay_labels/high_elo_sequence_labels.jsonl \
  --min-rating 1400 \
  --beam-width 32 \
  --max-depth 12 \
  --max-nodes 512
```

Use `--single-config` only for old single-deck ablations that force every replay through the configured default deck.

Pretrain the sequence scorer with imitation learning:

```bash
./.venv/bin/python -m pokemon_agent.replay_pipeline pretrain \
  --labels artifacts/replay_labels/high_elo_sequence_labels.jsonl \
  --out-dir artifacts/checkpoints/replay_pretrain \
  --epochs 5
```

Pretrain the PyTorch MLP sequence ranker on the same candidate-label JSONL:

```bash
./.venv/bin/python -m pokemon_agent.replay_pipeline pretrain-mlp \
  --labels artifacts/replay_labels/high_elo_sequence_labels.jsonl \
  --out-dir artifacts/checkpoints/mlp_ranker \
  --epochs 5 \
  --hidden-dim 0 \
  --hidden-dim2 0 \
  --max-features 512 \
  --validation-fraction 0.10 \
  --feature-schema-version full \
  --backend torch \
  --device auto \
  --batch-size 256
```

The current MLP checkpoint format is `best.pt`. It stores the feature vocabulary, input normalization stats, hidden-layer weights, and validation metrics. Legacy JSON checkpoints can still be loaded for debugging, but the main training and packaging flow now expects `.pt`.

The runtime is intentionally strict. New MLP checkpoints must declare an explicit feature schema such as `full` or `clean`; legacy/unversioned checkpoints are unnecessary for the current workflow and should be retrained from scratch after architecture changes.

To retrain only the MLP from an existing raw replay-label file, rebuild the compact MLP labels so passive expert END turns are filtered before pretraining:

```bash
TRAIN_ONLY=1 SKIP_LABEL_GENERATION=1 SKIP_SEQUENCE_TRAIN=1 ./scripts/train_clean_models.sh
```

`TRAIN_ONLY=1` exits after the training phases, before deck sweep, baseline evaluation, packaging, or Kaggle submission. `SKIP_SUBMIT=1` is not required for this retrain-only command.

Use `SKIP_COMPACT_LABELS=1` only when the compact label file was already produced by this filtered path. Old compact labels should be treated as stale after MLP architecture or label-filter changes.

Fine-tune the MLP ranker with self-play returns:

```bash
./.venv/bin/python -m pokemon_agent.replay_pipeline self-play-mlp \
  --model artifacts/checkpoints/mlp_ranker/best.pt \
  --out-dir artifacts/checkpoints/mlp_self_play \
  --games 5000 \
  --checkpoint-every 250
```

Evaluate an MLP checkpoint. Treat this as a candidate comparison against `sequence`, not as an automatic submission path:

```bash
./.venv/bin/python -m pokemon_agent.eval --agent sequence-mlp:artifacts/checkpoints/mlp_self_play/best.pt --opponent heuristic --games 100
./.venv/bin/python -m pokemon_agent.eval --agent sequence-mlp:artifacts/checkpoints/mlp_self_play/best.pt --opponent random --games 100 --opponent-decks deck_db
```

Evaluate an MLP checkpoint on top of learned sequence weights. This is unnecessary for the current main path and should be used only as an ablation:

```bash
./.venv/bin/python -m pokemon_agent.eval \
  --agent sequence-mlp:artifacts/checkpoints/mlp_depth_sweep_20260708T083117Z/four_192_128_64_32/best.pt,artifacts/versions/sequence_starmie_froslass/weights.json \
  --opponent heuristic \
  --games 100
```

Build an MLP submission only after it wins local baselines:

```bash
RUNTIME_POLICY=mlp SKIP_TRAIN=1 BEST_DECK_ID=<deck_id> SKIP_SUBMIT=1 ./scripts/train_clean_models.sh
```

Package a combined sequence-weight plus MLP ablation only after it beats the strict MLP-only path and the runtime linear weights pass safety checks. Do not use this as the default submission path:

```bash
./.venv/bin/python -m pokemon_agent.submission \
  --weights artifacts/versions/sequence_starmie_froslass/weights.json \
  --mlp-model artifacts/checkpoints/mlp_depth_sweep_20260708T083117Z/four_192_128_64_32/best.pt \
  --output artifacts/submission/mlp_sequence_submission.tar.gz
```

Fine-tune linear sequence weights with self-play returns. This is unnecessary for the current main path; keep it as an ablation/debug tool:

```bash
./.venv/bin/python -m pokemon_agent.replay_pipeline self-play \
  --weights artifacts/checkpoints/replay_pretrain/best.json \
  --out-dir artifacts/checkpoints/sequence_self_play \
  --games 5000 \
  --checkpoint-every 250
```

Evaluate a trained checkpoint:

```bash
./.venv/bin/python -m pokemon_agent.eval --agent sequence:artifacts/checkpoints/sequence_self_play/best.json --opponent heuristic --games 100
./.venv/bin/python -m pokemon_agent.eval --agent sequence:artifacts/checkpoints/sequence_self_play/best.json --opponent random --games 100 --opponent-decks deck_db
```

## Notes

- `LinearQPolicy` is retained as a lightweight fallback/baseline policy, not as a standalone training path.
- Legacy MLP schema compatibility and silent runtime fallback are unnecessary. Runtime/model/schema failures should fail fast and be fixed by regenerating labels and retraining.
- The Kaggle submission path now packages `mlp_model.pt` for MLP runs and expects PyTorch in the runtime environment.
- Deterministic reliability coverage includes deck-reveal prize inference and submission import checks.
