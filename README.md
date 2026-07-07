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
  -> linear sequence score + MLP ranker score + value guardrail features

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

## Clean Architecture Training

The clean path keeps the simulator-backed sequence tree but removes direct probability/value/resource/threat/card-attention features from final candidate scoring. It trains:

- clean linear sequence weights
- clean MLP sequence ranker

Run the full clean training pipeline:

```bash
./scripts/train_clean_models.sh
```

For a pilot run:

```bash
MAX_EXAMPLES=1000 SEQ_EPOCHS=2 MLP_EPOCHS=2 ./scripts/train_clean_models.sh
```

Include clean self-play fine-tuning after replay pretraining:

```bash
RUN_SELF_PLAY=1 SELF_PLAY_GAMES=5000 ./scripts/train_clean_models.sh
```

Evaluate clean checkpoints:

```bash
./.venv/bin/python -m pokemon_agent.eval \
  --agent sequence-clean-mlp:artifacts/checkpoints/clean_mlp_ranker/best.json,artifacts/checkpoints/clean_sequence_linear/best.json \
  --opponent heuristic \
  --games 100
```

Package a clean submission:

```bash
./.venv/bin/python -m pokemon_agent.submission \
  --clean \
  --weights artifacts/checkpoints/clean_sequence_linear/best.json \
  --mlp-model artifacts/checkpoints/clean_mlp_ranker/best.json \
  --output artifacts/submission/clean_sequence_submission.tar.gz
```

## Replay Imitation And Self-Play

Fetch top-team replay JSON with the Kaggle CLI after joining the competition and configuring Kaggle credentials:

```bash
./.venv/bin/python -m pokemon_agent.replay_pipeline fetch \
  --competition <competition-slug> \
  --out-dir artifacts/replays/kaggle \
  --top-teams 20 \
  --episodes-per-submission 50
```

Convert replay decisions into sequence labels. This matches each high-Elo selected action against simulator-generated candidate sequences and stores the candidate feature set plus the expert label:

```bash
./.venv/bin/python -m pokemon_agent.replay_pipeline labels \
  artifacts/replays/kaggle \
  --output artifacts/replay_labels/high_elo_sequence_labels.jsonl \
  --min-rating 1400 \
  --beam-width 32 \
  --max-depth 12 \
  --max-nodes 512
```

Pretrain the sequence scorer with imitation learning:

```bash
./.venv/bin/python -m pokemon_agent.replay_pipeline pretrain \
  --labels artifacts/replay_labels/high_elo_sequence_labels.jsonl \
  --out-dir artifacts/checkpoints/replay_pretrain \
  --epochs 5
```

Pretrain the dependency-light MLP sequence ranker on the same candidate-label JSONL:

```bash
./.venv/bin/python -m pokemon_agent.replay_pipeline pretrain-mlp \
  --labels artifacts/replay_labels/high_elo_sequence_labels.jsonl \
  --out-dir artifacts/checkpoints/mlp_ranker \
  --epochs 5 \
  --hidden-dim 128 \
  --max-features 256
```

The MLP checkpoint is a JSON file with its feature vocabulary, input normalization stats, and weights. It is intentionally framework-free so it can be loaded later without adding PyTorch or TensorFlow to the submission bundle.

The runtime is intentionally strict. New MLP checkpoints must declare an explicit feature schema such as `sequence_v3_full` or `sequence_v3_clean`; legacy/unversioned checkpoints are unnecessary for the current workflow and should be retrained from scratch after architecture changes.

Fine-tune the MLP ranker with self-play returns:

```bash
./.venv/bin/python -m pokemon_agent.replay_pipeline self-play-mlp \
  --model artifacts/checkpoints/mlp_ranker/best.json \
  --out-dir artifacts/checkpoints/mlp_self_play \
  --games 5000 \
  --checkpoint-every 250
```

Evaluate an MLP checkpoint:

```bash
./.venv/bin/python -m pokemon_agent.eval --agent sequence-mlp:artifacts/checkpoints/mlp_self_play/best.json --opponent heuristic --games 100
./.venv/bin/python -m pokemon_agent.eval --agent sequence-mlp:artifacts/checkpoints/mlp_self_play/best.json --opponent random --games 100 --opponent-decks deck_db
```

Evaluate an MLP checkpoint on top of learned sequence weights. This is unnecessary for the current main path and should be used only as an ablation:

```bash
./.venv/bin/python -m pokemon_agent.eval \
  --agent sequence-mlp:artifacts/checkpoints/mlp_ranker_high_elo/best.json,artifacts/checkpoints/replay_pretrain_top_sample/best.json \
  --opponent heuristic \
  --games 100
```

Build an MLP submission:

```bash
./.venv/bin/python -m pokemon_agent.submission \
  --mlp-model artifacts/checkpoints/mlp_self_play/best.json \
  --output artifacts/submission/mlp_sequence_submission.tar.gz
```

Build a combined sequence-weight plus MLP submission. This is unnecessary unless the combined ablation beats the strict MLP-only path:

```bash
./.venv/bin/python -m pokemon_agent.submission \
  --weights artifacts/checkpoints/replay_pretrain_top_sample/best.json \
  --mlp-model artifacts/checkpoints/mlp_ranker_high_elo/best.json \
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
- The Kaggle submission path remains dependency-light and uses stdlib JSON MLP inference.
- Deterministic reliability coverage includes deck-reveal prize inference and submission import checks.
