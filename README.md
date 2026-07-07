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
./.venv/bin/python -m pokemon_agent.overnight --baseline-games 100
```

## Clean Architecture Training

The clean path keeps the simulator-backed sequence tree but removes direct probability/value/resource/threat/card-attention features from final candidate scoring. It trains:

- clean linear sequence weights
- clean MLP sequence ranker
- reply-risk/hidden-capability model

Run the full clean training pipeline:

```bash
./scripts/train_clean_models.sh
```

For a pilot run:

```bash
MAX_EXAMPLES=1000 SEQ_EPOCHS=2 MLP_EPOCHS=2 REPLY_EPOCHS=2 ./scripts/train_clean_models.sh
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

For larger notebook/GPU training runs, train with PyTorch and export the same framework-free JSON checkpoint:

```bash
python -m pokemon_agent.torch_mlp_trainer \
  --labels artifacts/replay_labels/high_elo_sequence_labels.top16.compact.jsonl \
  --out-dir artifacts/checkpoints/torch_mlp_ranker \
  --epochs 20 \
  --hidden-dim 128 \
  --max-features 256 \
  --batch-size 512 \
  --device auto
```

`torch_mlp_trainer` requires PyTorch only in the training environment. The exported `best.json` still loads through the standard `MLPSequenceRanker`, so the Kaggle submission remains dependency-light.

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

Evaluate an MLP checkpoint on top of learned sequence weights:

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

Build a combined sequence-weight plus MLP submission:

```bash
./.venv/bin/python -m pokemon_agent.submission \
  --weights artifacts/checkpoints/replay_pretrain_top_sample/best.json \
  --mlp-model artifacts/checkpoints/mlp_ranker_high_elo/best.json \
  --output artifacts/submission/mlp_sequence_submission.tar.gz
```

## Probability Feature Training Refresh

The runtime now emits hidden-information probability features such as prize risk, energy access, and line accessibility. Existing MLP checkpoints do not use new feature names until labels are regenerated and the ranker is retrained.

Regenerate replay labels with probability features:

```bash
./.venv/bin/python -m pokemon_agent.replay_pipeline labels \
  artifacts/replays/kaggle \
  --output artifacts/replay_labels/high_elo_sequence_labels_probability.jsonl \
  --min-rating 1400 \
  --max-candidates 24 \
  --beam-width 32 \
  --max-depth 12 \
  --max-nodes 512 \
  --time-budget-ms 1000
```

Compact labels for MLP training:

```bash
./.venv/bin/python -m pokemon_agent.replay_pipeline compact-labels \
  --labels artifacts/replay_labels/high_elo_sequence_labels_probability.jsonl \
  --output artifacts/replay_labels/high_elo_sequence_labels_probability.top16.compact.jsonl \
  --max-candidates 16
```

Train refreshed sequence weights and MLP checkpoints:

```bash
./.venv/bin/python -m pokemon_agent.replay_pipeline pretrain \
  --labels artifacts/replay_labels/high_elo_sequence_labels_probability.jsonl \
  --out-dir artifacts/checkpoints/replay_pretrain_probability \
  --epochs 5

./.venv/bin/python -m pokemon_agent.replay_pipeline pretrain-mlp \
  --labels artifacts/replay_labels/high_elo_sequence_labels_probability.top16.compact.jsonl \
  --out-dir artifacts/checkpoints/mlp_ranker_probability \
  --epochs 5 \
  --hidden-dim 128 \
  --max-features 320
```

Evaluate before packaging:

```bash
./.venv/bin/python -m pokemon_agent.eval \
  --agent sequence-mlp:artifacts/checkpoints/mlp_ranker_probability/best.json,artifacts/checkpoints/replay_pretrain_probability/best.json \
  --opponent heuristic \
  --games 100

./.venv/bin/python -m pokemon_agent.eval \
  --agent sequence-mlp:artifacts/checkpoints/mlp_ranker_probability/best.json,artifacts/checkpoints/replay_pretrain_probability/best.json \
  --opponent random \
  --games 100 \
  --opponent-decks deck_db
```

Build a refreshed submission:

```bash
./.venv/bin/python -m pokemon_agent.submission \
  --weights artifacts/checkpoints/replay_pretrain_probability/best.json \
  --mlp-model artifacts/checkpoints/mlp_ranker_probability/best.json \
  --output artifacts/submission/mlp_probability_sequence_submission.tar.gz
```

Fine-tune with sequence-feature self-play returns:

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

- `train_linear.py` and `LinearQPolicy` are retained as lightweight baseline/ablation infrastructure, not as part of the default sequence-agent workflow.
- The Kaggle submission path remains dependency-light: no PyTorch runtime dependency is required.
- Deterministic reliability coverage includes deck-reveal prize inference and submission import checks.
