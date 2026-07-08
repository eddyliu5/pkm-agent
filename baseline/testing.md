# Testing Against Baselines

This runbook is for agents evaluating a code or model change.

## Prerequisites

- The Kaggle simulator/data bundle must be unpacked at `data/pokemon-tcg-ai-battle`.
- Use the project virtual environment commands shown below.
- Keep generated outputs under `artifacts/baseline/<run-name>/`.
- Do not compare runs with different configs unless the config change is the experiment.

## Quick Iteration

Use this when checking whether a change is obviously broken:

```bash
RUN=artifacts/baseline/quick
AGENT=sequence

./.venv/bin/python -m pokemon_agent.tests_smoke
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent random --games 24 --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent random --games 24 --opponent-decks deck_db --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent heuristic --games 24 --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent deck-heuristic --games 24 --opponent-decks top_winrate --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent sequence-starmie-froslass --games 24 --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent sequence --games 24 --out-dir "$RUN"
```

Do not use quick iteration results as proof of improvement. They are only a filter.

## Standard Baseline Run

Use this before claiming a local improvement:

```bash
BASE=artifacts/baseline/default-sequence
RUN=artifacts/baseline/<candidate-name>
AGENT='<candidate-agent-spec>'

./.venv/bin/python -m pokemon_agent.tests_smoke

./.venv/bin/python -m pokemon_agent.eval --agent sequence --opponent random --games 100 --out-dir "$BASE"
./.venv/bin/python -m pokemon_agent.eval --agent sequence --opponent random --games 100 --opponent-decks deck_db --out-dir "$BASE"
./.venv/bin/python -m pokemon_agent.eval --agent sequence --opponent heuristic --games 100 --out-dir "$BASE"
./.venv/bin/python -m pokemon_agent.eval --agent sequence --opponent deck-heuristic --games 100 --opponent-decks top_winrate --out-dir "$BASE"
./.venv/bin/python -m pokemon_agent.eval --agent sequence --opponent sequence-starmie-froslass --games 100 --out-dir "$BASE"
./.venv/bin/python -m pokemon_agent.eval --agent sequence --opponent sequence --games 100 --out-dir "$BASE"
./.venv/bin/python -m pokemon_agent.eval --agent sequence --opponent sequence --games 100 --opponent-decks deck_db --out-dir "$BASE"

./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent random --games 100 --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent random --games 100 --opponent-decks deck_db --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent heuristic --games 100 --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent deck-heuristic --games 100 --opponent-decks top_winrate --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent sequence-starmie-froslass --games 100 --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent sequence --games 100 --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent sequence --games 100 --opponent-decks deck_db --out-dir "$RUN"
```

If the baseline was already run from the same code state, reuse it and cite the artifact directory.

For sequence-policy changes, run the standardized top-deck matrix as the main strength check:

```bash
./.venv/bin/python scripts/run_top_deck_sequence_compare.py
```

Default behavior is 5 complete top-winrate opponent rotations per self deck. With 30 selected top decks, that is 150 games per self deck and 4500 total `sequence` games. The runner uses a fixed per-round seed schedule, starting with seed `42`, and records the seeds in `manifest.json`.

## Final Candidate Run

Use this before packaging a Kaggle candidate when time allows:

```bash
RUN=artifacts/baseline/final-<candidate-name>
AGENT='<candidate-agent-spec>'

./.venv/bin/python -m pokemon_agent.tests_smoke
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent random --games 300 --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent random --games 300 --opponent-decks deck_db --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent heuristic --games 300 --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent deck-heuristic --games 300 --opponent-decks top_winrate --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent sequence-starmie-froslass --games 300 --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent sequence --games 300 --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent sequence --games 300 --opponent-decks deck_db --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.submission --output "$RUN/submission.tar.gz"
```

For clean models, package with the same model files used in evaluation:

```bash
./.venv/bin/python -m pokemon_agent.submission \
  --clean \
  --weights artifacts/checkpoints/clean_sequence_linear/best.json \
  --mlp-model artifacts/checkpoints/clean_mlp_ranker/best.json \
  --output "$RUN/submission.tar.gz"
```

## What To Record

Every baseline report should include:

- Commit or working-tree description.
- Agent spec.
- Config path, if non-default.
- Game count per suite.
- `win_rate`, `invalid_actions`, `average_steps`, and `average_prize_diff` from each evaluator summary.
- Local Kaggle proxy score from `kaggle_proxy.md`.
- Paths to the JSONL and CSV artifacts.
- Any crashes, timeouts, or simulator setup issues.
- For top-deck matrix runs: rounds, seed schedule, total games, pooled win rate, mean deck win rate, invalid actions, and the artifact path under `baseline/compare/<run-id>/`.

Use this short format in notes or pull requests:

```text
Agent: sequence-mlp:artifacts/checkpoints/example/best.json
Config: default configs/starmie_froslass.json
Games: 100 per suite

random mirror: win_rate=..., invalid_actions=..., avg_prize_diff=...
deck-db rotation: win_rate=..., invalid_actions=..., avg_prize_diff=...
heuristic mirror: win_rate=..., invalid_actions=..., avg_prize_diff=...
deck-heuristic top winrate: win_rate=..., invalid_actions=..., avg_prize_diff=...
sequence-starmie-froslass: win_rate=..., invalid_actions=..., avg_prize_diff=...
sequence mirror: win_rate=..., invalid_actions=..., avg_prize_diff=...
sequence deck-db rotation: win_rate=..., invalid_actions=..., avg_prize_diff=...
local_kaggle_proxy=...

Artifacts: artifacts/baseline/<candidate-name>/
Decision: promote / repeat with 300 games / reject
```

## Interpreting Results

- Any invalid action is a blocker until explained and fixed.
- Deck-db rotation is the best local robustness signal because Kaggle opponents are not mirror-only.
- Sequence opponents are the main strength check. A 100% result against random or heuristic is not enough evidence that the agent improved.
- Average prize differential helps distinguish close wins from dominant wins, but it should not override win rate.
- Average steps is a runtime and game-shape signal. Large step increases may indicate slower decision loops or failure to close games.
- A candidate that wins more against random but loses ground against heuristic or deck-db rotation is probably overfit to weak opposition.
