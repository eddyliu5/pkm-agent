# Baseline System

This directory defines the local baseline system for evaluating Pokemon-Agent changes before submitting to Kaggle.

The goal is to make agent changes comparable across runs. A useful baseline run answers four questions:

1. Does the agent still satisfy basic runtime and packaging contracts?
2. Does it beat weak local opponents reliably?
3. Does it stay robust when opponent decks are rotated through the local deck database?
4. Does its local score move in a direction that is likely to improve Kaggle performance?

Use these files together:

- `baselines.md`: the baseline agents, opponent pools, and gates to compare against.
- `testing.md`: the command sequence and reporting format for running a baseline evaluation.
- `kaggle_proxy.md`: the local proxy score definition and how to interpret it.

## Default Baseline Suite

Run this suite before claiming an agent improvement:

```bash
./.venv/bin/python -m pokemon_agent.tests_smoke
./.venv/bin/python -m pokemon_agent.eval --agent sequence --opponent random --games 100 --out-dir artifacts/baseline/current
./.venv/bin/python -m pokemon_agent.eval --agent sequence --opponent random --games 100 --opponent-decks deck_db --out-dir artifacts/baseline/current
./.venv/bin/python -m pokemon_agent.eval --agent sequence --opponent heuristic --games 100 --out-dir artifacts/baseline/current
./.venv/bin/python -m pokemon_agent.submission --output artifacts/baseline/current/submission.tar.gz
```

For quick iteration, use 24 games per evaluation. For decision-making, use at least 100 games per matchup. For final candidate selection, prefer 300+ games per matchup when time allows.

## Baseline Principles

- Always compare against the same config and deck unless the experiment is explicitly about deck selection.
- Always report invalid actions. A higher win rate with invalid actions is not a clean improvement.
- Prefer deck-db rotation over mirror-only results when estimating Kaggle movement.
- Treat one short run as a signal, not proof. Repeat promising changes with more games.
- Preserve raw JSONL and CSV outputs under `artifacts/baseline/<run-name>/`.

