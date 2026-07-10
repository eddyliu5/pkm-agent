# Baselines

This directory holds the durable evaluation contract. Generated runs belong in
`artifacts/baseline/compare/`, not here.

## Ground truth

[`ground_truth/20260708T021405Z`](ground_truth/20260708T021405Z/) is the
historical general-agent reference:

- `sequence` versus round-robin `sequence` over 30 replay-observed top decks.
- Each self deck played 100 games: 3,000 total games.
- 1,571 wins, 1,429 losses, 0 invalid actions: **52.3667% win rate**.

This result demonstrates cross-deck robustness. It is a pinned historical
reference, not a seed-identical rerun target: its run did not record a seed
schedule.

## Evaluation levels

Run the smoke test before any evaluator run:

```bash
./.venv/bin/python -m pokemon_agent.tests_smoke
```

Use the fast top-deck screen while iterating. It evaluates each agent over the
same top-8 deck matrix for three full rotations (192 games per agent), with
policy seeds `42, 1729, 8675309`:

```bash
./.venv/bin/python scripts/run_top_deck_sequence_compare.py \
  --preset fast \
  --agent sequence \
  --agent '<candidate-agent-spec>'
```

`comparison.json` reports each candidate's pooled win-rate delta against
`sequence`. Fast results reject clear regressions only; they do not establish
an improvement.

Use the full matrix to promote a sequence-policy change. It is 30 top decks by
five rotations: 4,500 games per agent, using the first five policy seeds.

```bash
./.venv/bin/python scripts/run_top_deck_sequence_compare.py \
  --preset full \
  --agent sequence \
  --agent '<candidate-agent-spec>'
```

MLP ranking is deliberately disabled for the sequence-only baseline. Add
`--enable-mlp` when an MLP candidate is intentionally being evaluated.

The bundled simulator does not expose a game-RNG seed API, so the seed schedule
controls policy randomness and deck rotation only. Re-run close results before
making a decision; do not describe this protocol as fully deterministic.

Every promoted run must report its commit, agent spec, preset, seed schedule,
pooled and mean-deck win rate, and invalid-action count. A nonzero invalid
action count fails the run.
