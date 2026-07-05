# Pokemon Agent

Sequence-based Pokemon TCG battle agent for the Kaggle-style simulator expected at `data/pokemon-tcg-ai-battle`.

The current runtime path is:

```text
Observation -> AgentState -> Belief/GamePlan/Resource/Threat evaluators
  -> simulator-backed sequence beam search
  -> sequence featurizer/scorer/value guardrail
  -> first legal selection returned to the simulator
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

## Notes

- `train_linear.py` and `LinearQPolicy` are retained as lightweight baseline/ablation infrastructure, not as part of the default sequence-agent workflow.
- The Kaggle submission path remains dependency-light: no PyTorch runtime dependency is required.
- Deterministic reliability coverage includes deck-reveal prize inference and submission import checks.
