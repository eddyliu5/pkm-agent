# Baselines

This file lists the baseline agents and opponent suites that should be used when testing changes.

## Candidate Agents

These are valid `--agent` specs for the current evaluator.

| Agent spec | Purpose | When to use |
| --- | --- | --- |
| `random` | Sanity floor. | Confirms the simulator loop can run and exposes invalid-action problems in wrappers. |
| `heuristic` | Simple feature-based policy. | Checks whether a change beats static action scoring without sequence search. |
| `deck-heuristic` | Static heuristic with roles derived from the exact deck it pilots. | Stronger non-search opponent for top-deck matchup checks. |
| `deck-heuristic:<deck_id>` | Static heuristic with roles derived from a specific `data/deck_db/decks.json` list. | Reproduce a known archetype or replay-derived deck baseline. |
| `sequence` | Current default production baseline. | Main baseline for regressions and submissions. |
| `sequence-starmie-froslass` | Saved Starmie/Froslass sequence submission weights from `artifacts/submission/sequence_starmie_froslass_submission.tar.gz`. | Regression gate against the current best Kaggle-scoring local artifact. |
| `sequence:<weights.json>` | Sequence search with learned linear weights. | Unnecessary for the current improvement path; keep only as an ablation/debug baseline. |
| `sequence-mlp:<mlp.json>` | Sequence search with an MLP ranker. | Compare MLP checkpoints against default sequence and learned-linear variants. |
| `sequence-mlp:<mlp.json>,<weights.json>` | MLP ranker plus learned linear weights. | Unnecessary unless explicitly testing a combined ablation; prefer schema-strict MLP-only comparisons first. |
| `sequence-clean` | Clean evaluator path without direct probability/value/resource/threat/card-attention final scoring. | Architecture experiments and clean scoring regressions. |
| `sequence-clean-mlp:<mlp.json>,<weights.json>` | Clean evaluator plus trained MLP and weights. | Candidate clean-architecture submission testing. |

## Opponent Baselines

Run every serious candidate against these local opponents.

| Suite | Command shape | Measures |
| --- | --- | --- |
| Random mirror | `--opponent random` | Basic tactical competence, action validity, and regression against a weak opponent. |
| Random deck-db rotation | `--opponent random --opponent-decks deck_db` | Robustness to varied opposing decks and archetypes. This is the most important local proxy component. |
| Heuristic mirror | `--opponent heuristic` | Whether the agent beats a deterministic feature baseline using the same default deck. |
| Deck-heuristic top winrate rotation | `--opponent deck-heuristic --opponent-decks top_winrate` | Tests against replay-observed high-winrate decks with deck-specific static role priorities. |
| Starmie/Froslass Kaggle-best mirror | `--opponent sequence-starmie-froslass` | Tests against the saved best-scoring Starmie/Froslass sequence artifact. |
| Sequence mirror | `--opponent sequence` | Strong head-to-head check against the current production search policy. A candidate should not collapse below parity. |
| Sequence deck-db rotation | `--opponent sequence --opponent-decks deck_db` | Strong varied-deck check against search-based opponents. Use this before treating a local score as meaningful. |

## Baseline Gates

Use these gates when deciding whether a change is ready for more expensive testing or Kaggle submission.

| Gate | Required result |
| --- | --- |
| Smoke | `./.venv/bin/python -m pokemon_agent.tests_smoke` exits successfully. |
| Invalid actions | `invalid_actions == 0` in every baseline summary. |
| Runtime | No evaluation crashes, no submission packaging failure. |
| Random mirror | Candidate win rate should not be below current `sequence` by more than 2 percentage points on the same game count. |
| Deck-db rotation | Candidate should improve or match the current `sequence` proxy contribution. This gate matters more than random mirror. |
| Heuristic mirror | Candidate should improve or match current `sequence`, unless the experiment intentionally trades mirror performance for deck-db robustness. |

## Recommended Comparison Matrix

For a new candidate checkpoint, run:

```bash
RUN=artifacts/baseline/<candidate-name>
AGENT='<agent-spec>'

./.venv/bin/python -m pokemon_agent.tests_smoke
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent random --games 100 --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent random --games 100 --opponent-decks deck_db --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent heuristic --games 100 --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent deck-heuristic --games 100 --opponent-decks top_winrate --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent sequence-starmie-froslass --games 100 --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent sequence --games 100 --out-dir "$RUN"
./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent sequence --games 100 --opponent-decks deck_db --out-dir "$RUN"
```

Then compare against the same commands with `AGENT='sequence'` in a separate output directory, unless a fresh `sequence` baseline already exists from the same code state.

## Stress Baselines

Use these when the candidate changes search, state parsing, probability modeling, opponent memory, or action sanitization.

| Stress test | Command |
| --- | --- |
| Longer random mirror | `./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent random --games 300 --out-dir "$RUN"` |
| Longer deck-db rotation | `./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent random --games 300 --opponent-decks deck_db --out-dir "$RUN"` |
| Longer heuristic mirror | `./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent heuristic --games 300 --out-dir "$RUN"` |
| Longer Starmie/Froslass Kaggle-best mirror | `./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent sequence-starmie-froslass --games 300 --out-dir "$RUN"` |
| Longer sequence mirror | `./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent sequence --games 300 --out-dir "$RUN"` |
| Longer sequence deck-db rotation | `./.venv/bin/python -m pokemon_agent.eval --agent "$AGENT" --opponent sequence --games 300 --opponent-decks deck_db --out-dir "$RUN"` |
| Submission package smoke | `./.venv/bin/python -m pokemon_agent.submission --output "$RUN/submission.tar.gz"` |
