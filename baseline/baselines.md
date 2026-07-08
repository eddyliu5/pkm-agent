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
| `sequence` | Current default production baseline. | Main baseline for regressions and submissions. The training script's `RUNTIME_POLICY=auto` mode packages this unless an MLP candidate beats it locally. |
| `sequence-starmie-froslass` | Saved Starmie/Froslass sequence submission weights from `artifacts/submission/sequence_starmie_froslass_submission.tar.gz`. | Regression gate against the current best Kaggle-scoring local artifact. |
| `sequence:<weights.json>` | Sequence search with learned linear weights. | Unnecessary for the current improvement path; keep only as an ablation/debug baseline. |
| `sequence-mlp:<mlp.pt>` | Sequence search with an MLP ranker. | Compare MLP checkpoints against default sequence and learned-linear variants before packaging. |
| `sequence-mlp:<mlp.pt>,<weights.json>` | Linear weights for search/fallback plus MLP-only final ranking. | Ablation only. Replay-trained linear weights must pass safety checks before they are used at runtime. |
| `sequence-clean` | Clean evaluator path without direct probability/value/resource/threat/card-attention final scoring. | Architecture experiments and clean scoring regressions. |
| `sequence-clean-mlp:<mlp.pt>,<weights.json>` | Clean evaluator plus linear search weights and MLP-only final ranking. | Clean-schema ablation testing. |

For MLP retrains after label or architecture fixes, rebuild the compact MLP label file from raw labels instead of reusing stale compact labels:

```bash
TRAIN_ONLY=1 SKIP_LABEL_GENERATION=1 SKIP_SEQUENCE_TRAIN=1 ./scripts/train_clean_models.sh
```

`TRAIN_ONLY=1` stops before deck sweep, packaging, and submission.

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
| Top-deck sequence matrix | `scripts/run_top_deck_sequence_compare.py` | Standard sequence-policy baseline. Runs each top observed-winrate self deck against a round-robin top-winrate `sequence` opponent pool with fixed per-round seeds. Default is 5 full 30-deck rotations, 150 games per self deck. |

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
| Package gate | `scripts/train_clean_models.sh` refuses to package when the selected runtime's heuristic mirror win rate is below `MIN_HEURISTIC_WIN_RATE_FOR_PACKAGE`. |
| Runtime weights | Runtime linear weights must keep setup/action-safety signs sane unless `ALLOW_UNSAFE_RUNTIME_WEIGHTS=1` is explicitly set. |

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

For sequence-policy changes, also run the standardized top-deck matrix:

```bash
./.venv/bin/python scripts/run_top_deck_sequence_compare.py
```

Use `--rounds N` to change the number of complete top-deck opponent rotations. `--games` is retained for old automation, but it must be a multiple of the selected top-deck count. The default fixed seed schedule starts at `42` and is capped at 30 rounds.

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
