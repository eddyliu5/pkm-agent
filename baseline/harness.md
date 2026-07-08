# Rigorous Baseline Harness

This protocol is the stricter replacement for judging whether a candidate agent is meaningfully better than the current production baseline.

## Current Protocol Assessment

The existing baseline protocol has useful foundations:

- It checks smoke/runtime/package contracts before promotion.
- It includes weak, heuristic, deck-specific, saved-artifact, and sequence opponents.
- It rotates opponent decks through `deck_db` and top observed win-rate decks.
- It records raw JSONL/CSV artifacts and tracks invalid actions.
- The evaluator is deterministic by game index, so separate runs can be compared on matching game ids.

The main weaknesses are statistical and operational:

- Most decisions are made from aggregate win rates, not paired per-scenario deltas.
- There is no confidence interval, minimum detectable effect, or repeat rule.
- Packaging gates mainly use heuristic mirror win rate, which can miss broad regressions.
- The local Kaggle proxy uses only three suites and overweights weak opposition.
- Deck-db rotations are not stratified by archetype, so one deck cluster can dominate a result.
- There is no manifest tying an evaluation to git state, agent spec, config, deck id, suite list, and evaluator options.
- Runtime health is summarized by average steps only; long-tail games and planner budget hits need visibility.
- Quick 24/25-game runs are useful for breakage detection but too noisy to distinguish close agents.

## Harness Goal

The harness should answer one question:

```text
Is the candidate agent better than the control agent by enough margin, across enough realistic scenarios, that the result is unlikely to be noise or overfit?
```

A good harness must be fast at the first tier, reliable at the promotion tier, and hard to fool with narrow improvements.

## Evaluation Tiers

| Tier | Purpose | Games per suite | Decision |
| --- | --- | ---: | --- |
| Smoke | Contract and runtime breakage. | 1-5 targeted tests | Block on any failure. |
| Quick paired | Reject obvious regressions. | 32 | Continue only if no invalid actions and no broad suite collapse. |
| Standard paired | Decide whether a change is promising. | 128 | Promote only with positive weighted delta and no critical-suite regression. |
| Confirmation | Reduce noise before packaging/submission. | 384+ | Package/submit only if the standard signal repeats. |

Use powers or multiples that divide cleanly across deck pools. Avoid treating a single quick run as proof.

## Standard Suite

Run the same suite for control and candidate. The current control should usually be `sequence`.

| Suite id | Opponent | Deck pool | Weight | Role |
| --- | --- | --- | ---: | --- |
| `random_mirror` | `random` | mirror | 0.10 | Catches tactical collapse and invalid wrappers. |
| `random_deck_db` | `random` | `deck_db` | 0.20 | Measures varied-deck robustness against a weak policy. |
| `heuristic_mirror` | `heuristic` | mirror | 0.15 | Static feature baseline on the configured deck. |
| `deck_heuristic_top_winrate` | `deck-heuristic` | `top_winrate` | 0.15 | Tests high-observed-winrate archetypes with deck-aware roles. |
| `sequence_starmie_froslass` | `sequence-starmie-froslass` | mirror | 0.15 | Regression check against the saved Kaggle-best local artifact. |
| `sequence_mirror` | `sequence` | mirror | 0.10 | Head-to-head against current production search. |
| `sequence_deck_db` | `sequence` | `deck_db` | 0.15 | Strong varied-deck search check. |

Weighted score is the weighted average of suite win rates on a 0-100 scale. It is not a Kaggle prediction; it is a local promotion signal.

## Paired Comparison

For every suite, compare candidate and control by matching `game` ids from the evaluator CSV files. This is valid because `pokemon_agent.evaluation.match_eval` fixes side, deck selection, and policy seeds from the game index.

Report both:

- Aggregate win-rate delta: `candidate_win_rate - control_win_rate`.
- Paired win-rate delta: mean over matching games of `candidate_win_indicator - control_win_indicator`.

Prefer the paired delta when judging close changes. It removes variance from side order, deck order, and random-opponent seed order.

## Gates

Block immediately if any of these are true:

- Smoke tests fail.
- Candidate `invalid_actions > 0` in any suite.
- Any evaluation crashes or produces missing/empty CSV output.
- Candidate average steps exceeds control by more than 20% in a suite and the win-rate delta is not positive.
- Planner coverage reports budget-hit spikes for sequence agents without a matching win-rate gain.

Reject or revise if any critical suite has a paired 95% lower confidence bound below `-0.02`:

- `random_deck_db`
- `deck_heuristic_top_winrate`
- `sequence_starmie_froslass`
- `sequence_deck_db`

Promote from standard to confirmation only if:

- Candidate weighted score improves by at least 1.5 points.
- Candidate has no invalid actions.
- At least four of seven suites have non-negative paired deltas.
- No critical suite violates the regression gate.

Promote from confirmation to packaging/submission only if:

- The weighted score improvement remains at least 2.0 points.
- The critical-suite paired lower confidence bounds are all at least `-0.01`.
- The candidate beats or matches control on `sequence_deck_db`.
- The package smoke succeeds with the exact runtime artifacts used during evaluation.

## Required Run Manifest

Every harness run should write a manifest next to the artifacts:

```json
{
  "run_id": "20260708T000000Z",
  "tier": "standard",
  "control_agent": "sequence",
  "candidate_agent": "sequence-mlp:artifacts/checkpoints/example/best.json",
  "config": "configs/starmie_froslass.json",
  "agent_deck_id": null,
  "games_per_suite": 128,
  "workers": 1,
  "suite_ids": [
    "random_mirror",
    "random_deck_db",
    "heuristic_mirror",
    "deck_heuristic_top_winrate",
    "sequence_starmie_froslass",
    "sequence_mirror",
    "sequence_deck_db"
  ],
  "git_head": "<commit>",
  "git_dirty": true
}
```

## Commands

Use the harness runner for normal comparisons:

```bash
CONTROL_AGENT=sequence \
CANDIDATE_AGENT='sequence-mlp:artifacts/checkpoints/example/best.json' \
TIER=standard \
./scripts/run_baseline_harness.sh
```

Useful knobs:

```bash
GAMES=32                 # override tier game count
EVAL_WORKERS=4           # parallel evaluator workers
CONFIG=configs/foo.json  # non-default config
AGENT_DECK_ID=<deck_id>  # selected deck-db deck for the agent
```

The runner writes control artifacts, candidate artifacts, `manifest.json`, and `comparison.json` under `artifacts/baseline/harness_<run_id>/`.

To compare existing artifacts without rerunning games:

```bash
./.venv/bin/python -m pokemon_agent.evaluation.baseline_compare \
  --control-dir artifacts/baseline/harness_<run_id>/control \
  --candidate-dir artifacts/baseline/harness_<run_id>/candidate \
  --out-json artifacts/baseline/harness_<run_id>/comparison.json
```

## Future Hardening

The next worthwhile evaluator changes are:

- Add an explicit `--seed-offset` or `--scenario-file` so reruns can use independent paired scenario sets.
- Emit per-game wall-clock duration and timeout flags.
- Emit suite summaries as machine-readable JSON files, not only printed JSON and CSV rows.
- Add archetype-stratified deck-db summaries so broad robustness cannot hide a severe matchup regression.
- Add fixed tactical micro-scenarios for known failure modes: empty-board survival, final-prize attack selection, gust denial, energy attachment, and setup-vs-attack tempo.
