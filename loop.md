# Iteration Loop

Use this every time you add, remove, or change one feature. Keep each run small and reversible.

## Rule

Change one concept at a time. Compare against a pinned baseline. Keep only if it improves or clearly preserves strength while fixing a specific defect.

## Loop

```text
baseline -> hypothesis -> implement -> smoke -> eval -> compare -> keep/revert -> log
```

## 1. Baseline

Before editing, record:

```text
baseline_name:
baseline_commit:
baseline_config:
eval_command:
key_metrics:
```

Do not move the baseline during an experiment.

## 2. Hypothesis

Write one sentence:

```text
Changing <module> should improve <metric/behavior> because <reason>.
```

Examples:

```text
Moving probability out of pruning should improve candidate recall because low-probability lines can still be tactically correct.
Adding candidate diversification should improve first-action coverage because beam search currently over-selects similar lines.
```

## 3. Scope

Allowed change size:

```text
module:
files:
config flag:
expected behavior change:
rollback plan:
```

No unrelated cleanup during experiments.

## 4. Smoke Test

Run the fastest checks first:

```text
import/package test
1-5 games vs baseline
no invalid-action increase
no crash in observe/plan/evaluate/act
```

If smoke fails, revert or fix only the smoke failure. Smoke runs are for crash and invalid-action screening only; do not promote a change from smoke results.

## 5. Eval

Use fixed seeds when possible. Use `./baseline/` as the source of truth for the current baseline agents, commands, proxy score, and reporting format.

Minimum tiers:

```text
smoke: 1-5 games, reject only
decision: 100+ games per relevant matchup
final: 300+ games per important matchup when time allows
```

Do not promote a change based on fewer than 100 games per relevant matchup. Runs below 100 games are only for crash, runtime, and invalid-action screening.

Record:

```text
win_rate
avg_prize_diff
invalid_actions
avg_turns
fallback_rate
candidate_count
first_action_coverage
expert_action_in_candidates, if replay eval
```

## 6. Decision

Keep if:

```text
win_rate improves meaningfully, OR
prize_diff improves with stable win_rate, OR
it fixes a targeted bug without strength loss.
```

Revert if:

```text
invalid_actions increase
candidate coverage drops unexpectedly
fallback_rate spikes
100+ game decision eval is clearly worse
change affects multiple modules without clear attribution
```

Mark inconclusive if results are noisy. Then run more seeds or simplify the change.

## 7. Log

Append one entry:

```text
## YYYY-MM-DD experiment_name
Hypothesis:
Change:
Eval:
Result:
Decision: keep | revert | inconclusive
Notes:
```

## Preferred Experiment Order

1. Disable new module entirely.
2. Enable it as evaluation-only.
3. Add it as soft score with small weight.
4. Add it as expansion ordering only.
5. Allow hard pruning only after coverage proves safe.

## Codex Instruction

```text
Run one iteration only. Start from the pinned baseline. Make one scoped change behind a config flag. Run smoke, then a 100+ game decision eval using the current procedure in ./baseline/. Compare against baseline metrics. Keep, revert, or mark inconclusive. Update the experiment log with exact commands and results. Do not perform unrelated refactors.
```
