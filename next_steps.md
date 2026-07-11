# Next Steps

Ideas from the research report that look like useful additions to the current architecture. Completed experiments are recorded separately so the remaining priority list stays actionable.

## Completed Experiments

### Grouped Immediate-Root Ranker

Promoted as an opt-in policy after a frozen 30-deck, five-rotation evaluation. The compact XGBoost-equivalent runtime achieved 61.07% versus 50.29% for `sequence` over 4,500 games per arm, with zero invalid actions and a paired improvement of 10.78 percentage points (95% CI 8.89 to 12.66).

The retained design ranks canonical immediate action roots rather than full sequence variants. Native XGBoost is used only for offline training; production inference uses the checked-in compact JSON model and a dependency-free loader. Linear guardrails can still select a safety fallback outside the model's 24-root selectable cap.

### Rejected: Hidden-State Root Aggregation

A belief-root determinization experiment aggregated candidate values across sampled hidden states. In the frozen 4,500-game-per-arm matrix it reached 49.58% versus 48.73% for its baseline, a +0.84 percentage-point result whose confidence interval crossed zero. Prize performance worsened and the apparent gain was concentrated by seat, so the implementation was reverted.

### Rejected: Structured Critique Penalties

Relational critique rules were evaluated on 5,917 untouched validation groups using the committed root ranker. Rules covered missed forced wins, missed KO lines, ignored must-answer targets, premature endings, counter-KO exposure, low-value energy attachments, and setup overcommitment. No rule met the predeclared support and +0.25 percentage-point top-1 gate. The best supported rule improved expert-root top-1 by only 0.034 points; several intuitive rules reduced accuracy. No action-changing critique code was retained.

### Rejected: Analytic Opponent-Response Tail Reranking

Expected-risk and upper-tail aggregations over the existing active, gust, switch, energy, tool, and evolution response components were tested on the same validation split. Max-risk and tail-risk penalties reduced top-1 accuracy by roughly 0.12 to 0.15 percentage points; the best fragility variant changed two groups and gained one correct prediction (+0.017 points), below the promotion gate.

True legal opponent-response search remains a larger architectural option rather than a ready evaluator feature. Simulator branch handles are valid only during sequence generation and are closed before XGBoost identifies the top roots. Implementing honest top-root reply search therefore requires a live-search callback or deterministic replay/re-root layer plus an opponent policy, rather than resuming stored candidate IDs.

## Priority Order

1. Expand decision logging around ranker/guardrail disagreements.
2. Replace the heuristic value proxy with a calibrated value model.
3. Add local replay retrieval features.
4. Add opponent-belief confidence and a robust/exploit gate.
5. Revisit legal opponent-response search only after adding resumable search sessions and an opponent policy.

## Ideas To Try

### Richer Decision Logging

Add a stable JSONL schema for every decision:

- root state summary
- legal actions
- generated candidate sequences
- candidate feature vectors
- score breakdowns
- selected action
- hidden-sample support
- guardrail or critique reason codes
- terminal outcome or later reward

This should come first because it feeds retrieval, ranking, value training, and critique labels.

### Local Replay Retrieval

Build a local decision-memory index from previous games. Retrieval should add evaluator features, not directly override policy.

Possible features:

- `retrieval_support_count`
- `retrieval_mean_return`
- `retrieval_lower_quartile_return`
- `retrieval_action_success_rate`
- `retrieval_novelty`

Start with handcrafted numeric state/action embeddings from existing features before trying learned embeddings.

### Calibrated Value Model

Replace or augment the current heuristic value proxy with a model that predicts win probability or expected return for a candidate sequence.

Track calibration separately from win rate:

- Brier score
- calibration error
- value prediction by matchup/archetype
- disagreement with the ranker

### Opponent Confidence And Exploit Gate

The current belief system already tracks opponent archetype and intent. Add an explicit confidence or entropy signal, then use it to blend robust and exploitative scoring.

Runtime idea:

```text
hybrid_score = robust_score * (1 - exploit_weight) + exploit_score * exploit_weight
```

Keep `exploit_weight` low early in the game or when archetype confidence is weak.

### Root Opponent Response Search

For the top few self candidates, estimate likely opponent replies and penalize lines that are fragile to immediate counterplay.

Start small:

- top 3 self candidates
- top 2 opponent replies
- short response depth
- risk-adjusted aggregation such as p20 or CVaR

This should be tried after evaluator quality improves, because deeper response search is only useful if the agent can evaluate the resulting states reliably.

## Evaluation Notes

Use the existing baseline harness before promoting any idea:

- compare against `sequence`
- use paired scenarios where possible
- require zero invalid actions
- watch p95/p99 latency, not just average steps
- keep changes that improve broad suites, not just one matchup

Do not merge multiple idea families at once until each one has a clean ablation.
