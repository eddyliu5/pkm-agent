# Next Steps

Ideas from the research report that look like useful additions to the current architecture, but have not been tried yet.

## Priority Order

1. Add richer decision logging.
2. Train a grouped candidate ranker baseline.
3. Add structured critique reason codes.
4. Add local replay retrieval features.
5. Replace the heuristic value proxy with a calibrated value model.
6. Add opponent-belief confidence and a robust/exploit gate.
7. Try root-level opponent response search.

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

### Grouped Candidate Ranker

Train a LightGBM or XGBoost style ranker offline using each decision as one query group and candidate sequences as ranked items. Keep this as an experiment first; only consider runtime packaging if the dependency/runtime cost is worth it.

Useful labels:

- expert action from replay labels
- stronger-search top action
- self-play return
- pairwise preference between candidates from the same decision

### Structured Critique

Add deterministic reason codes after evaluation and before action selection. Start with bounded penalties, not hard rewrites.

Candidate reason codes:

- `missed_ko_line`
- `exposed_active_to_ko`
- `ended_with_useful_action_available`
- `wasted_energy_attachment`
- `overcommitted_setup`
- `ignored_must_answer_target`
- `low_confidence_retrieval`

Log every critique so it can later become training data.

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
