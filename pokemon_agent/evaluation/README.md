# Evaluation Package

Purpose: score complete candidate sequences against strategic goals, risk, hidden-information probability, opponent counterplay, learned ranking, and heuristic outcome proxies.

Target behavior: evaluation is pure for a fixed observed context, planning result, and model state. It returns immutable `ScoredCandidate` records with feature dictionaries, final scalar scores, score components, ranks, and diagnostics. It must not mutate planner candidates.

Trainable modules: the MLP sequence ranker is the current training target. Future value/risk models may live here after the ranker path is stable. Metrics are model top-1 when the expert action is present, expert top-k rate, score component ablation win rate, evaluation score versus baseline policies, invalid-action rate after selection, and strict schema failures caught at load time.

Unnecessary for the current workflow: backward-compatible schema loading, silent model fallback, and treating linear-weight training as the main improvement path. Runtime/schema errors should fail fast.
