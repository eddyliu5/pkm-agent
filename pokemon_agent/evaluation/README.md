# Evaluation Package

Purpose: score complete candidate sequences against strategic goals, risk, hidden-information probability, opponent counterplay, learned ranking, and heuristic outcome proxies.

Target behavior: evaluation is pure for a fixed observed context, planning result, and model state. It returns immutable `ScoredCandidate` records with feature dictionaries, final scalar scores, score components, ranks, and diagnostics. It must not mutate planner candidates.

Trainable modules: sequence rankers and future value/risk models belong here. Metrics are model top-1 when the expert action is present, expert top-k rate, score component ablation win rate, evaluation score versus baseline policies, invalid-action rate after selection, and schema compatibility failures caught at load time.
