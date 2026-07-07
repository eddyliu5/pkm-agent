# Evaluation Package

Purpose: score complete candidate sequences against strategic goals, risk, hidden-information probability, opponent counterplay, learned ranking, and heuristic outcome proxies.

Target behavior: evaluation is pure for a fixed observed context, planning result, and model state. It returns immutable `ScoredCandidate` records with feature dictionaries, final scalar scores, score components, ranks, and diagnostics. It must not mutate planner candidates.

Trainable modules: the MLP sequence ranker is the current training target. Linear weights are retained for search/fallback support; when an MLP checkpoint is loaded, final ranking should be MLP-only by default. Replay imitation labels default to per-player submitted decks rather than a single configured deck, so validation should cover self-deck archetype diversity. Future value/risk models may live here after the ranker path is stable. Metrics are validation top-1 when the expert action is present, validation top-k rate, score component ablation win rate, evaluation score versus baseline policies and top-winrate deck rotations, invalid-action rate after selection, and strict schema failures caught at load time.

Unnecessary for the current workflow: backward-compatible schema loading, silent model fallback, and treating linear-weight training as the main improvement path. Runtime/schema errors should fail fast.
