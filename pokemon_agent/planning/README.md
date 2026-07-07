# Planning Package

Purpose: generate feasible candidate sequences under plausible hidden information using simulator search and budget controls.

Target behavior: planning answers which lines are possible. It may order expansion with cheap priors and enforce legality, duplicate elimination, simulator-impossibility, safety, time, node, depth, and beam budgets. It must not assign final policy utility or use probability features to erase branches before evaluation.

Planning may include a warm-start continuation from the previously selected sequence when the next cached step still matches the current legal option identity. Warm-start continuations are candidates for evaluation, not committed actions.

Trainable modules: none directly. Planner quality is measured by expert-action-in-candidates rate, candidate coverage by action type, number of represented legal first actions, missing first-action diagnostics, node/time budget hit rates, and probability-on/off candidate coverage stability.
