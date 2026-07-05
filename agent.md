# Project Development Rules

Use an object-oriented approach for new development. Split behavior into focused modules so each part is easier to develop, test, and debug.

Keep helper components isolated and stable. Examples include legal-action parsing, state parsing, card metadata lookups, config loading, reward logging, gameplay logging, and self-play data capture. Once these helpers are unit tested and working correctly, avoid modifying them unless their public contract needs to change.

Keep trainable components and architecture experiments in separate modules. Policies, feature extractors, trainers, model architectures, search/planning strategies, evaluation harnesses, and self-play evolution logic should be independently replaceable so they can evolve without destabilizing the helper layer.

Each module must have an explicit purpose and target behavior. For trainable modules, also define the measurable goal or metric they optimize or are evaluated against, such as win rate, invalid-action rate, average reward, checkpoint score, evaluation score versus a baseline, or another task-specific metric.
