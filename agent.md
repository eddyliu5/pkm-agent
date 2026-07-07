# Project Development Rules

Use an object-oriented approach for new development. Split behavior into focused modules so each part is easier to develop, test, and debug.

Keep helper components isolated and stable. Once these helpers are unit tested and working correctly, avoid modifying them unless their public contract needs to change.

Keep trainable components and architecture experiments in separate modules. Policies, feature extractors, trainers, model architectures, search/planning strategies, evaluation harnesses, and self-play evolution logic should be independently replaceable so they can evolve without destabilizing the helper layer.

Overall architecture is documented in ./pokemon_agent/architecture.md, and individual module descriptions are in their respective folder's md files. When making changes, update descriptions to keep them up-to-date.

Each module must have an explicit purpose and target behavior. For trainable modules, also define the measurable goal or metric they optimize or are evaluated against, such as win rate, invalid-action rate, average reward, checkpoint score, evaluation score versus a baseline, or another task-specific metric.

Refer to ./loop.md when adding or removing features.

Reduce use of fallbacks: if a module fails, I want to know about it without silently failing and going to fallback.