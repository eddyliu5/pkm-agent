# Observation Package

Purpose: convert raw simulator observations into agent-owned state, legal actions, persistent opponent memory, hidden-information belief, probability summaries, game-plan phase, resources, and threats.

Target behavior: observation is deterministic for a fixed raw observation and memory state. It must not run simulator search, score candidate lines, or make policy decisions.

Trainable modules: none in this package. Evaluation metrics are parser determinism, legal-action parse correctness, prize-belief card-count consistency, weighted opponent-hypothesis updates, remaining-count/exhaustion consistency, revealed-hand tracking, missed-response evidence updates, and probability snapshot stability for unchanged visible/belief inputs.
