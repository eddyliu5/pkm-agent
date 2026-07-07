# Pokemon Agent Architecture

`pokemon_agent` is a Pokemon TCG simulator agent built around a sequence-first decision loop:

```text
observe -> plan -> evaluate -> act
```

`DecisionEngine` is the runtime orchestrator. It owns the long-lived opponent memory, observation builder, sequence simulator, sequence evaluator, action selector, and fallback policy. The engine exposes each stage as a public method for testing and diagnostics, while `choose(obs)` runs the full loop and returns one immediate simulator selection.

Module-level details live in each folder's `README.md`. This document is the package-level map and should be updated whenever a feature changes the runtime architecture.

## Runtime Entrypoints

`main.py` imports `pokemon_agent.runtime_agent.agent` for Kaggle submission. `runtime_agent.py` locates the bundle root, reads `deck.csv`, loads a deck config, optional `weights.json`, optional `mlp_model.json`, and optional `runtime_config.json`, then creates one process-global `DecisionEngine`.

At inference time:

```text
agent(obs_dict)
  -> RuntimeAdapter.parse(obs_dict)
  -> DecisionEngine.choose(obs)
  -> RuntimeAdapter.clamp(obs, selected, deck)
```

`RuntimeAdapter` converts raw Kaggle dictionaries into simulator observations and clamps returned indices to the current selection bounds.

`policy.py` provides reusable policies for evaluation and self-play: random, linear heuristic, deck-specific heuristic, epsilon-greedy, and `SequencePolicy`. `SequencePolicy` wraps `DecisionEngine` with the same observe-plan-evaluate-act path used at runtime.

## Decision Contracts

The cross-stage data contracts are in `decision_types.py`.

- `ObservedContext`: raw observation, legal `ActionView`s, parsed `AgentState`, persistent `OpponentMemory`, `BeliefState`, probability snapshot, game-plan context, resource snapshot, threat snapshot, and diagnostics.
- `PlanningResult`: generated `CandidateSequence`s, optional `SearchTree`, coverage metrics, pruned-action diagnostics, hidden samples used, and planner diagnostics.
- `ScoredCandidate`: immutable evaluation output for one candidate, including features, scalar score, component breakdown, rank, and warnings.
- `EvaluationResult`: ranked scored candidates plus score calibration and diagnostics.
- `AgentDecision`: final selected indices, chosen sequence view when applicable, scored candidate, and decision trace.

The simulator accepts only the next legal selection, so even when planning explores a multi-step line, acting commits only `candidate.first_selection`.

## Observation Layer

`observation/` builds decision-time context without choosing actions.

`ObservationBuilder` wires together:

- `actions.py`: parses legal simulator options into `ActionView`s and sanitizes selections.
- `state.py`: converts simulator observations into `AgentState`, `CardView`, and `PokemonView` structures.
- `opponent_memory.py`: persists visible opponent cards, recent actions, setup targets, and inferred intent.
- `belief.py`: builds own hidden-information samples, prize belief, and opponent archetype/intent belief.
- `probability.py`: derives probability features from deck, visible cards, prizes, and belief state.
- `game_plan.py`: builds phase and deck-strategy context from config card groups.
- `observation/snapshots.py`: computes resource and threat snapshots.

Observation is deterministic for a fixed raw observation and memory state. It does not run simulator search or assign candidate utility.

## Planning And Sequence Search

`sequence/` and `planning/` generate feasible candidate lines under plausible hidden information.

`FeasibleSequenceSimulator` is the planner facade. It uses hidden samples from `BeliefState`, `SearchAdapter` around the simulator `search_begin`, `search_step`, and `search_end` API, and `SearchBudget` limits for depth, beam width, node count, time, option pool size, and hidden sample count.

`planning.SequencePruner` supplies planner-local selection candidates: it filters illegal or duplicate choices, applies budget-oriented expansion priorities, and records pruned actions. The score stored on `SelectionStep` and `CandidateSequence.planner_score` is a planner priority, not final policy utility.

Important sequence types:

- `SequenceSearchConfig`: search depth, beam, node, time, option-pool, combination, and hidden-sample limits.
- `SelectionStep`: selected indices, associated actions, local features, planner priority, and labels.
- `SearchTree`/`SearchNode`: parent-linked simulator search tree with path reconstruction.
- `CandidateSequence`: immutable planned line with steps, final observation/state, search metadata, generation tags, and simulator result.

If simulator search is unavailable or fails, planning falls back to single-step candidate generation. `DecisionEngine` also ensures the fallback linear-policy action is represented as a candidate when it is legal and missing.

## Evaluation Layer

`evaluation/` scores complete candidate sequences after planning.

`SequenceEvaluator` featurizes each candidate using either `SequenceFeaturizer` or `CleanSequenceFeaturizer`, then computes:

- linear feature score through `SequenceScorer`
- optional MLP ranker score from `MLPSequenceRanker`
- optional heuristic state-delta value proxy
- risk-model score
- opponent-counterplay score
- plan-priority bonus

The result is a ranked tuple of `ScoredCandidate` records. Evaluation does not mutate `CandidateSequence`; compatibility views are created only when older training/evaluation code expects sequence-shaped scored objects.

`sequence/featurizer.py` owns candidate feature extraction. The full featurizer includes action, consequence, game-plan, resource, threat, opponent, probability, card-knowledge, and field-attention features. The clean featurizer keeps a narrower consequence-focused schema for clean training/evaluation runs.

## Acting Layer

`acting/` converts evaluated candidates into one legal immediate selection.

`ActionSelector` chooses the top-ranked candidate in eval mode and samples from candidate scores in train mode. `FallbackGuardrail` can override the evaluator in eval mode when the fallback linear policy is safer and the chosen candidate is not convincingly better. All final selections are passed through `sanitize_selection`; invalid selections fall back to `FallbackLinearPolicy`.

Decision traces record sequence count, whether simulator search was used, chosen score, chosen labels, and any fallback reason.

## Card And Model Data

The agent reasons primarily over integer card IDs.

Card IDs come from `deck.csv`, `configs/*.json`, simulator observations, and `data/deck_db/decks.json`. `card_db.py` and `knowledge.py` attach metadata, strategic roles, card embeddings, evolution flags, attack data, energy classification, and prize values. Config card groups drive game-plan features, action labels, probability summaries, and deck-specific heuristic behavior.

Trainable or learned components are intentionally replaceable:

- linear weights in `weights.json`
- JSON MLP sequence rankers in `mlp_model.json`
- sequence feature schemas in `SequenceFeaturizer` and `CleanSequenceFeaturizer`
- training and replay utilities in `train_linear.py`, `torch_mlp_trainer.py`, and `replay_pipeline.py`

Runtime options in `runtime_config.json` can enable or disable clean evaluation, probability features, MLP scoring, value proxy scoring, fallback guardrails, score weights, and hidden sample count.
