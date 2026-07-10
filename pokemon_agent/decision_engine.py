from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any

from .actions import sanitize_selection
from .acting import ActionSelector, FallbackGuardrail
from .decision_types import (
    AgentDecision,
    DecisionTrace,
    EvaluationResult,
    ObservedContext,
    PlanningCoverage,
    PlanningResult,
    ScoredCandidate,
)
from .evaluation import HeuristicValueProxy, SequenceEvaluator
from .features import combine_feature_dicts, extract_action_features
from .knowledge import CardKnowledgeStore
from .mlp_ranker import MLPSequenceRanker
from .observation.builder import ObservationBuilder
from .opponent_memory import OpponentMemory
from .policy import LinearQPolicy
from .sequence import (
    CandidateSequence,
    CleanSequenceFeaturizer,
    FeasibleSequenceSimulator,
    SequenceFeaturizer,
    SequenceScorer,
    SequenceSearchConfig,
    SelectionStep,
)


CandidateEvaluation = ScoredCandidate


@dataclass(frozen=True)
class WarmStartContinuation:
    """Validated candidate suffix from the previous selected sequence."""

    remaining_steps: tuple[SelectionStep, ...]
    final_observation: Any
    final_state: Any | None
    used_search: bool
    terminal: bool
    generation_tags: tuple[str, ...]
    sim_result: str | None
    turn: int
    player_index: int


class DecisionEngine:
    """Sequence-first observe -> plan -> evaluate -> act orchestrator."""

    ROOT_COVERAGE_COMBINATION_CAP = 2048

    def __init__(
        self,
        *,
        deck: list[int],
        config: dict[str, Any],
        weights: dict[str, float] | None = None,
        mlp_ranker: MLPSequenceRanker | None = None,
        linear_score_weight: float = 1.0,
        mlp_score_weight: float = 5.0,
        search_config: SequenceSearchConfig | None = None,
        clean_evaluator: bool = False,
        probability_features_enabled: bool = True,
        mlp_enabled: bool = True,
        value_proxy_enabled: bool = True,
        fallback_guardrail_enabled: bool = True,
        engine_exception_fallback_enabled: bool = False,
        seed: int | None = None,
    ):
        self.deck = [int(card_id) for card_id in deck]
        self.config = config
        self.search_config = search_config or SequenceSearchConfig()
        self.fallback = LinearQPolicy(weights=weights, config=config)
        self.opponent_memory = OpponentMemory()
        self.observation_builder = ObservationBuilder(
            config=config,
            decklist=self.deck,
            opponent_memory=self.opponent_memory,
            num_hidden_samples=self.search_config.num_hidden_samples,
        )
        self.knowledge = CardKnowledgeStore(config)
        self.simulator = FeasibleSequenceSimulator(config=config, weights=weights)
        self.clean_evaluator = bool(clean_evaluator)
        self.probability_features_enabled = bool(probability_features_enabled)
        self.value_proxy_enabled = bool(value_proxy_enabled)
        self.fallback_guardrail_enabled = bool(fallback_guardrail_enabled)
        self.engine_exception_fallback_enabled = bool(engine_exception_fallback_enabled)
        self.feature_schema_version = "clean" if self.clean_evaluator else "full"
        if not mlp_enabled:
            mlp_ranker = None
        if mlp_ranker is not None:
            mlp_ranker.validate_feature_schema(self.feature_schema_version)
        self.featurizer = (
            CleanSequenceFeaturizer(config=config)
            if self.clean_evaluator
            else SequenceFeaturizer(config=config, knowledge=self.knowledge, include_probability=self.probability_features_enabled)
        )
        self.scorer = SequenceScorer(
            weights=weights,
            mlp_ranker=mlp_ranker,
            linear_weight=linear_score_weight,
            mlp_weight=mlp_score_weight,
            clean=self.clean_evaluator,
        )
        self.value_model = HeuristicValueProxy()
        self.evaluator = SequenceEvaluator(
            featurizer=self.featurizer,
            scorer=self.scorer,
            value_proxy=self.value_model,
            clean=self.clean_evaluator,
            value_proxy_enabled=self.value_proxy_enabled,
        )
        self.selector = ActionSelector(seed=seed)
        self.guardrail = FallbackGuardrail()
        self.last_trace: DecisionTrace | None = None
        self.last_chosen_sequence: CandidateSequence | None = None
        self.last_observed_context: ObservedContext | None = None
        self.last_planning_result: PlanningResult | None = None
        self.last_evaluation_result: EvaluationResult | None = None
        self.last_decision: AgentDecision | None = None
        self.last_planning_coverage: PlanningCoverage | None = None
        self.pending_warm_start: WarmStartContinuation | None = None
        self.last_warm_start_status: str | None = None

    def choose(self, obs: Any, mode: str = "eval") -> list[int]:
        if obs.select is None:
            self.last_chosen_sequence = None
            self._clear_warm_start("missing_selection")
            return []
        if self.engine_exception_fallback_enabled:
            try:
                decision = self._choose_sequence(obs, mode=mode)
            except Exception:
                return self._fallback_choose(obs)
        else:
            decision = self._choose_sequence(obs, mode=mode)
        self._arm_opponent_response_evidence(decision)
        self.last_chosen_sequence = decision.chosen_sequence
        self.last_trace = decision.trace
        self.last_decision = decision
        return decision.selected

    def _arm_opponent_response_evidence(self, decision: AgentDecision) -> None:
        observed = self.last_observed_context
        scored = decision.scored_candidate
        if observed is None or scored is None or decision.chosen_sequence is None or not scored.candidate.steps:
            return
        first_actions = scored.candidate.steps[0].actions
        if not any(action.is_attack or action.is_end for action in first_actions):
            return
        opportunities = self.evaluator.counterplay_model.response_opportunities(
            scored.candidate.state,
            observed.belief.opponent,
        )
        self.opponent_memory.arm_response_opportunities(observed.state.turn, opportunities)

    def score_sequences(self, obs: Any) -> list[CandidateSequence]:
        if obs.current is None:
            return []
        return self.evaluate(self.plan(self.observe(obs))).sequences

    def observe(self, obs: Any) -> ObservedContext:
        return self.observation_builder.build(obs)

    def plan(self, observed: ObservedContext) -> PlanningResult:
        sequences = self.simulator.generate(
            observed.observation,
            observed.state,
            observed.belief,
            self.search_config,
        )
        sequences = self._ensure_warm_start_candidate(observed, sequences)
        sequences = self._ensure_root_coverage_candidates(observed, sequences)
        sequences = self._ensure_fallback_candidate(observed, sequences)
        simulator_diagnostics = dict(self.simulator.last_diagnostics)
        coverage = self._planning_coverage(observed, sequences, simulator_diagnostics)
        self.last_planning_coverage = coverage
        return PlanningResult(
            observed=observed,
            tree=self.simulator.last_tree,
            candidates=tuple(sequences),
            coverage=coverage,
            pruned_actions=tuple(getattr(self.simulator.pruner, "last_pruned_actions", ())),
            hidden_samples_used=tuple(str(value) for value in simulator_diagnostics.get("hidden_samples_used", ())),
            diagnostics={
                "used_search_tree": self.simulator.last_tree is not None,
                **simulator_diagnostics,
                "budget_missing_first_actions": coverage.missing_first_actions,
                "warm_start_status": self.last_warm_start_status,
            },
        )

    def evaluate(
        self,
        observed_or_planning: ObservedContext | PlanningResult,
        planning: PlanningResult | None = None,
    ) -> EvaluationResult:
        if planning is None:
            planning = observed_or_planning  # type: ignore[assignment]
        return self.evaluator.evaluate(planning)

    def act(
        self,
        observed_or_evaluated: ObservedContext | EvaluationResult,
        planning: PlanningResult | str | None = None,
        evaluated: EvaluationResult | None = None,
        mode: str = "eval",
    ) -> AgentDecision:
        if isinstance(planning, str):
            mode = planning
            planning = None
        if evaluated is None:
            evaluated = observed_or_evaluated  # type: ignore[assignment]
        obs = evaluated.planning.observed.observation
        scored = evaluated.scored
        if not scored:
            return self._fallback_decision(obs)

        chosen = self.selector.select(scored, mode=mode)
        if chosen is None:
            return self._fallback_decision(obs)
        fallback_selected, _ = sanitize_selection(obs, self.fallback.choose(obs))
        use_fallback, fallback_reason = self.guardrail.should_use_fallback(chosen, scored, fallback_selected)
        if mode == "eval" and self.fallback_guardrail_enabled and use_fallback:
            self._clear_warm_start(fallback_reason or "fallback_guardrail")
            return AgentDecision(
                selected=fallback_selected,
                chosen_sequence=None,
                scored_candidate=chosen,
                trace=DecisionTrace(
                    sequence_count=len(scored),
                    used_search=any(candidate.candidate.used_search for candidate in scored),
                    chosen_score=0.0,
                    chosen_labels=("fallback_guardrail",),
                    fallback_reason=fallback_reason,
                ),
            )
        selected, invalid = sanitize_selection(obs, chosen.candidate.first_selection)
        if invalid:
            return self._fallback_decision(obs)
        self._cache_warm_start(evaluated.planning.observed, chosen, selected)
        chosen_view = chosen.to_sequence_view()
        return AgentDecision(
            selected=selected,
            chosen_sequence=chosen_view,
            scored_candidate=chosen,
            trace=DecisionTrace(
                sequence_count=len(scored),
                used_search=any(candidate.candidate.used_search for candidate in scored),
                chosen_score=chosen.score,
                chosen_labels=tuple(label for step in chosen.candidate.steps for label in step.labels),
            ),
        )

    def _choose_sequence(self, obs: Any, mode: str) -> AgentDecision:
        observed = self.observe(obs)
        self.last_observed_context = observed
        planning = self.plan(observed)
        self.last_planning_result = planning
        evaluated = self.evaluate(observed, planning)
        self.last_evaluation_result = evaluated
        decision = self.act(observed, planning, evaluated, mode=mode)
        self.last_decision = decision
        return decision

    def _fallback_choose(self, obs: Any) -> list[int]:
        decision = self._fallback_decision(obs)
        self.last_chosen_sequence = decision.chosen_sequence
        self.last_trace = decision.trace
        self.last_decision = decision
        return decision.selected

    def _fallback_decision(self, obs: Any) -> AgentDecision:
        self._clear_warm_start("fallback_linear_policy")
        selected, _invalid = sanitize_selection(obs, self.fallback.choose(obs))
        return AgentDecision(
            selected=selected,
            chosen_sequence=None,
            trace=DecisionTrace(
                sequence_count=0,
                used_search=False,
                chosen_score=0.0,
                chosen_labels=("fallback_linear_policy",),
                fallback_reason="fallback_linear_policy",
            ),
        )

    def _cache_warm_start(
        self,
        observed: ObservedContext,
        chosen: ScoredCandidate,
        selected: list[int],
    ) -> None:
        sequence = chosen.candidate
        if tuple(selected) != tuple(sequence.first_selection) or len(sequence.steps) <= 1:
            self._clear_warm_start("no_remainder")
            return
        self.pending_warm_start = WarmStartContinuation(
            remaining_steps=tuple(sequence.steps[1:]),
            final_observation=sequence.observation,
            final_state=sequence.state,
            used_search=bool(sequence.used_search),
            terminal=bool(sequence.terminal),
            generation_tags=tuple(sequence.generation_tags),
            sim_result=sequence.sim_result,
            turn=int(observed.state.turn),
            player_index=int(observed.state.player_index),
        )
        self.last_warm_start_status = "cached"

    def _clear_warm_start(self, reason: str) -> None:
        self.pending_warm_start = None
        self.last_warm_start_status = f"cleared:{reason}"

    def _ensure_warm_start_candidate(
        self,
        observed: ObservedContext,
        sequences: list[CandidateSequence],
    ) -> list[CandidateSequence]:
        continuation = self.pending_warm_start
        if continuation is None:
            self.last_warm_start_status = "none"
            return sequences
        candidate = self._warm_start_candidate(observed, continuation)
        if candidate is None:
            return sequences
        candidate_key = self._sequence_key(candidate)
        if any(self._sequence_key(sequence) == candidate_key for sequence in sequences):
            self.last_warm_start_status = "duplicate"
            return sequences
        self.last_warm_start_status = "used"
        return [candidate, *sequences]

    def _warm_start_candidate(
        self,
        observed: ObservedContext,
        continuation: WarmStartContinuation,
    ) -> CandidateSequence | None:
        if observed.state.turn != continuation.turn or observed.state.player_index != continuation.player_index:
            self._clear_warm_start("turn_or_player_changed")
            return None
        if not continuation.remaining_steps:
            self._clear_warm_start("empty_remainder")
            return None
        cached_step = continuation.remaining_steps[0]
        if cached_step.select_type != observed.state.select_type or cached_step.context != observed.state.select_context:
            self._clear_warm_start("selection_context_changed")
            return None
        selected, invalid = sanitize_selection(observed.observation, list(cached_step.selected_indices))
        if invalid or tuple(selected) != tuple(cached_step.selected_indices):
            self._clear_warm_start("selection_invalid")
            return None
        actions_by_index = {action.index: action for action in observed.actions}
        current_actions = tuple(actions_by_index[index] for index in selected if index in actions_by_index)
        if len(current_actions) != len(selected):
            self._clear_warm_start("action_missing")
            return None
        if not self._cached_actions_match(cached_step, current_actions):
            self._clear_warm_start("action_mismatch")
            return None

        refreshed_features = [
            extract_action_features(observed.state, action, self.config, opponent_intent=observed.belief.opponent.intent)
            for action in current_actions
        ]
        refreshed_step = SelectionStep(
            selected_indices=tuple(selected),
            actions=current_actions,
            select_type=observed.state.select_type,
            context=observed.state.select_context,
            features=combine_feature_dicts(refreshed_features),
            score=cached_step.score,
            labels=tuple(action.label for action in current_actions),
        )
        remaining_steps = (refreshed_step, *continuation.remaining_steps[1:])
        return CandidateSequence(
            steps=remaining_steps,
            observation=continuation.final_observation,
            state=continuation.final_state,
            planner_score=sum(float(step.score) for step in remaining_steps),
            terminal=continuation.terminal,
            used_search=continuation.used_search,
            generation_tags=tuple(dict.fromkeys((*continuation.generation_tags, "warm_start"))),
            sim_result=continuation.sim_result or "warm_start",
        )

    @staticmethod
    def _cached_actions_match(cached_step: SelectionStep, current_actions: tuple[Any, ...]) -> bool:
        if len(cached_step.actions) != len(current_actions):
            return False
        for cached_action, current_action in zip(cached_step.actions, current_actions):
            if DecisionEngine._action_identity(cached_action) != DecisionEngine._action_identity(current_action):
                return False
        return True

    @staticmethod
    def _action_identity(action: Any) -> tuple[Any, ...]:
        return (
            getattr(action, "option_type", None),
            getattr(action, "source_player_index", None),
            getattr(action, "source_card_id", None),
            getattr(action, "target_player_index", None),
            getattr(action, "target_card_id", None),
            getattr(action, "attack_id", None),
            getattr(action, "label", None),
        )

    @staticmethod
    def _sequence_key(sequence: CandidateSequence) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(step.selected_indices) for step in sequence.steps)

    def _ensure_fallback_candidate(
        self,
        observed: ObservedContext,
        sequences: list[CandidateSequence],
    ) -> list[CandidateSequence]:
        fallback_selected, invalid = sanitize_selection(observed.observation, self.fallback.choose(observed.observation))
        if invalid or not fallback_selected:
            return sequences
        fallback_key = tuple(fallback_selected)
        if any(tuple(sequence.first_selection) == fallback_key for sequence in sequences):
            return sequences
        actions_by_index = {action.index: action for action in observed.actions}
        selected_actions = [actions_by_index[index] for index in fallback_selected if index in actions_by_index]
        if len(selected_actions) != len(fallback_selected):
            return sequences
        step_features = [
            extract_action_features(observed.state, action, self.config, opponent_intent=observed.belief.opponent.intent)
            for action in selected_actions
        ]
        step = SelectionStep(
            selected_indices=fallback_key,
            actions=tuple(selected_actions),
            select_type=observed.state.select_type,
            context=observed.state.select_context,
            features=combine_feature_dicts(step_features),
            score=0.0,
            labels=tuple(action.label for action in selected_actions),
        )
        return [
            *sequences,
            CandidateSequence(
                steps=(step,),
                observation=observed.observation,
                state=observed.state,
                planner_score=0.0,
                terminal=True,
                used_search=False,
                generation_tags=("fallback_linear",),
                sim_result="fallback",
            ),
        ]

    def _ensure_root_coverage_candidates(
        self,
        observed: ObservedContext,
        sequences: list[CandidateSequence],
    ) -> list[CandidateSequence]:
        if not observed.actions:
            return sequences

        existing = {tuple(sequence.first_selection) for sequence in sequences}
        additions: list[CandidateSequence] = []

        for step in self.simulator.pruner.selection_candidates(
            observed.observation,
            observed.state,
            self.search_config,
            observed.belief.opponent.intent,
        ):
            key = tuple(step.selected_indices)
            if key in existing:
                continue
            existing.add(key)
            additions.append(self._root_coverage_sequence(observed, step))

        represented_indices = {
            index
            for selection in existing
            for index in selection
            if isinstance(index, int)
        }
        for action in observed.actions:
            if action.index in represented_indices:
                continue
            step = self._root_coverage_step(observed, action.index)
            if step is None:
                continue
            key = tuple(step.selected_indices)
            if key in existing:
                continue
            existing.add(key)
            represented_indices.update(index for index in key if isinstance(index, int))
            additions.append(self._root_coverage_sequence(observed, step))

        for step in self._root_multi_select_coverage_steps(observed, existing):
            key = tuple(step.selected_indices)
            if key in existing:
                continue
            existing.add(key)
            additions.append(self._root_coverage_sequence(observed, step))

        if not additions:
            return sequences
        return [*sequences, *additions]

    def _root_multi_select_coverage_steps(
        self,
        observed: ObservedContext,
        existing: set[tuple[int, ...]],
    ) -> list[SelectionStep]:
        obs = observed.observation
        if obs.select is None or len(observed.actions) < 2:
            return []
        min_count = max(0, int(obs.select.minCount or 0))
        max_count = min(int(obs.select.maxCount or 0), len(observed.actions))
        if max_count < 2:
            return []
        counts = [count for count in range(min_count, max_count + 1) if count >= 2]
        if not counts:
            return []
        actions_by_index = {action.index: action for action in observed.actions}
        indices = [action.index for action in sorted(observed.actions, key=lambda action: action.index)]
        steps: list[SelectionStep] = []
        counts.sort(key=lambda count: (math.comb(len(indices), count), count))
        for count in counts:
            for combo in itertools.combinations(indices, count):
                if len(steps) >= self.ROOT_COVERAGE_COMBINATION_CAP:
                    return steps
                if tuple(combo) in existing:
                    continue
                selected, invalid = sanitize_selection(obs, list(combo))
                if invalid or tuple(selected) != tuple(combo):
                    continue
                selected_actions = [actions_by_index[index] for index in selected if index in actions_by_index]
                if len(selected_actions) != len(selected):
                    continue
                step_features = [
                    extract_action_features(observed.state, action, self.config, opponent_intent=observed.belief.opponent.intent)
                    for action in selected_actions
                ]
                steps.append(
                    SelectionStep(
                        selected_indices=tuple(selected),
                        actions=tuple(selected_actions),
                        select_type=observed.state.select_type,
                        context=observed.state.select_context,
                        features=combine_feature_dicts(step_features),
                        score=0.0,
                        labels=tuple(action.label for action in selected_actions),
                    )
                )
        return steps

    def _root_coverage_step(self, observed: ObservedContext, action_index: int) -> SelectionStep | None:
        obs = observed.observation
        if obs.select is None:
            return None
        actions_by_index = {action.index: action for action in observed.actions}
        if action_index not in actions_by_index:
            return None
        min_count = max(0, int(obs.select.minCount or 0))
        max_count = min(int(obs.select.maxCount or 0), len(observed.actions))
        if max_count <= 0:
            return None
        selected = [int(action_index)]
        for action in observed.actions:
            if len(selected) >= min_count:
                break
            if action.index not in selected:
                selected.append(action.index)
        selected = selected[:max_count]
        clean, invalid = sanitize_selection(obs, selected)
        if invalid or int(action_index) not in clean:
            return None
        selected_actions = [actions_by_index[index] for index in clean if index in actions_by_index]
        if len(selected_actions) != len(clean):
            return None
        step_features = [
            extract_action_features(observed.state, action, self.config, opponent_intent=observed.belief.opponent.intent)
            for action in selected_actions
        ]
        return SelectionStep(
            selected_indices=tuple(clean),
            actions=tuple(selected_actions),
            select_type=observed.state.select_type,
            context=observed.state.select_context,
            features=combine_feature_dicts(step_features),
            score=0.0,
            labels=tuple(action.label for action in selected_actions),
        )

    @staticmethod
    def _root_coverage_sequence(observed: ObservedContext, step: SelectionStep) -> CandidateSequence:
        return CandidateSequence(
            steps=(step,),
            observation=observed.observation,
            state=observed.state,
            planner_score=float(step.score),
            terminal=True,
            used_search=False,
            generation_tags=("root_coverage",),
            sim_result="coverage",
        )

    @staticmethod
    def _planning_coverage(
        observed: ObservedContext,
        sequences: list[CandidateSequence],
        diagnostics: dict[str, Any] | None = None,
    ) -> PlanningCoverage:
        diagnostics = diagnostics or {}
        legal = {int(action.index) for action in observed.actions}
        represented = {
            int(index)
            for sequence in sequences
            for index in sequence.first_selection
            if isinstance(index, int)
        }
        missing = tuple((index,) for index in sorted(legal - represented))
        max_depth = max((len(sequence.steps) for sequence in sequences), default=0)
        tags = tuple(sorted({label for sequence in sequences for step in sequence.steps for label in step.labels}))
        return PlanningCoverage(
            num_legal_first_actions=len(legal),
            num_first_actions_represented=len(represented & legal),
            missing_first_actions=missing,
            num_candidates=len(sequences),
            max_depth_reached=max_depth,
            node_budget_hit=bool(diagnostics.get("node_budget_hit", False)),
            time_budget_hit=bool(diagnostics.get("time_budget_hit", False)),
            candidate_tags=tags,
        )
