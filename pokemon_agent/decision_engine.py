from __future__ import annotations

from typing import Any

from .actions import sanitize_selection
from .acting import ActionSelector, FallbackGuardrail, FallbackLinearPolicy
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


class DecisionEngine:
    """Sequence-first observe -> plan -> evaluate -> act orchestrator."""

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
        seed: int | None = None,
    ):
        self.deck = [int(card_id) for card_id in deck]
        self.config = config
        self.search_config = search_config or SequenceSearchConfig()
        self.fallback = FallbackLinearPolicy(weights=weights, config=config)
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
        self.feature_schema_version = "sequence_v3_clean" if self.clean_evaluator else "sequence_v3_full"
        if not mlp_enabled:
            mlp_ranker = None
        if mlp_ranker is not None:
            mlp_ranker.validate_feature_schema(self.feature_schema_version, allow_legacy=True)
        self.featurizer = (
            CleanSequenceFeaturizer()
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
        self.last_planning_coverage: PlanningCoverage | None = None

    def choose(self, obs: Any, mode: str = "eval") -> list[int]:
        if obs.select is None:
            self.last_chosen_sequence = None
            return []
        try:
            decision = self._choose_sequence(obs, mode=mode)
            self.last_chosen_sequence = decision.chosen_sequence
            self.last_trace = decision.trace
            return decision.selected
        except Exception:
            return self._fallback_choose(obs)

    def score_sequences(self, obs: Any) -> list[CandidateSequence]:
        if obs.current is None:
            return []
        try:
            return self.evaluate(self.plan(self.observe(obs))).sequences
        except Exception:
            return []

    def observe(self, obs: Any) -> ObservedContext:
        return self.observation_builder.build(obs)

    def plan(self, observed: ObservedContext) -> PlanningResult:
        sequences = self.simulator.generate(
            observed.observation,
            observed.state,
            observed.belief,
            self.search_config,
        )
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
        planning = self.plan(observed)
        evaluated = self.evaluate(observed, planning)
        return self.act(observed, planning, evaluated, mode=mode)

    def _fallback_choose(self, obs: Any) -> list[int]:
        decision = self._fallback_decision(obs)
        self.last_chosen_sequence = decision.chosen_sequence
        self.last_trace = decision.trace
        return decision.selected

    def _fallback_decision(self, obs: Any) -> AgentDecision:
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
