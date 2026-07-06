from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import legal_actions, sanitize_selection
from .belief import BeliefBuilder
from .evaluators import ResourceEvaluator, ThreatEvaluator
from .game_plan import GamePlanManager
from .knowledge import CardKnowledgeStore
from .mlp_ranker import MLPSequenceRanker
from .policy import LinearQPolicy
from .sequence import (
    ActionSelector,
    CandidateSequence,
    FeasibleSequenceSimulator,
    SequenceFeaturizer,
    SequenceScorer,
    SequenceSearchConfig,
    ValueModel,
)
from .state import build_agent_state


@dataclass(frozen=True)
class DecisionTrace:
    sequence_count: int
    used_search: bool
    chosen_score: float
    chosen_labels: tuple[str, ...]


class DecisionEngine:
    """Sequence-first decision stack with a linear-policy fallback."""

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
        seed: int | None = None,
    ):
        self.deck = [int(card_id) for card_id in deck]
        self.config = config
        self.search_config = search_config or SequenceSearchConfig()
        self.fallback = LinearQPolicy(weights=weights, config=config)
        self.beliefs = BeliefBuilder(config=config, decklist=self.deck)
        self.plans = GamePlanManager(config)
        self.resources = ResourceEvaluator(config)
        self.threats = ThreatEvaluator(config)
        self.knowledge = CardKnowledgeStore(config)
        self.simulator = FeasibleSequenceSimulator(config=config, weights=weights)
        self.featurizer = SequenceFeaturizer(config=config, knowledge=self.knowledge)
        self.scorer = SequenceScorer(
            weights=weights,
            mlp_ranker=mlp_ranker,
            linear_weight=linear_score_weight,
            mlp_weight=mlp_score_weight,
        )
        self.value_model = ValueModel()
        self.selector = ActionSelector(seed=seed)
        self.last_trace: DecisionTrace | None = None
        self.last_chosen_sequence: CandidateSequence | None = None

    def choose(self, obs: Any, mode: str = "eval") -> list[int]:
        if obs.select is None:
            self.last_chosen_sequence = None
            return []
        try:
            return self._choose_sequence(obs, mode=mode)
        except Exception:
            return self._fallback_choose(obs)

    def score_sequences(self, obs: Any) -> list[CandidateSequence]:
        actions = legal_actions(obs)
        if obs.current is None or not actions:
            return []
        root_state = build_agent_state(obs, actions=actions)
        belief = self.beliefs.build(obs, root_state)
        plan = self.plans.build(root_state)
        resources = self.resources.evaluate(root_state)
        threat = self.threats.evaluate(root_state, belief.opponent)
        sequences = self.simulator.generate(obs, root_state, belief, self.search_config)
        scored: list[CandidateSequence] = []
        for sequence in sequences:
            sequence.features = self.featurizer.featurize(root_state, sequence, belief, plan, resources, threat)
            sequence.features["value_model_score"] = self.value_model.evaluate(root_state, sequence.state)
            sequence.score = self.scorer.score(sequence.features)
            scored.append(sequence)
        return scored

    def _choose_sequence(self, obs: Any, mode: str) -> list[int]:
        sequences = self.score_sequences(obs)
        if not sequences:
            return self._fallback_choose(obs)

        chosen = self.selector.select(sequences, mode=mode)
        if chosen is None:
            return self._fallback_choose(obs)
        fallback_selected, _ = sanitize_selection(obs, self.fallback.choose(obs))
        if mode == "eval" and self._fallback_is_safer(chosen, sequences, fallback_selected):
            self.last_chosen_sequence = None
            self.last_trace = DecisionTrace(
                sequence_count=len(sequences),
                used_search=any(sequence.used_search for sequence in sequences),
                chosen_score=0.0,
                chosen_labels=("fallback_guardrail",),
            )
            return fallback_selected
        selected, invalid = sanitize_selection(obs, chosen.first_selection)
        if invalid:
            return self._fallback_choose(obs)
        self.last_chosen_sequence = chosen
        self.last_trace = DecisionTrace(
            sequence_count=len(sequences),
            used_search=any(sequence.used_search for sequence in sequences),
            chosen_score=chosen.score,
            chosen_labels=tuple(label for step in chosen.steps for label in step.labels),
        )
        return selected

    @staticmethod
    def _fallback_is_safer(chosen: CandidateSequence, sequences: list[CandidateSequence], fallback_selected: list[int]) -> bool:
        if not fallback_selected or chosen.first_selection == fallback_selected:
            return False
        tactical_payoff = (
            chosen.features.get("seq_terminal_win", 0.0) > 0
            or chosen.features.get("attack_wins_game", 0.0) > 0
            or chosen.features.get("seq_my_prizes_gained", 0.0) > 0
            or chosen.features.get("attack_takes_ko", 0.0) > 0
            or chosen.features.get("value_model_score", 0.0) >= 0.35
        )
        if tactical_payoff:
            return False
        fallback_key = tuple(fallback_selected)
        fallback_sequence = next((sequence for sequence in sequences if tuple(sequence.first_selection) == fallback_key), None)
        if fallback_sequence is None:
            return chosen.score < 250.0
        return chosen.score < fallback_sequence.score + 250.0

    def _fallback_choose(self, obs: Any) -> list[int]:
        selected, _invalid = sanitize_selection(obs, self.fallback.choose(obs))
        self.last_chosen_sequence = None
        self.last_trace = DecisionTrace(
            sequence_count=0,
            used_search=False,
            chosen_score=0.0,
            chosen_labels=("fallback_linear_policy",),
        )
        return selected
