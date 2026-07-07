from __future__ import annotations

from typing import Any

from ..actions import ActionView, legal_actions
from ..features import combine_feature_dicts
from ..opponent_memory import OpponentIntentSnapshot
from ..sequence.action_prior import ExpansionPrior, PriorAction
from ..sequence.diversifier import CandidateDiversifier
from ..sequence.types import SelectionStep
from ..state import AgentState


class LegalityFilter:
    """Remove duplicate or structurally impossible actions before expansion."""

    def filter_actions(self, actions: list[ActionView]) -> list[ActionView]:
        seen: set[int] = set()
        kept: list[ActionView] = []
        for action in actions:
            if action.index in seen:
                continue
            seen.add(action.index)
            kept.append(action)
        return kept


class SequencePruner:
    """Compatibility facade for planner expansion components."""

    def __init__(self, config: dict[str, Any], weights: dict[str, float] | None = None):
        self.legality = LegalityFilter()
        self.prior = ExpansionPrior(config)
        self.diversifier = CandidateDiversifier()
        self.last_missing_action_indices: tuple[int, ...] = ()
        self.last_pruned_actions: tuple[str, ...] = ()

    def selection_candidates(
        self,
        obs: Any,
        state: AgentState,
        search_config: Any,
        opponent_intent: OpponentIntentSnapshot | None = None,
    ) -> list[Any]:
        if obs.select is None:
            return []
        actions = self.legality.filter_actions(legal_actions(obs))
        if not actions:
            return []
        scored_actions = self.prior.score_actions(state, actions, opponent_intent=opponent_intent)
        kept = self.diversifier.keep_action_pool(scored_actions, option_pool_size=search_config.option_pool_size)
        kept_indices = {item.action.index for item in kept}
        self.last_missing_action_indices = tuple(action.index for action in actions if action.index not in kept_indices)
        self.last_pruned_actions = tuple(f"budget:{index}" for index in self.last_missing_action_indices)
        by_index = {item.action.index: item for item in kept}
        min_count = max(0, int(obs.select.minCount or 0))
        max_count = min(int(obs.select.maxCount or 0), len(actions))
        counts = self.diversifier.selection_counts(min_count, max_count)
        indices = [item.action.index for item in kept]
        steps = [
            self._step_from_items(obs, state, list(combo), by_index)
            for combo in self.diversifier.combinations(indices, counts, max_combinations=search_config.max_combinations_per_node)
        ]
        steps.sort(key=lambda step: (step.score, -len(step.selected_indices)), reverse=True)
        return steps

    @staticmethod
    def _step_from_items(
        obs: Any,
        state: AgentState,
        selected: list[int],
        by_index: dict[int, PriorAction],
    ) -> SelectionStep:
        selected_actions = [by_index[index].action for index in selected if index in by_index]
        features = combine_feature_dicts([by_index[index].features for index in selected if index in by_index])
        score = sum(by_index[index].score for index in selected if index in by_index)
        return SelectionStep(
            selected_indices=tuple(selected),
            actions=tuple(selected_actions),
            select_type=state.select_type,
            context=state.select_context,
            features=features,
            score=score,
            labels=tuple(action.label for action in selected_actions),
        )
