from __future__ import annotations

from typing import Any

from ..actions import legal_actions
from ..belief import BeliefBuilder
from ..decision_types import ObservedContext
from ..game_plan import GamePlanManager
from ..opponent_memory import OpponentMemory
from .snapshots import ResourceEvaluator, ThreatEvaluator
from ..state import build_agent_state


class ObservationBuilder:
    """Build the complete observation-stage context for a decision."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        decklist: list[int],
        opponent_memory: OpponentMemory | None = None,
        num_hidden_samples: int = 1,
    ):
        self.opponent_memory = opponent_memory or OpponentMemory()
        self.beliefs = BeliefBuilder(config=config, decklist=decklist, num_hidden_samples=num_hidden_samples)
        self.plans = GamePlanManager(config)
        self.resources = ResourceEvaluator(config)
        self.threats = ThreatEvaluator(config)

    def build(self, obs: Any) -> ObservedContext:
        actions = legal_actions(obs)
        if obs.current is None or not actions:
            raise ValueError("Cannot observe a decision state without current state and legal actions")
        state = build_agent_state(obs, actions=actions)
        self.opponent_memory.update(obs, state)
        belief = self.beliefs.build(obs, state, memory=self.opponent_memory)
        return ObservedContext(
            observation=obs,
            actions=tuple(actions),
            state=state,
            opponent_memory=self.opponent_memory,
            belief=belief,
            probabilities=belief.probabilities,
            game_plan=self.plans.build(state),
            resources=self.resources.evaluate(state),
            threat=self.threats.evaluate(state, belief.opponent),
            diagnostics={
                "legal_action_count": len(actions),
                "probability_features": len(belief.probabilities.to_features()),
                "opponent_archetype": belief.opponent.top_archetype,
            },
        )
