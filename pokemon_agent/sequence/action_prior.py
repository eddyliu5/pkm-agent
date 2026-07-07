from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..actions import ActionView
from ..features import extract_action_features
from ..opponent_memory import OpponentIntentSnapshot
from ..state import AgentState


@dataclass(frozen=True)
class PriorAction:
    action: ActionView
    score: float
    features: dict[str, float]
    bucket: str


class ExpansionPrior:
    """Cheap planner ordering signal, independent from evaluator weights."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def score_actions(
        self,
        state: AgentState,
        actions: list[ActionView],
        opponent_intent: OpponentIntentSnapshot | None = None,
    ) -> list[PriorAction]:
        scored: list[PriorAction] = []
        for action in actions:
            features = extract_action_features(state, action, self.config, opponent_intent=opponent_intent)
            scored.append(PriorAction(action=action, score=self._priority(action, features), features=features, bucket=self.bucket(action, features)))
        scored.sort(key=lambda item: (item.score, -item.action.index), reverse=True)
        return scored

    @staticmethod
    def bucket(action: ActionView, features: dict[str, float]) -> str:
        if action.is_attack:
            return "attack"
        if action.is_end:
            return "end"
        if features.get("play_search") or features.get("select_search_card"):
            return "search"
        if features.get("play_draw") or features.get("select_draw_card"):
            return "draw"
        if features.get("attach_energy"):
            return "attach"
        if features.get("evolve"):
            return "evolve"
        if features.get("retreat"):
            return "retreat"
        if features.get("play_gust"):
            return "gust"
        if features.get("ability"):
            return "ability"
        return "other"

    @staticmethod
    def _priority(action: ActionView, features: dict[str, float]) -> float:
        score = 0.0
        if features.get("attack_wins_game"):
            score += 1000.0
        if features.get("attack_takes_ko"):
            score += 350.0
        if action.is_attack:
            score += 120.0
        if features.get("evolve_key") or features.get("evolve_main_attacker"):
            score += 90.0
        if features.get("attach_energy"):
            score += 70.0
        if features.get("play_search") or features.get("select_search_card"):
            score += 55.0
        if features.get("play_draw") or features.get("select_draw_card"):
            score += 45.0
        if features.get("play_gust"):
            score += 35.0
        if features.get("ability"):
            score += 30.0
        if features.get("retreat"):
            score += 20.0
        if action.is_end:
            score -= 10.0
        return score
