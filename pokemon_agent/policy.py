from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .actions import ActionView, legal_actions
from .config import load_config, read_deck
from .features import default_weights, extract_action_features, score_features
from .state import build_agent_state


class Policy:
    def choose(self, obs: Any) -> list[int]:
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose(self, obs: Any) -> list[int]:
        if obs.select is None:
            return []
        indices = list(range(len(obs.select.option)))
        count = min(obs.select.maxCount, len(indices))
        if count <= 0:
            return []
        return self.rng.sample(indices, count)


class LinearQPolicy(Policy):
    def __init__(self, weights: dict[str, float] | None = None, config: dict | None = None):
        self.weights = dict(default_weights())
        if weights:
            self.weights.update({str(k): float(v) for k, v in weights.items()})
        self.config = config or load_config()

    @classmethod
    def from_file(cls, path: str | Path, config: dict | None = None) -> "LinearQPolicy":
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        weights = payload.get("weights", payload)
        return cls(weights=weights, config=config)

    def action_scores(self, obs: Any) -> list[tuple[ActionView, float, dict[str, float]]]:
        actions = legal_actions(obs)
        if obs.current is None:
            return [(action, 0.0, {}) for action in actions]
        state = build_agent_state(obs, actions=actions)
        scored = []
        for action in actions:
            features = extract_action_features(state, action, self.config)
            scored.append((action, score_features(features, self.weights), features))
        return scored

    def choose(self, obs: Any) -> list[int]:
        if obs.select is None:
            return []
        scored = self.action_scores(obs)
        scored.sort(key=lambda item: (item[1], -item[0].index), reverse=True)
        count = min(obs.select.maxCount, len(scored))
        if count <= 0:
            return []
        return [action.index for action, _, _ in scored[:count]]


class HeuristicPolicy(LinearQPolicy):
    def __init__(self, config: dict | None = None):
        super().__init__(weights=default_weights(), config=config)


class EpsilonGreedyPolicy(Policy):
    def __init__(self, policy: Policy, epsilon: float, seed: int | None = None):
        self.policy = policy
        self.epsilon = float(epsilon)
        self.random_policy = RandomPolicy(seed=seed)
        self.rng = random.Random(seed)

    def choose(self, obs: Any) -> list[int]:
        if self.rng.random() < self.epsilon:
            return self.random_policy.choose(obs)
        return self.policy.choose(obs)


class SequencePolicy(Policy):
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        config: dict | None = None,
        deck: list[int] | None = None,
        seed: int | None = None,
    ):
        from .decision_engine import DecisionEngine
        from .sequence import SequenceSearchConfig

        self.config = config or load_config()
        self.deck = deck or read_deck(config=self.config)
        self.engine = DecisionEngine(
            deck=self.deck,
            config=self.config,
            weights=weights,
            search_config=SequenceSearchConfig(beam_width=8, max_depth=8, max_nodes=128, time_budget_ms=500),
            seed=seed,
        )

    def choose(self, obs: Any) -> list[int]:
        return self.engine.choose(obs)


def load_weights(path: str | Path) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    weights = payload.get("weights", payload)
    return {str(k): float(v) for k, v in weights.items()}


def make_policy(spec: str, config: dict | None = None, seed: int | None = None) -> Policy:
    if spec == "random":
        return RandomPolicy(seed=seed)
    if spec == "heuristic":
        return HeuristicPolicy(config=config)
    if spec == "sequence":
        cfg = config or load_config()
        return SequencePolicy(config=cfg, deck=read_deck(config=cfg), seed=seed)
    if spec.startswith("sequence:"):
        cfg = config or load_config()
        return SequencePolicy(weights=load_weights(spec.split(":", 1)[1]), config=cfg, deck=read_deck(config=cfg), seed=seed)
    if spec.startswith("linear:"):
        return LinearQPolicy.from_file(spec.split(":", 1)[1], config=config)
    if spec.endswith(".json"):
        return LinearQPolicy.from_file(spec, config=config)
    raise ValueError(f"Unknown policy spec: {spec}")
