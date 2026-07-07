from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .actions import ActionView, legal_actions
from .cg_loader import PROJECT_ROOT
from .config import load_config, read_deck
from .deck_db import deck_counts, deck_heuristic_config, get_deck, load_deck_db, top_observed_winrate_decks
from .features import default_weights, extract_action_features, score_features
from .mlp_ranker import MLPSequenceRanker
from .state import build_agent_state


SEQUENCE_STARMIE_FROSLASS_WEIGHTS = PROJECT_ROOT / "artifacts" / "versions" / "sequence_starmie_froslass" / "weights.json"


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


class DeckSpecificHeuristicPolicy(LinearQPolicy):
    """Heuristic policy whose role config is derived from the exact deck it pilots."""

    def __init__(
        self,
        *,
        deck: list[int] | None = None,
        deck_id: str | None = None,
        config: dict | None = None,
        top_rank: int = 1,
    ):
        db = load_deck_db()
        deck_record: dict[str, Any] | None = None
        if deck_id:
            deck_record = get_deck(deck_id, db)
        elif deck is not None:
            wanted = deck_counts({"counts": [{"id": card_id, "count": count} for card_id, count in _counts(deck).items()]})
            for candidate in db.get("decks", []):
                if deck_counts(candidate) == wanted:
                    deck_record = candidate
                    break
        if deck_record is None:
            ranked = top_observed_winrate_decks(db, min_observed_wins=25)
            if not ranked:
                raise ValueError("No top observed-winrate decks available for deck-specific heuristic policy")
            deck_record = ranked[max(0, min(len(ranked) - 1, top_rank - 1))]
        role_config = deck_heuristic_config(deck_record, db)
        if config:
            merged = dict(config)
            merged.update(role_config)
            role_config = merged
        weights = default_weights()
        weights.update(
            {
                "attack_wins_game": 30000.0,
                "attack_takes_ko": 1800.0,
                "attack_prize_value": 700.0,
                "attack_with_main": 420.0,
                "attack_with_backup": 180.0,
                "attack_before_setup": -450.0,
                "attack_with_empty_board_risk": -1800.0,
                "end_with_empty_board_risk": -2600.0,
                "play_pokemon": 80.0,
                "play_pokemon_to_avoid_empty_board_loss": 2600.0,
                "play_pokemon_behind_low_active": 1500.0,
                "play_pokemon_to_stabilize_board": 520.0,
                "play_setup_basic": 650.0,
                "play_main_line": 520.0,
                "play_main_line_after_setup": -220.0,
                "evolve_key": 850.0,
                "evolve_main_attacker": 950.0,
                "attach_energy": 520.0,
                "attach_to_main_line": 360.0,
                "attach_to_backup": 120.0,
                "attach_energy_to_low_value": -650.0,
                "play_search": 480.0,
                "play_draw": 360.0,
                "play_gust": 260.0,
                "play_gust_without_ready_attacker": -900.0,
                "play_gust_to_deny_evolution": 900.0,
                "select_main_line": 520.0,
                "select_backup": 220.0,
                "select_energy": 300.0,
                "select_search_card": 260.0,
                "select_draw_card": 220.0,
                "end_with_useful_action": -1600.0,
                "ready_attackers": 650.0,
                "main_attacker_ready": 900.0,
                "main_line_in_play": 520.0,
            }
        )
        self.deck_record = deck_record
        super().__init__(weights=weights, config=role_config)


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


def _counts(deck: list[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for card_id in deck:
        counts[int(card_id)] = counts.get(int(card_id), 0) + 1
    return counts


class SequencePolicy(Policy):
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        config: dict | None = None,
        deck: list[int] | None = None,
        mlp_ranker: MLPSequenceRanker | None = None,
        mlp_model_path: str | Path | None = None,
        linear_score_weight: float | None = None,
        mlp_score_weight: float | None = None,
        clean_evaluator: bool = False,
        probability_features_enabled: bool = True,
        mlp_enabled: bool = True,
        value_proxy_enabled: bool = True,
        fallback_guardrail_enabled: bool = True,
        engine_exception_fallback_enabled: bool = False,
        num_hidden_samples: int = 1,
        seed: int | None = None,
    ):
        from .decision_engine import DecisionEngine
        from .sequence import SequenceSearchConfig

        self.config = config or load_config()
        self.deck = deck or read_deck(config=self.config)
        if mlp_ranker is None and mlp_model_path is not None:
            mlp_ranker = MLPSequenceRanker.from_file(mlp_model_path)
        if linear_score_weight is None:
            linear_score_weight = 0.0 if mlp_ranker is not None and mlp_enabled else 1.0
        if mlp_score_weight is None:
            mlp_score_weight = 1.0
        self.engine = DecisionEngine(
            deck=self.deck,
            config=self.config,
            weights=weights,
            mlp_ranker=mlp_ranker,
            linear_score_weight=linear_score_weight,
            mlp_score_weight=mlp_score_weight,
            clean_evaluator=clean_evaluator,
            probability_features_enabled=probability_features_enabled,
            mlp_enabled=mlp_enabled,
            value_proxy_enabled=value_proxy_enabled,
            fallback_guardrail_enabled=fallback_guardrail_enabled,
            engine_exception_fallback_enabled=engine_exception_fallback_enabled,
            search_config=SequenceSearchConfig(
                beam_width=8,
                max_depth=8,
                max_nodes=128,
                time_budget_ms=500,
                num_hidden_samples=max(1, int(num_hidden_samples)),
            ),
            seed=seed,
        )

    def choose(self, obs: Any) -> list[int]:
        return self.engine.choose(obs)


def load_weights(path: str | Path) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    weights = payload.get("weights", payload)
    return {str(k): float(v) for k, v in weights.items()}


def make_policy(
    spec: str,
    config: dict | None = None,
    seed: int | None = None,
    deck: list[int] | None = None,
    sequence_options: dict[str, Any] | None = None,
) -> Policy:
    sequence_options = sequence_options or {}
    if spec == "random":
        return RandomPolicy(seed=seed)
    if spec == "heuristic":
        return HeuristicPolicy(config=config)
    if spec == "deck-heuristic":
        return DeckSpecificHeuristicPolicy(deck=deck, config=config)
    if spec.startswith("deck-heuristic:"):
        payload = spec.split(":", 1)[1]
        if payload.startswith("top"):
            rank = 1
            if payload != "top":
                try:
                    rank = int(payload.removeprefix("top"))
                except ValueError:
                    rank = 1
            return DeckSpecificHeuristicPolicy(deck=deck, config=config, top_rank=rank)
        return DeckSpecificHeuristicPolicy(deck_id=payload, deck=deck, config=config)
    if spec == "sequence":
        cfg = config or load_config()
        return SequencePolicy(config=cfg, deck=deck or read_deck(config=cfg), seed=seed, **sequence_options)
    if spec == "sequence-starmie-froslass":
        if not SEQUENCE_STARMIE_FROSLASS_WEIGHTS.exists():
            raise FileNotFoundError(
                f"Missing sequence_starmie_froslass baseline weights: {SEQUENCE_STARMIE_FROSLASS_WEIGHTS}"
            )
        cfg = config or load_config()
        return SequencePolicy(
            weights=load_weights(SEQUENCE_STARMIE_FROSLASS_WEIGHTS),
            config=cfg,
            deck=deck or read_deck(config=cfg),
            seed=seed,
            **sequence_options,
        )
    if spec.startswith("sequence:"):
        cfg = config or load_config()
        return SequencePolicy(weights=load_weights(spec.split(":", 1)[1]), config=cfg, deck=deck or read_deck(config=cfg), seed=seed, **sequence_options)
    if spec.startswith("sequence-mlp:"):
        cfg = config or load_config()
        payload = spec.split(":", 1)[1]
        if "," in payload:
            mlp_path, weights_path = payload.split(",", 1)
            return SequencePolicy(
                weights=load_weights(weights_path),
                mlp_model_path=mlp_path,
                config=cfg,
                deck=deck or read_deck(config=cfg),
                seed=seed,
                **sequence_options,
            )
        return SequencePolicy(mlp_model_path=payload, config=cfg, deck=deck or read_deck(config=cfg), seed=seed, **sequence_options)
    if spec == "sequence-clean":
        cfg = config or load_config()
        return SequencePolicy(
            config=cfg,
            deck=deck or read_deck(config=cfg),
            clean_evaluator=True,
            linear_score_weight=1.0,
            mlp_score_weight=1.0,
            seed=seed,
            **sequence_options,
        )
    if spec.startswith("sequence-clean:"):
        cfg = config or load_config()
        return SequencePolicy(
            weights=load_weights(spec.split(":", 1)[1]),
            config=cfg,
            deck=deck or read_deck(config=cfg),
            clean_evaluator=True,
            linear_score_weight=1.0,
            mlp_score_weight=1.0,
            seed=seed,
            **sequence_options,
        )
    if spec.startswith("sequence-clean-mlp:"):
        cfg = config or load_config()
        payload = spec.split(":", 1)[1]
        if "," in payload:
            mlp_path, weights_path = payload.split(",", 1)
            return SequencePolicy(
                weights=load_weights(weights_path),
                mlp_model_path=mlp_path,
                config=cfg,
                deck=deck or read_deck(config=cfg),
                clean_evaluator=True,
                linear_score_weight=0.0,
                mlp_score_weight=1.0,
                seed=seed,
                **sequence_options,
            )
        return SequencePolicy(
            mlp_model_path=payload,
            config=cfg,
            deck=deck or read_deck(config=cfg),
            clean_evaluator=True,
            linear_score_weight=0.0,
            mlp_score_weight=1.0,
            seed=seed,
            **sequence_options,
        )
    if spec.startswith("linear:"):
        return LinearQPolicy.from_file(spec.split(":", 1)[1], config=config)
    if spec.endswith(".json"):
        return LinearQPolicy.from_file(spec, config=config)
    raise ValueError(f"Unknown policy spec: {spec}")
