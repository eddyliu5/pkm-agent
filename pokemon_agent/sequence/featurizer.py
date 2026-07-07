from __future__ import annotations

import math
import re

from ..belief import BeliefState
from ..features import combine_feature_dicts
from ..game_plan import PlanContext
from ..knowledge import CardKnowledgeStore
from ..observation.snapshots import ResourceSnapshot, ThreatSnapshot
from ..state import AgentState, PokemonView
from .types import CandidateSequence


def _feature_token(value: object) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return token or "unknown"


def _self_deck_features(config: dict | None) -> dict[str, float]:
    if not config:
        return {}
    features: dict[str, float] = {}
    archetype = config.get("archetype")
    deck_plan_type = config.get("deck_plan_type")
    if archetype:
        features[f"self_archetype_{_feature_token(archetype)}"] = 1.0
    if deck_plan_type:
        features[f"self_plan_type_{_feature_token(deck_plan_type)}"] = 1.0
    if config.get("replay_deck_matched_db"):
        features["self_deck_matched_db"] = 1.0
    return features


class SequenceFeaturizer:
    def __init__(self, config: dict, knowledge: CardKnowledgeStore, *, include_probability: bool = True):
        self.config = config
        self.knowledge = knowledge
        self.include_probability = bool(include_probability)

    def featurize(
        self,
        root_state: AgentState,
        sequence: CandidateSequence,
        belief: BeliefState,
        plan: PlanContext,
        resources: ResourceSnapshot,
        threat: ThreatSnapshot,
    ) -> dict[str, float]:
        features = combine_feature_dicts([step.features for step in sequence.steps])
        sample_tags = [tag for tag in sequence.generation_tags if tag.startswith("sample_")]
        features["hidden_sample_support_count"] = float(len(set(sample_tags)))
        features["hidden_sample_support"] = min(1.0, float(len(set(sample_tags))) / 4.0)
        features["seq_length"] = min(1.0, float(len(sequence.steps)) / 8.0)
        features["seq_uses_attack"] = 1.0 if any(action.is_attack for step in sequence.steps for action in step.actions) else 0.0
        features["seq_uses_end"] = 1.0 if any(action.is_end for step in sequence.steps for action in step.actions) else 0.0
        features.update(_self_deck_features(self.config))
        features[f"plan_{plan.deck_plan_type}"] = 1.0
        features[f"phase_{plan.phase}"] = 1.0
        features[f"turn_goal_{plan.turn_plan.primary_goal}"] = 1.0
        for goal in plan.turn_plan.secondary_goals:
            features[f"turn_secondary_{goal}"] = 1.0
        for threat_name in plan.turn_plan.must_answer_threats:
            features[f"turn_threat_{threat_name}"] = 1.0
        for resource_name in plan.turn_plan.desired_resources:
            features[f"turn_resource_need_{resource_name}"] = 1.0
        for pattern in plan.turn_plan.avoid_patterns:
            features[f"turn_avoid_{pattern}"] = 1.0
        for key, value in plan.priority_weights.items():
            features[f"plan_priority_{key}"] = float(value)
        features["resource_ready_attackers"] = min(1.0, float(resources.ready_attackers) / 3.0)
        features["resource_hand_quality"] = resources.hand_quality
        features["resource_energy_in_play"] = min(1.0, float(resources.energy_in_play) / 8.0)
        features["resource_bench_space"] = min(1.0, float(resources.bench_space) / 5.0)
        features["threat_can_ko_active"] = threat.opponent_can_ko_active
        features["threat_can_win_next_turn"] = threat.opponent_can_win_next_turn
        features["opponent_gust_risk"] = belief.opponent.prob_has_gust
        features["opponent_switch_risk"] = belief.opponent.prob_has_switch
        features["opponent_recent_pokemon_search"] = belief.opponent.intent.recent_pokemon_search
        features["opponent_recent_evolution_search"] = belief.opponent.intent.recent_evolution_search
        features["opponent_recent_gust_seen"] = belief.opponent.intent.recent_gust_seen
        features["opponent_recent_switch_seen"] = belief.opponent.intent.recent_switch_seen
        features["opponent_likely_setting_up_attacker"] = belief.opponent.intent.likely_setting_up_attacker
        features["opponent_evolution_window_open"] = belief.opponent.intent.evolution_window_open
        features["opponent_has_setup_targets"] = 1.0 if belief.opponent.intent.has_setup_targets else 0.0
        features["opponent_has_must_answer_targets"] = 1.0 if belief.opponent.intent.has_must_answer_targets else 0.0
        if belief.opponent.plan_type:
            features[f"opponent_{belief.opponent.plan_type}_plan"] = 1.0
        features["opponent_expected_first_attack_turn"] = min(1.0, belief.opponent.expected_first_attack_turn / 6.0)
        features["opponent_expected_game_end_turn"] = min(1.0, belief.opponent.expected_game_end_turn / 14.0)
        features["prize_known_key_prized"] = min(1.0, float(sum(belief.prize_info.known_prized.values())) / 6.0)
        features["prize_unknown_count"] = min(1.0, float(belief.prize_info.unknown_prize_count) / 6.0)
        if self.include_probability:
            features.update(belief.probabilities.to_features())

        if sequence.state is not None:
            features["seq_my_prizes_gained"] = float(root_state.my_prizes_remaining - sequence.state.my_prizes_remaining)
            features["seq_opp_prizes_gained"] = float(root_state.opp_prizes_remaining - sequence.state.opp_prizes_remaining)
            features["seq_energy_delta"] = max(-1.0, min(1.0, float(sequence.state.my_energy_in_play - root_state.my_energy_in_play) / 4.0))
            features["seq_bench_delta"] = max(-1.0, min(1.0, float(len(sequence.state.my_bench) - len(root_state.my_bench)) / 5.0))
            if sequence.state.result == root_state.player_index:
                features["seq_terminal_win"] = 1.0
            elif sequence.state.result not in (-1, 2, root_state.player_index):
                features["seq_terminal_loss"] = 1.0

        for index, value in enumerate(self._pooled_card_embedding(root_state, sequence)):
            features[f"card_embed_{index}"] = value
        for index, value in enumerate(self._field_attention_embedding(root_state, sequence)):
            features[f"field_card_attn_{index}"] = value
        return features

    def _pooled_card_embedding(self, root_state: AgentState, sequence: CandidateSequence) -> tuple[float, ...]:
        return self._mean_vector(self._action_card_vectors(root_state, sequence))

    def _action_card_vectors(self, root_state: AgentState, sequence: CandidateSequence) -> list[tuple[float, ...]]:
        vectors: list[tuple[float, ...]] = []
        for step in sequence.steps:
            for action in step.actions:
                if action.source_card_id is not None:
                    vectors.append(self.knowledge.vector(action.source_card_id))
                elif action.is_attack and root_state.my_active is not None:
                    vectors.append(self.knowledge.vector(root_state.my_active.id))
                if action.target_card_id is not None:
                    vectors.append(self.knowledge.vector(action.target_card_id))
        return vectors

    def _field_attention_embedding(self, root_state: AgentState, sequence: CandidateSequence) -> tuple[float, ...]:
        query = self._mean_vector(self._action_card_vectors(root_state, sequence))
        field_vectors = [self.knowledge.vector(card_id) for card_id in self._visible_in_play_card_ids(root_state)]
        if not field_vectors or not any(query):
            return self._zero_vector()
        scale = math.sqrt(float(len(query))) or 1.0
        scores = [sum(q * k for q, k in zip(query, vector)) / scale for vector in field_vectors]
        best = max(scores)
        weights = [math.exp(max(-40.0, min(40.0, score - best))) for score in scores]
        total = sum(weights) or 1.0
        values = [0.0] * len(field_vectors[0])
        for weight, vector in zip(weights, field_vectors):
            normalized_weight = weight / total
            for index, value in enumerate(vector):
                values[index] += normalized_weight * value
        return tuple(values)

    def _visible_in_play_card_ids(self, state: AgentState) -> tuple[int, ...]:
        ids: list[int] = []
        if state.stadium_id is not None:
            ids.append(state.stadium_id)
        for pokemon in ([state.my_active] if state.my_active else []) + list(state.my_bench):
            ids.extend(self._pokemon_in_play_card_ids(pokemon))
        for pokemon in ([state.opp_active] if state.opp_active else []) + list(state.opp_bench):
            ids.extend(self._pokemon_in_play_card_ids(pokemon))
        return tuple(ids)

    @staticmethod
    def _pokemon_in_play_card_ids(pokemon: PokemonView) -> tuple[int, ...]:
        return (pokemon.id,) + pokemon.energy_ids + pokemon.tool_ids

    def _mean_vector(self, vectors: list[tuple[float, ...]]) -> tuple[float, ...]:
        if not vectors:
            return self._zero_vector()
        dim = len(vectors[0])
        values = [0.0] * dim
        for vector in vectors:
            for index, value in enumerate(vector):
                values[index] += value
        scale = 1.0 / float(len(vectors))
        return tuple(value * scale for value in values)

    def _zero_vector(self) -> tuple[float, ...]:
        return self.knowledge.vector(None)


class CleanSequenceFeaturizer:
    """Narrow consequence-focused features for the clean runtime path."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def featurize(
        self,
        root_state: AgentState,
        sequence: CandidateSequence,
        belief: BeliefState,
        plan: PlanContext,
        resources: ResourceSnapshot,
        threat: ThreatSnapshot,
    ) -> dict[str, float]:
        features = self._compact_step_features(sequence)
        sample_tags = [tag for tag in sequence.generation_tags if tag.startswith("sample_")]
        features["hidden_sample_support_count"] = float(len(set(sample_tags)))
        features["hidden_sample_support"] = min(1.0, float(len(set(sample_tags))) / 4.0)
        features["bias"] = 1.0
        features["seq_length"] = min(1.0, float(len(sequence.steps)) / 8.0)
        features["seq_uses_attack"] = 1.0 if any(action.is_attack for step in sequence.steps for action in step.actions) else 0.0
        features["seq_uses_end"] = 1.0 if any(action.is_end for step in sequence.steps for action in step.actions) else 0.0
        features.update(_self_deck_features(self.config))
        features[f"turn_goal_{plan.turn_plan.primary_goal}"] = 1.0
        features.update(self._board_features(root_state))
        intent = belief.opponent.intent
        features["opponent_recent_pokemon_search"] = intent.recent_pokemon_search
        features["opponent_recent_evolution_search"] = intent.recent_evolution_search
        features["opponent_likely_setting_up_attacker"] = intent.likely_setting_up_attacker
        features["opponent_evolution_window_open"] = intent.evolution_window_open
        features["opponent_has_setup_targets"] = 1.0 if intent.has_setup_targets else 0.0
        features["opponent_has_must_answer_targets"] = 1.0 if intent.has_must_answer_targets else 0.0
        if belief.opponent.top_archetype:
            features[f"opponent_archetype_{belief.opponent.top_archetype}"] = 1.0
        if belief.opponent.plan_type:
            features[f"opponent_plan_{belief.opponent.plan_type}"] = 1.0
        if sequence.state is not None:
            features.update(self._consequence_features(root_state, sequence.state))
        return features

    @staticmethod
    def _compact_step_features(sequence: CandidateSequence) -> dict[str, float]:
        keep_prefixes = ("context_", "option_")
        keep_exact = {
            "simple_selection",
            "choose_yes",
            "choose_no",
            "number_value",
            "end_turn",
            "end_with_useful_action",
            "end_with_empty_board_risk",
            "attack",
            "attack_damage_frac",
            "attack_takes_ko",
            "attack_prize_value",
            "attack_wins_game",
            "attack_denies_evolution",
            "attack_removes_must_answer",
            "attack_before_setup",
            "attack_with_empty_board_risk",
            "play_pokemon",
            "play_pokemon_to_avoid_empty_board_loss",
            "play_pokemon_behind_low_active",
            "play_pokemon_to_stabilize_board",
            "play_gust",
            "play_gust_without_ready_attacker",
            "play_gust_to_deny_evolution",
            "attach_energy",
            "evolve",
            "evolve_key",
            "evolve_main_attacker",
            "ability",
            "retreat",
            "select_opponent_setup_target",
            "select_evolution_denial_target",
            "select_must_answer_target",
        }
        features: dict[str, float] = {}
        count = 0
        for step in sequence.steps:
            count += 1
            for key, value in step.features.items():
                if key in keep_exact or any(key.startswith(prefix) for prefix in keep_prefixes):
                    features[key] = features.get(key, 0.0) + float(value)
        if count <= 0:
            return features
        return {key: value / float(count) for key, value in features.items()}

    @staticmethod
    def _board_features(state: AgentState) -> dict[str, float]:
        my_active_hp = float(state.my_active.hp if state.my_active else 0)
        my_active_max = float(state.my_active.max_hp if state.my_active else 1)
        opp_active_hp = float(state.opp_active.hp if state.opp_active else 0)
        opp_active_max = float(state.opp_active.max_hp if state.opp_active else 1)
        return {
            "turn_frac": min(1.0, float(state.turn) / 16.0),
            "turn_early": 1.0 if state.turn <= 4 else 0.0,
            "turn_late": 1.0 if state.my_prizes_remaining <= 2 or state.opp_prizes_remaining <= 2 else 0.0,
            "prize_diff": float(max(-6, min(6, state.prize_diff))) / 6.0,
            "my_prize_pressure": float(6 - state.my_prizes_remaining) / 6.0,
            "opp_prize_pressure": float(6 - state.opp_prizes_remaining) / 6.0,
            "my_energy_in_play_clean": min(1.0, float(state.my_energy_in_play) / 10.0),
            "opp_energy_in_play_clean": min(1.0, float(state.opp_energy_in_play) / 10.0),
            "my_bench_count": min(1.0, float(len(state.my_bench)) / max(1, state.my_bench_max)),
            "opp_bench_count": min(1.0, float(len(state.opp_bench)) / max(1, state.opp_bench_max)),
            "my_active_hp_frac": max(0.0, min(1.0, my_active_hp / max(1.0, my_active_max))),
            "opp_active_hp_frac": max(0.0, min(1.0, opp_active_hp / max(1.0, opp_active_max))),
            "my_active_damage_frac": max(0.0, min(1.0, 1.0 - (my_active_hp / max(1.0, my_active_max)))),
            "opp_active_damage_frac": max(0.0, min(1.0, 1.0 - (opp_active_hp / max(1.0, opp_active_max)))),
            "my_active_prize_value": float(state.my_active.prize_value if state.my_active else 0) / 3.0,
            "opp_active_prize_value": float(state.opp_active.prize_value if state.opp_active else 0) / 3.0,
            "hand_size_clean": min(1.0, float(len(state.my_hand)) / 12.0),
            "opp_hand_size_clean": min(1.0, float(state.opp_hand_count) / 12.0),
        }

    @staticmethod
    def _consequence_features(root_state: AgentState, state: AgentState) -> dict[str, float]:
        features = {
            "seq_my_prizes_gained": float(root_state.my_prizes_remaining - state.my_prizes_remaining),
            "seq_opp_prizes_gained": float(root_state.opp_prizes_remaining - state.opp_prizes_remaining),
            "seq_energy_delta": max(-1.0, min(1.0, float(state.my_energy_in_play - root_state.my_energy_in_play) / 4.0)),
            "seq_bench_delta": max(-1.0, min(1.0, float(len(state.my_bench) - len(root_state.my_bench)) / 5.0)),
        }
        if state.result == root_state.player_index:
            features["seq_terminal_win"] = 1.0
        elif state.result not in (-1, 2, root_state.player_index):
            features["seq_terminal_loss"] = 1.0
        return features
