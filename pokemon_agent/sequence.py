from __future__ import annotations

import itertools
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

from .actions import ActionView, legal_actions, sanitize_selection
from .belief import BeliefState
from .evaluators import ResourceSnapshot, ThreatSnapshot
from .features import combine_feature_dicts, default_weights, extract_action_features, score_features
from .game_plan import PlanContext
from .knowledge import CardKnowledgeStore
from .state import AgentState, build_agent_state

from .cg_loader import ensure_cg_path

ensure_cg_path()

from cg.api import search_begin, search_end, search_step  # noqa: E402


@dataclass(frozen=True)
class SequenceSearchConfig:
    beam_width: int = 8
    max_depth: int = 8
    max_nodes: int = 128
    time_budget_ms: int = 500
    option_pool_size: int = 10
    max_combinations_per_node: int = 24


@dataclass(frozen=True)
class SelectionStep:
    selected_indices: tuple[int, ...]
    actions: tuple[ActionView, ...]
    select_type: str
    context: str
    features: dict[str, float]
    score: float
    labels: tuple[str, ...]


@dataclass
class CandidateSequence:
    steps: tuple[SelectionStep, ...]
    observation: Any
    state: AgentState | None
    score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    prune_tier: str = "PLAUSIBLE"
    prune_reasons: tuple[str, ...] = ()
    search_id: int | None = None
    terminal: bool = False
    used_search: bool = False

    @property
    def first_selection(self) -> list[int]:
        if not self.steps:
            return []
        return list(self.steps[0].selected_indices)


def default_sequence_weights() -> dict[str, float]:
    weights = default_weights()
    weights.update(
        {
            "seq_length": -6.0,
            "seq_uses_attack": 50.0,
            "seq_uses_end": -25.0,
            "seq_my_prizes_gained": 850.0,
            "seq_opp_prizes_gained": -850.0,
            "seq_energy_delta": 120.0,
            "seq_bench_delta": 45.0,
            "seq_terminal_win": 20000.0,
            "seq_terminal_loss": -20000.0,
            "plan_pressure": 70.0,
            "plan_setup_big_attacker": 80.0,
            "phase_setup": 50.0,
            "phase_pressure": 35.0,
            "phase_close": 150.0,
            "resource_ready_attackers": 120.0,
            "resource_hand_quality": 180.0,
            "resource_energy_in_play": 70.0,
            "threat_can_ko_active": -140.0,
            "threat_can_win_next_turn": -500.0,
            "opponent_gust_risk": -80.0,
            "opponent_pressure_plan": -90.0,
            "opponent_setup_big_attacker_plan": 25.0,
            "opponent_expected_first_attack_turn": 65.0,
            "opponent_expected_game_end_turn": 15.0,
            "prize_known_key_prized": -120.0,
            "prize_unknown_count": -10.0,
            "value_model_score": 400.0,
        }
    )
    return weights


class SequencePruner:
    def __init__(self, config: dict[str, Any], weights: dict[str, float] | None = None):
        self.config = config
        self.weights = dict(default_weights())
        if weights:
            self.weights.update({str(key): float(value) for key, value in weights.items()})

    def selection_candidates(self, obs: Any, state: AgentState, search_config: SequenceSearchConfig) -> list[SelectionStep]:
        if obs.select is None:
            return []
        actions = legal_actions(obs)
        if not actions:
            return []
        scored_actions: list[tuple[ActionView, float, dict[str, float]]] = []
        winning_indices: set[int] = set()
        for action in actions:
            features = extract_action_features(state, action, self.config)
            score = score_features(features, self.weights)
            scored_actions.append((action, score, features))
            if features.get("attack_wins_game", 0.0) > 0:
                winning_indices.add(action.index)

        hard_pruned = {
            action.index
            for action, _, _ in scored_actions
            if action.is_end and winning_indices
        }
        pool = [item for item in scored_actions if item[0].index not in hard_pruned]
        pool.sort(key=lambda item: (item[1], -item[0].index), reverse=True)
        must_keep = [item for item in pool if item[0].index in winning_indices]
        kept = must_keep + [item for item in pool if item[0].index not in winning_indices]
        kept = kept[: max(1, search_config.option_pool_size)]
        by_index = {action.index: (action, score, features) for action, score, features in kept}

        min_count = max(0, int(obs.select.minCount or 0))
        max_count = min(int(obs.select.maxCount or 0), len(actions))
        counts = self._selection_counts(min_count, max_count)
        selections: list[SelectionStep] = []
        if 0 in counts:
            selections.append(self._step_from_items(obs, state, [], by_index))
        indices = [action.index for action, _, _ in kept]
        for count in counts:
            if count <= 0:
                continue
            for combo in itertools.combinations(indices, count):
                if winning_indices and not any(index in winning_indices for index in combo):
                    continue
                selections.append(self._step_from_items(obs, state, list(combo), by_index))
        selections.sort(key=lambda step: (step.score, -len(step.selected_indices)), reverse=True)
        return selections[: max(1, search_config.max_combinations_per_node)]

    @staticmethod
    def _selection_counts(min_count: int, max_count: int) -> list[int]:
        if max_count <= 0:
            return [0] if min_count == 0 else []
        if min_count == max_count:
            return [min_count]
        values = {min_count, max_count}
        if min_count <= 1 <= max_count:
            values.add(1)
        return sorted(value for value in values if min_count <= value <= max_count)

    def _step_from_items(
        self,
        obs: Any,
        state: AgentState,
        selected: list[int],
        by_index: dict[int, tuple[ActionView, float, dict[str, float]]],
    ) -> SelectionStep:
        selected_actions = [by_index[index][0] for index in selected if index in by_index]
        features = combine_feature_dicts([by_index[index][2] for index in selected if index in by_index])
        score = score_features(features, self.weights)
        return SelectionStep(
            selected_indices=tuple(selected),
            actions=tuple(selected_actions),
            select_type=state.select_type,
            context=state.select_context,
            features=features,
            score=score,
            labels=tuple(action.label for action in selected_actions),
        )


class SequenceFeaturizer:
    def __init__(self, config: dict[str, Any], knowledge: CardKnowledgeStore):
        self.config = config
        self.knowledge = knowledge

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
        features["seq_length"] = min(1.0, float(len(sequence.steps)) / 8.0)
        features["seq_uses_attack"] = 1.0 if any(action.is_attack for step in sequence.steps for action in step.actions) else 0.0
        features["seq_uses_end"] = 1.0 if any(action.is_end for step in sequence.steps for action in step.actions) else 0.0
        features[f"plan_{plan.deck_plan_type}"] = 1.0
        features[f"phase_{plan.phase}"] = 1.0
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
        if belief.opponent.plan_type:
            features[f"opponent_{belief.opponent.plan_type}_plan"] = 1.0
        features["opponent_expected_first_attack_turn"] = min(1.0, belief.opponent.expected_first_attack_turn / 6.0)
        features["opponent_expected_game_end_turn"] = min(1.0, belief.opponent.expected_game_end_turn / 14.0)
        features["prize_known_key_prized"] = min(1.0, float(sum(belief.prize_info.known_prized.values())) / 6.0)
        features["prize_unknown_count"] = min(1.0, float(belief.prize_info.unknown_prize_count) / 6.0)

        if sequence.state is not None:
            features["seq_my_prizes_gained"] = float(root_state.my_prizes_remaining - sequence.state.my_prizes_remaining)
            features["seq_opp_prizes_gained"] = float(root_state.opp_prizes_remaining - sequence.state.opp_prizes_remaining)
            features["seq_energy_delta"] = max(-1.0, min(1.0, float(sequence.state.my_energy_in_play - root_state.my_energy_in_play) / 4.0))
            features["seq_bench_delta"] = max(-1.0, min(1.0, float(len(sequence.state.my_bench) - len(root_state.my_bench)) / 5.0))
            if sequence.state.result == root_state.player_index:
                features["seq_terminal_win"] = 1.0
            elif sequence.state.result not in (-1, 2, root_state.player_index):
                features["seq_terminal_loss"] = 1.0

        pooled = self._pooled_card_embedding(sequence)
        for index, value in enumerate(pooled[:12]):
            features[f"card_embed_{index}"] = value
        return features

    def _pooled_card_embedding(self, sequence: CandidateSequence) -> tuple[float, ...]:
        vectors: list[tuple[float, ...]] = []
        for step in sequence.steps:
            for action in step.actions:
                if action.source_card_id is not None:
                    vectors.append(self.knowledge.vector(action.source_card_id))
                if action.target_card_id is not None:
                    vectors.append(self.knowledge.vector(action.target_card_id))
        if not vectors:
            return (0.0,) * 12
        dim = len(vectors[0])
        values = [0.0] * dim
        for vector in vectors:
            for index, value in enumerate(vector):
                values[index] += value
        scale = 1.0 / float(len(vectors))
        return tuple(value * scale for value in values)


class SequenceScorer:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = default_sequence_weights()
        if weights:
            self.weights.update({str(key): float(value) for key, value in weights.items()})

    def score(self, features: dict[str, float]) -> float:
        return score_features(features, self.weights)


class ValueModel:
    """Heuristic state value placeholder with the same runtime shape as a learned value head."""

    def evaluate(self, root_state: AgentState, state: AgentState | None) -> float:
        if state is None:
            return 0.0
        if state.result == root_state.player_index:
            return 1.0
        if state.result not in (-1, 2, root_state.player_index):
            return -1.0
        prize_gain = float(root_state.my_prizes_remaining - state.my_prizes_remaining)
        prize_loss = float(root_state.opp_prizes_remaining - state.opp_prizes_remaining)
        energy_delta = float(state.my_energy_in_play - root_state.my_energy_in_play) / 6.0
        board_delta = float(len(state.my_bench) - len(root_state.my_bench)) / 5.0
        prize_value = (prize_gain - prize_loss) / 6.0
        return max(-1.0, min(1.0, prize_value + 0.25 * energy_delta + 0.1 * board_delta))


class ActionSelector:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select(self, sequences: list[CandidateSequence], mode: str = "eval") -> CandidateSequence | None:
        if not sequences:
            return None
        if mode == "train":
            temperature = 250.0
            best = max(sequence.score for sequence in sequences)
            weights = [math.exp(max(-20.0, min(20.0, (sequence.score - best) / temperature))) for sequence in sequences]
            total = sum(weights) or 1.0
            pick = self.rng.random() * total
            running = 0.0
            for sequence, weight in zip(sequences, weights):
                running += weight
                if running >= pick:
                    return sequence
        return max(sequences, key=lambda sequence: (sequence.score, -len(sequence.steps)))


class FeasibleSequenceSimulator:
    def __init__(self, config: dict[str, Any], weights: dict[str, float] | None = None):
        self.config = config
        self.pruner = SequencePruner(config, weights=weights)

    def generate(
        self,
        obs: Any,
        root_state: AgentState,
        belief: BeliefState,
        search_config: SequenceSearchConfig,
    ) -> list[CandidateSequence]:
        sample = belief.primary_hidden_sample
        if obs.select is None or sample is None:
            return []
        if not getattr(obs, "search_begin_input", None):
            return self._single_step_sequences(obs, root_state, search_config)

        start = time.monotonic()
        nodes = 0
        try:
            root_search = search_begin(
                obs,
                list(sample.your_deck),
                list(sample.your_prize),
                list(sample.opponent_deck),
                list(sample.opponent_prize),
                list(sample.opponent_hand),
                list(sample.opponent_active),
                False,
            )
            beam = [
                CandidateSequence(
                    steps=(),
                    observation=root_search.observation,
                    state=root_state,
                    search_id=root_search.searchId,
                    used_search=True,
                )
            ]
            complete: list[CandidateSequence] = []
            root_player = root_state.player_index
            root_turn = root_state.turn
            for _depth in range(search_config.max_depth):
                expanded: list[CandidateSequence] = []
                for candidate in beam:
                    if self._should_stop(candidate.observation, root_player, root_turn):
                        candidate.terminal = True
                        complete.append(candidate)
                        continue
                    if nodes >= search_config.max_nodes or self._timed_out(start, search_config):
                        candidate.terminal = True
                        complete.append(candidate)
                        continue
                    current_obs = candidate.observation
                    actions = legal_actions(current_obs)
                    current_state = build_agent_state(current_obs, perspective=root_player, actions=actions)
                    for step in self.pruner.selection_candidates(current_obs, current_state, search_config):
                        if nodes >= search_config.max_nodes or self._timed_out(start, search_config):
                            break
                        selected, invalid = sanitize_selection(current_obs, list(step.selected_indices))
                        if invalid:
                            continue
                        try:
                            next_search = search_step(int(candidate.search_id), selected)
                        except Exception:
                            continue
                        next_state = self._state_from_observation(next_search.observation, root_player)
                        expanded.append(
                            CandidateSequence(
                                steps=candidate.steps + (step,),
                                observation=next_search.observation,
                                state=next_state,
                                score=candidate.score + step.score,
                                search_id=next_search.searchId,
                                used_search=True,
                            )
                        )
                        nodes += 1
                if not expanded:
                    break
                expanded.sort(key=lambda sequence: sequence.score, reverse=True)
                beam = expanded[: search_config.beam_width]
            complete.extend(beam)
            return complete or self._single_step_sequences(obs, root_state, search_config)
        except Exception:
            return self._single_step_sequences(obs, root_state, search_config)
        finally:
            try:
                search_end()
            except Exception:
                pass

    @staticmethod
    def _timed_out(start: float, search_config: SequenceSearchConfig) -> bool:
        return (time.monotonic() - start) * 1000.0 >= float(search_config.time_budget_ms)

    @staticmethod
    def _should_stop(obs: Any, root_player: int, root_turn: int) -> bool:
        if obs.current is None:
            return True
        if obs.current.result != -1:
            return True
        if obs.select is None:
            return True
        if obs.current.yourIndex != root_player:
            return True
        if obs.current.turn != root_turn:
            return True
        return False

    @staticmethod
    def _state_from_observation(obs: Any, root_player: int) -> AgentState | None:
        if obs.current is None:
            return None
        try:
            actions = legal_actions(obs) if obs.select is not None else []
            return build_agent_state(obs, perspective=root_player, actions=actions)
        except Exception:
            return None

    def _single_step_sequences(
        self,
        obs: Any,
        root_state: AgentState,
        search_config: SequenceSearchConfig,
    ) -> list[CandidateSequence]:
        sequences: list[CandidateSequence] = []
        for step in self.pruner.selection_candidates(obs, root_state, search_config):
            sequences.append(
                CandidateSequence(
                    steps=(step,),
                    observation=obs,
                    state=root_state,
                    score=step.score,
                    terminal=True,
                    used_search=False,
                )
            )
        return sequences
