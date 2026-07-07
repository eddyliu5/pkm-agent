from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..actions import legal_actions, sanitize_selection
from ..belief import BeliefState, HiddenSample
from ..opponent_memory import OpponentIntentSnapshot
from ..planning import SequencePruner as PlannerSequencePruner
from ..state import AgentState, build_agent_state
from .budget import SearchBudget
from .search_adapter import SearchAdapter
from .types import CandidateSequence, SearchTree, SequenceSearchConfig


class FeasibleSequenceSimulator:
    def __init__(self, config: dict[str, Any], weights: dict[str, float] | None = None):
        self.config = config
        self.pruner = PlannerSequencePruner(config, weights=weights)
        self.search = SearchAdapter()
        self.last_tree: SearchTree | None = None
        self.last_diagnostics: dict[str, Any] = {}

    def generate(
        self,
        obs: Any,
        root_state: AgentState,
        belief: BeliefState,
        search_config: SequenceSearchConfig,
    ) -> list[CandidateSequence]:
        self.last_tree = None
        self.last_diagnostics = {}
        samples = belief.hidden_samples[: max(1, int(search_config.num_hidden_samples))]
        if obs.select is None or not samples:
            self.last_diagnostics = {"reason": "missing_selection_or_hidden_sample"}
            return []
        if not getattr(obs, "search_begin_input", None):
            self.last_diagnostics = {"hidden_samples_used": ("sample_0",), "sample_count": 1}
            return self._single_step_sequences(obs, root_state, search_config, belief.opponent.intent)

        all_sequences: list[CandidateSequence] = []
        diagnostics: dict[str, Any] = {
            "node_budget_hit": False,
            "time_budget_hit": False,
            "nodes_expanded": 0,
            "hidden_samples_used": tuple(f"sample_{index}" for index in range(len(samples))),
            "sample_count": len(samples),
        }
        per_sample_config = replace(
            search_config,
            max_nodes=max(1, int(search_config.max_nodes) // len(samples)),
            time_budget_ms=max(1, int(search_config.time_budget_ms) // len(samples)),
            num_hidden_samples=1,
        )
        for sample_index, sample in enumerate(samples):
            sequences, sample_diagnostics, sample_tree = self._generate_for_sample(
                obs,
                root_state,
                belief,
                per_sample_config,
                sample,
                sample_index,
            )
            if self.last_tree is None and sample_tree is not None:
                self.last_tree = sample_tree
            all_sequences.extend(sequences)
            diagnostics["node_budget_hit"] = bool(diagnostics["node_budget_hit"] or sample_diagnostics.get("node_budget_hit", False))
            diagnostics["time_budget_hit"] = bool(diagnostics["time_budget_hit"] or sample_diagnostics.get("time_budget_hit", False))
            diagnostics["nodes_expanded"] = int(diagnostics["nodes_expanded"]) + int(sample_diagnostics.get("nodes_expanded", 0))
        self.last_diagnostics = diagnostics
        deduped = self._dedupe_sequences(all_sequences)
        return deduped or self._single_step_sequences(obs, root_state, search_config, belief.opponent.intent)

    def _generate_for_sample(
        self,
        obs: Any,
        root_state: AgentState,
        belief: BeliefState,
        search_config: SequenceSearchConfig,
        sample: HiddenSample,
        sample_index: int,
    ) -> tuple[list[CandidateSequence], dict[str, Any], SearchTree | None]:
        budget = SearchBudget(search_config)
        tree: SearchTree | None = None
        try:
            root_search = self.search.begin(obs, sample)
            tree = SearchTree()
            root_id = tree.add_node(
                observation=root_search.observation,
                state=root_state,
                search_id=root_search.searchId,
                used_search=True,
            )
            beam = [root_id]
            complete: list[int] = []
            completed_seen: set[int] = set()
            root_player = root_state.player_index
            root_turn = root_state.turn

            def mark_complete(node_id: int) -> None:
                tree.nodes[node_id].terminal = True
                if node_id not in completed_seen:
                    complete.append(node_id)
                    completed_seen.add(node_id)

            for _depth in range(search_config.max_depth):
                expanded: list[int] = []
                for node_id in beam:
                    node = tree.nodes[node_id]
                    if self._should_stop(node.observation, root_player, root_turn):
                        mark_complete(node_id)
                        continue
                    if not budget.can_expand():
                        mark_complete(node_id)
                        continue
                    current_obs = node.observation
                    actions = legal_actions(current_obs)
                    current_state = build_agent_state(current_obs, perspective=root_player, actions=actions)
                    for step in self.pruner.selection_candidates(current_obs, current_state, search_config, belief.opponent.intent):
                        if not budget.can_expand():
                            break
                        selected, invalid = sanitize_selection(current_obs, list(step.selected_indices))
                        if invalid:
                            continue
                        try:
                            next_search = self.search.step(int(node.search_id), selected)
                        except Exception:
                            continue
                        next_state = self._state_from_observation(next_search.observation, root_player)
                        child_id = tree.add_node(
                            observation=next_search.observation,
                            state=next_state,
                            search_id=next_search.searchId,
                            parent_id=node_id,
                            incoming_step=step,
                            depth=node.depth + 1,
                            cumulative_score=node.cumulative_score + step.score,
                            used_search=True,
                        )
                        expanded.append(child_id)
                        budget.record_node()
                if not expanded:
                    break
                expanded.sort(key=lambda node_id: tree.nodes[node_id].cumulative_score, reverse=True)
                beam = expanded[: search_config.beam_width]
            for node_id in beam:
                if node_id not in completed_seen:
                    complete.append(node_id)
                    completed_seen.add(node_id)
            sequences = [
                replace(
                    tree.to_candidate(node_id),
                    generation_tags=(*tree.to_candidate(node_id).generation_tags, f"sample_{sample_index}"),
                )
                for node_id in complete
            ]
            diagnostics = {
                "node_budget_hit": budget.node_budget_hit,
                "time_budget_hit": budget.time_budget_hit,
                "nodes_expanded": budget.nodes,
            }
            return sequences or self._single_step_sequences(obs, root_state, search_config, belief.opponent.intent), diagnostics, tree
        except Exception as exc:
            diagnostics = {"fallback_reason": type(exc).__name__, "nodes_expanded": budget.nodes}
            return self._single_step_sequences(obs, root_state, search_config, belief.opponent.intent), diagnostics, tree
        finally:
            try:
                self.search.end()
            except Exception:
                pass

    @staticmethod
    def _dedupe_sequences(sequences: list[CandidateSequence]) -> list[CandidateSequence]:
        deduped: list[CandidateSequence] = []
        seen: dict[tuple[tuple[int, ...], ...], int] = {}
        for sequence in sequences:
            key = tuple(tuple(step.selected_indices) for step in sequence.steps)
            if key in seen:
                existing_index = seen[key]
                existing = deduped[existing_index]
                merged_tags = tuple(dict.fromkeys((*existing.generation_tags, *sequence.generation_tags)))
                deduped[existing_index] = replace(existing, generation_tags=merged_tags)
                continue
            seen[key] = len(deduped)
            deduped.append(sequence)
        return deduped

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
        opponent_intent: OpponentIntentSnapshot | None = None,
    ) -> list[CandidateSequence]:
        sequences: list[CandidateSequence] = []
        for step in self.pruner.selection_candidates(obs, root_state, search_config, opponent_intent):
            sequences.append(
                CandidateSequence(
                    steps=(step,),
                    observation=obs,
                    state=root_state,
                    planner_score=step.score,
                    terminal=True,
                    used_search=False,
                    generation_tags=("single_step",),
                    sim_result="terminal",
                )
            )
        return sequences
