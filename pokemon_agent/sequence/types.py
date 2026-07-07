from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ..actions import ActionView
from ..state import AgentState


def _empty_features() -> Mapping[str, float]:
    return MappingProxyType({})


@dataclass(frozen=True)
class SequenceSearchConfig:
    beam_width: int = 8
    max_depth: int = 8
    max_nodes: int = 128
    time_budget_ms: int = 500
    option_pool_size: int = 10
    max_combinations_per_node: int = 24
    num_hidden_samples: int = 1


@dataclass(frozen=True)
class SelectionStep:
    selected_indices: tuple[int, ...]
    actions: tuple[ActionView, ...]
    select_type: str
    context: str
    features: dict[str, float]
    score: float
    labels: tuple[str, ...]


@dataclass(frozen=True)
class CandidateSequence:
    steps: tuple[SelectionStep, ...]
    observation: Any
    state: AgentState | None
    planner_score: float = 0.0
    features: Mapping[str, float] = field(default_factory=_empty_features)
    prune_tier: str = "PLAUSIBLE"
    prune_reasons: tuple[str, ...] = ()
    search_id: int | None = None
    terminal: bool = False
    used_search: bool = False
    generation_tags: tuple[str, ...] = ()
    sim_result: str | None = None

    @property
    def first_selection(self) -> list[int]:
        if not self.steps:
            return []
        return list(self.steps[0].selected_indices)

    @property
    def score(self) -> float:
        """Compatibility alias for planner-local priority, not final utility."""

        return self.planner_score

    @property
    def final_obs(self) -> Any:
        return self.observation

    @property
    def final_state(self) -> AgentState | None:
        return self.state


@dataclass(frozen=True)
class EvaluatedSequenceView:
    """Compatibility view for training code that consumes scored sequences."""

    candidate: CandidateSequence
    features: dict[str, float]
    score: float

    @property
    def steps(self) -> tuple[SelectionStep, ...]:
        return self.candidate.steps

    @property
    def first_selection(self) -> list[int]:
        return self.candidate.first_selection

    @property
    def observation(self) -> Any:
        return self.candidate.observation

    @property
    def state(self) -> AgentState | None:
        return self.candidate.state

    @property
    def terminal(self) -> bool:
        return self.candidate.terminal

    @property
    def used_search(self) -> bool:
        return self.candidate.used_search


@dataclass
class SearchNode:
    node_id: int
    observation: Any
    state: AgentState | None
    search_id: int | None
    parent_id: int | None = None
    incoming_step: SelectionStep | None = None
    depth: int = 0
    cumulative_score: float = 0.0
    children: list[int] = field(default_factory=list)
    terminal: bool = False
    used_search: bool = False


@dataclass
class SearchTree:
    nodes: list[SearchNode] = field(default_factory=list)

    def add_node(
        self,
        *,
        observation: Any,
        state: AgentState | None,
        search_id: int | None,
        parent_id: int | None = None,
        incoming_step: SelectionStep | None = None,
        depth: int = 0,
        cumulative_score: float = 0.0,
        terminal: bool = False,
        used_search: bool = False,
    ) -> int:
        node_id = len(self.nodes)
        self.nodes.append(
            SearchNode(
                node_id=node_id,
                observation=observation,
                state=state,
                search_id=search_id,
                parent_id=parent_id,
                incoming_step=incoming_step,
                depth=depth,
                cumulative_score=cumulative_score,
                terminal=terminal,
                used_search=used_search,
            )
        )
        if parent_id is not None:
            self.nodes[parent_id].children.append(node_id)
        return node_id

    def path_steps(self, node_id: int) -> tuple[SelectionStep, ...]:
        steps: list[SelectionStep] = []
        current_id: int | None = node_id
        while current_id is not None:
            node = self.nodes[current_id]
            if node.incoming_step is not None:
                steps.append(node.incoming_step)
            current_id = node.parent_id
        steps.reverse()
        return tuple(steps)

    def to_candidate(self, node_id: int) -> CandidateSequence:
        node = self.nodes[node_id]
        return CandidateSequence(
            steps=self.path_steps(node_id),
            observation=node.observation,
            state=node.state,
            planner_score=node.cumulative_score,
            search_id=node.search_id,
            terminal=node.terminal,
            used_search=node.used_search,
            generation_tags=("simulated",) if node.used_search else ("single_step",),
            sim_result="terminal" if node.terminal else "partial",
        )
