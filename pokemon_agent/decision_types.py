from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .actions import ActionView
from .belief import BeliefState
from .game_plan import PlanContext
from .observation.snapshots import ResourceSnapshot, ThreatSnapshot
from .opponent_memory import OpponentMemory
from .probability import ProbabilitySnapshot
from .state import AgentState


@dataclass(frozen=True)
class DecisionTrace:
    sequence_count: int
    used_search: bool
    chosen_score: float
    chosen_labels: tuple[str, ...]
    fallback_reason: str | None = None


@dataclass(frozen=True)
class ObservedContext:
    """Parsed public state plus belief snapshots for the current selection."""

    observation: Any
    actions: tuple[ActionView, ...]
    state: AgentState
    opponent_memory: OpponentMemory
    belief: BeliefState
    probabilities: ProbabilitySnapshot
    game_plan: PlanContext
    resources: ResourceSnapshot
    threat: ThreatSnapshot
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanningCoverage:
    """Planner coverage summary for first actions and budget limits."""

    num_legal_first_actions: int
    num_first_actions_represented: int
    missing_first_actions: tuple[tuple[int, ...], ...] = ()
    num_candidates: int = 0
    max_depth_reached: int = 0
    node_budget_hit: bool = False
    time_budget_hit: bool = False
    candidate_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanningResult:
    """Simulator-backed sequence-search output."""

    observed: ObservedContext
    tree: Any | None
    candidates: tuple[Any, ...]
    coverage: PlanningCoverage | None = None
    pruned_actions: tuple[str, ...] = ()
    hidden_samples_used: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreBreakdown:
    linear: float = 0.0
    mlp: float | None = None
    xgb: float | None = None
    value_proxy: float = 0.0
    risk: float = 0.0
    plan_bonus: float = 0.0
    opponent_counterplay: float = 0.0


@dataclass(frozen=True)
class ScoredCandidate:
    """Immutable evaluation output for one candidate sequence."""

    candidate: Any
    features: dict[str, float]
    score: float
    components: ScoreBreakdown
    rank: int = 0
    warnings: tuple[str, ...] = ()

    @property
    def sequence(self) -> Any:
        return self.candidate

    @property
    def linear_score(self) -> float:
        return self.components.linear

    @property
    def mlp_score(self) -> float | None:
        return self.components.mlp

    @property
    def xgb_score(self) -> float | None:
        return self.components.xgb

    @property
    def value_score(self) -> float:
        return self.components.value_proxy

    def to_sequence_view(self) -> Any:
        """Return a sequence-shaped compatibility object with evaluation fields."""

        from .sequence.types import EvaluatedSequenceView

        return EvaluatedSequenceView(candidate=self.candidate, features=dict(self.features), score=float(self.score))


@dataclass(frozen=True)
class EvaluationResult:
    """All candidate evaluations for an observed/planned state."""

    planning: PlanningResult
    scored: tuple[ScoredCandidate, ...]
    best: ScoredCandidate | None = None
    score_breakdown: dict[int, ScoreBreakdown] = field(default_factory=dict)
    calibration: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    guardrail_scored: tuple[ScoredCandidate, ...] = ()

    @property
    def evaluations(self) -> tuple[ScoredCandidate, ...]:
        return self.scored

    @property
    def sequences(self) -> list[Any]:
        return [candidate.to_sequence_view() for candidate in self.scored]


@dataclass(frozen=True)
class AgentDecision:
    """Final committed simulator selection and trace."""

    selected: list[int]
    trace: DecisionTrace
    chosen_sequence: Any | None = None
    scored_candidate: ScoredCandidate | None = None
