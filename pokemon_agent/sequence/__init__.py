from __future__ import annotations

from ..evaluation.value_proxy import HeuristicValueProxy
from .featurizer import CleanSequenceFeaturizer, SequenceFeaturizer
from .generator import FeasibleSequenceSimulator
from .scorer import SequenceScorer
from .types import CandidateSequence, EvaluatedSequenceView, SearchNode, SearchTree, SelectionStep, SequenceSearchConfig
from .weights import default_clean_sequence_weights, default_sequence_weights


class ValueModel(HeuristicValueProxy):
    """Backward-compatible alias for the heuristic state-delta value proxy."""


__all__ = [
    "CandidateSequence",
    "CleanSequenceFeaturizer",
    "EvaluatedSequenceView",
    "FeasibleSequenceSimulator",
    "SearchNode",
    "SearchTree",
    "SelectionStep",
    "SequenceFeaturizer",
    "SequenceScorer",
    "SequenceSearchConfig",
    "ValueModel",
    "default_clean_sequence_weights",
    "default_sequence_weights",
]
