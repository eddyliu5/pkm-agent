from __future__ import annotations

from ..features import score_features
from ..mlp_ranker import MLPSequenceRanker
from .weights import default_clean_sequence_weights, default_sequence_weights


class SequenceScorer:
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        *,
        mlp_ranker: MLPSequenceRanker | None = None,
        linear_weight: float = 1.0,
        mlp_weight: float = 5.0,
        clean: bool = False,
    ):
        self.weights = default_clean_sequence_weights() if clean else default_sequence_weights()
        if weights:
            self.weights.update({str(key): float(value) for key, value in weights.items()})
        self.mlp_ranker = mlp_ranker
        self.linear_weight = float(linear_weight)
        self.mlp_weight = float(mlp_weight)

    def score(self, features: dict[str, float]) -> float:
        linear_score = score_features(features, self.weights)
        features["linear_score"] = linear_score
        if self.mlp_ranker is None:
            return linear_score
        mlp_score = self.mlp_ranker.score_features(features)
        features["mlp_score"] = mlp_score
        return (self.linear_weight * linear_score) + (self.mlp_weight * mlp_score)
