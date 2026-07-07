from __future__ import annotations


class ScoreCalibrator:
    """Compute score-range diagnostics for an evaluated candidate set."""

    def summarize(self, scores: list[float]) -> dict[str, float]:
        if not scores:
            return {"min_score": 0.0, "max_score": 0.0, "candidate_count": 0.0, "score_range": 0.0}
        min_score = min(scores)
        max_score = max(scores)
        return {
            "min_score": min_score,
            "max_score": max_score,
            "candidate_count": float(len(scores)),
            "score_range": max_score - min_score,
        }
