from __future__ import annotations

from dataclasses import replace
import math
import time
from typing import Any

from ..decision_types import EvaluationResult, PlanningResult, ScoreBreakdown, ScoredCandidate
from .counterplay_model import CounterplayModel
from .risk_model import RiskModel
from .score_calibration import ScoreCalibrator
from .value_proxy import HeuristicValueProxy


class SequenceEvaluator:
    """Evaluate complete candidate lines and return immutable score records."""

    XGB_ROOT_CAP = 24

    def __init__(
        self,
        *,
        featurizer: Any,
        scorer: Any,
        value_proxy: HeuristicValueProxy,
        clean: bool = False,
        value_proxy_enabled: bool = True,
        xgb_ranker: Any | None = None,
        risk_model: RiskModel | None = None,
        counterplay_model: CounterplayModel | None = None,
        calibrator: ScoreCalibrator | None = None,
    ):
        self.featurizer = featurizer
        self.scorer = scorer
        self.value_proxy = value_proxy
        self.clean = bool(clean)
        self.value_proxy_enabled = bool(value_proxy_enabled)
        self.xgb_ranker = xgb_ranker
        self.risk_model = risk_model or RiskModel()
        self.counterplay_model = counterplay_model or CounterplayModel()
        self.calibrator = calibrator or ScoreCalibrator()

    def evaluate(self, planning: PlanningResult) -> EvaluationResult:
        observed = planning.observed
        scored: list[ScoredCandidate] = []
        root_counterplay_risk = self.counterplay_model.response_risk(observed.state, observed.belief.opponent).total
        for sequence in planning.candidates:
            features = self.featurizer.featurize(
                observed.state,
                sequence,
                observed.belief,
                observed.game_plan,
                observed.resources,
                observed.threat,
            )
            value_score = 0.0 if self.clean or not self.value_proxy_enabled else self.value_proxy.evaluate(observed.state, sequence.state)
            if not self.clean and self.value_proxy_enabled:
                features["value_model_score"] = value_score
            risk_score = self.risk_model.score(features)
            counterplay = self.counterplay_model.evaluate(
                observed.state,
                sequence.state,
                observed.belief.opponent,
                root_risk=root_counterplay_risk,
            )
            counterplay_score = counterplay.score
            plan_bonus = self._plan_bonus(features)
            features["risk_model_score"] = risk_score
            features["opponent_counterplay_score"] = counterplay_score
            features["opponent_response_risk"] = counterplay.final_risk.total
            features["opponent_response_fragility"] = counterplay.fragility
            features["opponent_response_active_attack"] = counterplay.final_risk.active_attack
            features["opponent_response_gust_attack"] = counterplay.final_risk.gust_attack
            features["opponent_response_switch_attack"] = counterplay.final_risk.switch_attack
            features["opponent_response_energy_attack"] = counterplay.final_risk.energy_attack
            features["opponent_response_tool_attack"] = counterplay.final_risk.tool_attack
            features["opponent_response_evolution_attack"] = counterplay.final_risk.evolution_attack
            features["plan_bonus_score"] = plan_bonus
            score = self.scorer.score(features)
            scored.append(
                ScoredCandidate(
                    candidate=sequence,
                    features=dict(features),
                    score=float(score),
                    components=ScoreBreakdown(
                        linear=float(features.get("linear_score", 0.0)),
                        mlp=float(features["mlp_score"]) if "mlp_score" in features else None,
                        value_proxy=float(value_score),
                        risk=float(risk_score),
                        plan_bonus=float(plan_bonus),
                        opponent_counterplay=float(counterplay_score),
                    ),
                )
            )
        ranked = sorted(scored, key=lambda item: (item.score, -len(item.candidate.steps)), reverse=True)
        diagnostics: dict[str, Any] = {}
        guardrail_ranked: list[ScoredCandidate] = []
        if self.xgb_ranker is not None:
            ranked, guardrail_ranked, diagnostics = self._rerank_roots_with_xgb(ranked)
        ranked = [
            ScoredCandidate(
                candidate=item.candidate,
                features=item.features,
                score=item.score,
                components=item.components,
                rank=index + 1,
                warnings=item.warnings,
            )
            for index, item in enumerate(ranked)
        ]
        best = ranked[0] if ranked else None
        scores = [item.score for item in ranked]
        return EvaluationResult(
            planning=planning,
            scored=tuple(ranked),
            guardrail_scored=tuple(guardrail_ranked),
            best=best,
            score_breakdown={index: item.components for index, item in enumerate(ranked)},
            calibration=self.calibrator.summarize(scores),
            diagnostics=diagnostics,
        )

    def _rerank_roots_with_xgb(
        self,
        base_ranked: list[ScoredCandidate],
    ) -> tuple[list[ScoredCandidate], list[ScoredCandidate], dict[str, Any]]:
        from ..xgb_root_ranker import dedupe_scored_roots

        deduped_roots = list(dedupe_scored_roots(base_ranked))
        representatives = deduped_roots[: self.XGB_ROOT_CAP]
        if not representatives:
            return [], [], {
                "xgb_enabled": True,
                "base_candidate_count": len(base_ranked),
                "root_candidate_count": 0,
                "deduped_root_count": len(deduped_roots),
                "root_cap": self.XGB_ROOT_CAP,
                "roots_outside_cap": 0,
                "guardrail_root_count": 0,
                "outside_cap_guardrail_root_count": 0,
                "duplicate_root_candidates_removed": len(base_ranked) - len(deduped_roots),
                "base_best_first_selection": None,
                "xgb_best_first_selection": None,
                "xgb_changed_root": False,
                "xgb_scoring_ms": 0.0,
            }
        started_at = time.perf_counter()
        scores = list(self.xgb_ranker.score_batch([candidate.features for candidate in deduped_roots]))
        scoring_ms = (time.perf_counter() - started_at) * 1000.0
        if len(scores) != len(deduped_roots):
            raise ValueError(
                f"XGB root ranker returned {len(scores)} scores for {len(deduped_roots)} candidates"
            )
        guardrail_ranked: list[ScoredCandidate] = []
        for candidate, raw_score in zip(deduped_roots, scores):
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError("XGB root ranker returned a non-finite score")
            features = dict(candidate.features)
            features["xgb_score"] = score
            guardrail_ranked.append(
                replace(
                    candidate,
                    features=features,
                    score=score,
                    components=replace(candidate.components, xgb=score),
                    rank=0,
                )
            )
        selectable_keys = {self._root_key(candidate) for candidate in representatives}
        reranked = [
            candidate for candidate in guardrail_ranked if self._root_key(candidate) in selectable_keys
        ]
        reranked.sort(key=lambda item: (item.score, -len(item.candidate.steps)), reverse=True)
        guardrail_ranked.sort(key=lambda item: (item.score, -len(item.candidate.steps)), reverse=True)
        base_best = base_ranked[0] if base_ranked else None
        xgb_best = reranked[0]
        base_key = self._root_key(base_best) if base_best is not None else None
        xgb_key = self._root_key(xgb_best)
        return reranked, guardrail_ranked, {
            "xgb_enabled": True,
            "base_candidate_count": len(base_ranked),
            "root_candidate_count": len(representatives),
            "deduped_root_count": len(deduped_roots),
            "root_cap": self.XGB_ROOT_CAP,
            "roots_outside_cap": len(deduped_roots) - len(representatives),
            "guardrail_root_count": len(guardrail_ranked),
            "outside_cap_guardrail_root_count": len(guardrail_ranked) - len(reranked),
            "duplicate_root_candidates_removed": len(base_ranked) - len(deduped_roots),
            "base_best_first_selection": list(base_key) if base_key is not None else None,
            "xgb_best_first_selection": list(xgb_key),
            "xgb_changed_root": base_key != xgb_key,
            "xgb_scoring_ms": scoring_ms,
        }

    @staticmethod
    def _root_key(candidate: ScoredCandidate) -> tuple[int, ...]:
        return tuple(sorted(int(index) for index in candidate.candidate.first_selection))

    @staticmethod
    def _plan_bonus(features: dict[str, float]) -> float:
        bonus = 0.0
        for key, value in features.items():
            if key.startswith("plan_priority_"):
                bonus += float(value)
        return bonus
