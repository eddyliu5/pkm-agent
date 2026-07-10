from __future__ import annotations

from typing import Any

from ..decision_types import EvaluationResult, PlanningResult, ScoreBreakdown, ScoredCandidate
from .counterplay_model import CounterplayModel
from .risk_model import RiskModel
from .score_calibration import ScoreCalibrator
from .value_proxy import HeuristicValueProxy


class SequenceEvaluator:
    """Evaluate complete candidate lines and return immutable score records."""

    def __init__(
        self,
        *,
        featurizer: Any,
        scorer: Any,
        value_proxy: HeuristicValueProxy,
        clean: bool = False,
        value_proxy_enabled: bool = True,
        risk_model: RiskModel | None = None,
        counterplay_model: CounterplayModel | None = None,
        calibrator: ScoreCalibrator | None = None,
    ):
        self.featurizer = featurizer
        self.scorer = scorer
        self.value_proxy = value_proxy
        self.clean = bool(clean)
        self.value_proxy_enabled = bool(value_proxy_enabled)
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
            best=best,
            score_breakdown={index: item.components for index, item in enumerate(ranked)},
            calibration=self.calibrator.summarize(scores),
        )

    @staticmethod
    def _plan_bonus(features: dict[str, float]) -> float:
        bonus = 0.0
        for key, value in features.items():
            if key.startswith("plan_priority_"):
                bonus += float(value)
        return bonus
