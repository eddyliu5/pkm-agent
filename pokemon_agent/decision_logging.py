from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .actions import ActionView, legal_actions


DECISION_LOG_SCHEMA_VERSION = 1


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0
    return number


def _action_record(action: ActionView) -> dict[str, Any]:
    return {
        "index": int(action.index),
        "label": action.label,
        "select_type": action.select_type,
        "context": action.context,
        "option_type": action.option_type,
        "source_player_index": action.source_player_index,
        "source_card_id": action.source_card_id,
        "source_name": action.source_name,
        "target_player_index": action.target_player_index,
        "target_card_id": action.target_card_id,
        "target_name": action.target_name,
        "attack_id": action.attack_id,
        "attack_name": action.attack_name,
    }


def _features_record(features: dict[str, Any]) -> dict[str, float]:
    return {str(key): _finite_float(value) for key, value in sorted(features.items())}


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return _finite_float(value)
    return str(value)


def _coverage_record(coverage: Any) -> dict[str, Any]:
    if coverage is None:
        return {}
    return {
        "num_legal_first_actions": int(getattr(coverage, "num_legal_first_actions", 0)),
        "num_first_actions_represented": int(getattr(coverage, "num_first_actions_represented", 0)),
        "missing_first_actions": [list(item) for item in getattr(coverage, "missing_first_actions", ())],
        "num_candidates": int(getattr(coverage, "num_candidates", 0)),
        "max_depth_reached": int(getattr(coverage, "max_depth_reached", 0)),
        "node_budget_hit": bool(getattr(coverage, "node_budget_hit", False)),
        "time_budget_hit": bool(getattr(coverage, "time_budget_hit", False)),
        "candidate_tags": list(getattr(coverage, "candidate_tags", ())),
    }


def _trace_record(trace: Any) -> dict[str, Any]:
    if trace is None:
        return {}
    return {
        "sequence_count": int(getattr(trace, "sequence_count", 0)),
        "used_search": bool(getattr(trace, "used_search", False)),
        "chosen_score": _finite_float(getattr(trace, "chosen_score", 0.0)),
        "chosen_labels": list(getattr(trace, "chosen_labels", ())),
        "fallback_reason": getattr(trace, "fallback_reason", None),
    }


def _candidate_record(candidate: Any) -> dict[str, Any]:
    sequence = getattr(candidate, "candidate", None)
    steps = list(getattr(sequence, "steps", ()) or ())
    components = getattr(candidate, "components", None)
    return {
        "rank": int(getattr(candidate, "rank", 0)),
        "score": _finite_float(getattr(candidate, "score", 0.0)),
        "components": {
            "linear": _finite_float(getattr(components, "linear", 0.0)),
            "mlp": None if getattr(components, "mlp", None) is None else _finite_float(getattr(components, "mlp")),
            "value_proxy": _finite_float(getattr(components, "value_proxy", 0.0)),
            "risk": _finite_float(getattr(components, "risk", 0.0)),
            "plan_bonus": _finite_float(getattr(components, "plan_bonus", 0.0)),
            "opponent_counterplay": _finite_float(getattr(components, "opponent_counterplay", 0.0)),
        },
        "first_selection": list(getattr(sequence, "first_selection", []) or []),
        "steps": [
            {
                "selected_indices": list(getattr(step, "selected_indices", ()) or ()),
                "labels": list(getattr(step, "labels", ()) or ()),
                "select_type": getattr(step, "select_type", None),
                "context": getattr(step, "context", None),
                "score": _finite_float(getattr(step, "score", 0.0)),
                "features": _features_record(getattr(step, "features", {}) or {}),
            }
            for step in steps
        ],
        "depth": len(steps),
        "used_search": bool(getattr(sequence, "used_search", False)),
        "terminal": bool(getattr(sequence, "terminal", False)),
        "generation_tags": list(getattr(sequence, "generation_tags", ()) or ()),
        "sim_result": getattr(sequence, "sim_result", None),
        "features": _features_record(getattr(candidate, "features", {}) or {}),
        "warnings": list(getattr(candidate, "warnings", ()) or ()),
    }


def serialize_policy_decision(
    policy: Any,
    obs: Any,
    *,
    selected: list[int],
    selected_raw: list[int],
    invalid: bool,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    engine = getattr(policy, "engine", None)
    if engine is None:
        return None
    observed = getattr(engine, "last_observed_context", None)
    planning = getattr(engine, "last_planning_result", None)
    evaluated = getattr(engine, "last_evaluation_result", None)
    decision = getattr(engine, "last_decision", None)
    if observed is None or planning is None or evaluated is None:
        return None

    actions = legal_actions(obs)
    actions_by_index = {action.index: action for action in actions}
    selected_labels = [actions_by_index[index].label for index in selected if index in actions_by_index]
    state = getattr(observed, "state", None)
    threat = getattr(observed, "threat", None)
    resources = getattr(observed, "resources", None)
    probabilities = getattr(observed, "probabilities", None)
    belief = getattr(observed, "belief", None)
    opponent = getattr(belief, "opponent", None)
    record = {
        "schema_version": DECISION_LOG_SCHEMA_VERSION,
        **(metadata or {}),
        "player_index": int(getattr(state, "player_index", -1)),
        "turn": int(getattr(state, "turn", -1)),
        "select_type": getattr(state, "select_type", None),
        "select_context": getattr(state, "select_context", None),
        "selected": list(selected),
        "selected_raw": list(selected_raw),
        "selected_labels": selected_labels,
        "invalid_selection": bool(invalid),
        "legal_actions": [_action_record(action) for action in actions],
        "root_state": {
            "my_prizes_remaining": getattr(state, "my_prizes_remaining", None),
            "opponent_prizes_remaining": getattr(state, "opp_prizes_remaining", None),
            "prize_diff": getattr(state, "prize_diff", None),
            "hand_size": len(getattr(state, "my_hand", ()) or ()),
            "deck_size": getattr(state, "my_deck_count", None),
            "active_card_id": getattr(getattr(state, "my_active", None), "id", None),
            "opponent_active_card_id": getattr(getattr(state, "opp_active", None), "id", None),
            "bench_size": len(getattr(state, "my_bench", ()) or ()),
            "opponent_bench_size": len(getattr(state, "opp_bench", ()) or ()),
        },
        "belief": {
            "opponent_intent": _json_value(getattr(opponent, "intent", None)),
            "opponent_archetype": _json_value(getattr(opponent, "archetype", None)),
        },
        "resources": _json_value(resources),
        "threat": _json_value(threat),
        "probabilities": _json_value(probabilities),
        "planning_coverage": _coverage_record(getattr(planning, "coverage", None)),
        "planning_diagnostics": _json_value(getattr(planning, "diagnostics", {}) or {}),
        "hidden_samples_used": list(getattr(planning, "hidden_samples_used", ()) or ()),
        "trace": _trace_record(getattr(decision, "trace", None)),
        "calibration": _json_value(getattr(evaluated, "calibration", {}) or {}),
        "candidates": [_candidate_record(candidate) for candidate in getattr(evaluated, "scored", ())],
    }
    return record
