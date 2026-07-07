from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .actions import ActionView
from .card_db import card_data, is_energy
from .features import extract_action_features
from .state import AgentState, PokemonView


MODEL_TYPE = "reply_risk_linear"
SCHEMA_VERSION = 1

TARGET_NAMES = (
    "hand_has_gust",
    "hand_has_switch",
    "hand_has_energy",
    "hand_has_search",
    "hand_has_draw",
    "hand_has_evolution",
    "hand_has_main_attacker",
    "selected_attack",
    "selected_attach",
    "selected_evolve",
    "selected_ability",
    "selected_retreat",
    "selected_gust",
    "selected_search",
    "selected_draw",
    "selected_switch",
    "selected_end",
    "takes_ko",
    "wins_game",
    "selects_setup_target",
    "selects_must_answer_target",
)


@dataclass(frozen=True)
class ReplyRiskStats:
    feature_names: list[str]
    feature_counts: dict[str, int]
    example_count: int
    target_positive_counts: dict[str, int]


def _ids(config: dict[str, Any], key: str) -> set[int]:
    return {int(card_id) for card_id in config.get(key, [])}


def _norm(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return max(-1.0, min(1.0, float(value) / float(scale)))


def _sigmoid(value: float) -> float:
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _clean_features(features: Any) -> dict[str, float]:
    if not isinstance(features, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key, value in features.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number and math.isfinite(number):
            cleaned[str(key)] = number
    return cleaned


def _clean_labels(labels: Any) -> dict[str, float]:
    if not isinstance(labels, dict):
        return {}
    cleaned: dict[str, float] = {}
    for target in TARGET_NAMES:
        cleaned[target] = 1.0 if float(labels.get(target, 0.0) or 0.0) > 0.0 else 0.0
    return cleaned


def _iter_examples(path: str | Path, max_examples: int | None = None) -> Iterable[tuple[dict[str, float], dict[str, float]]]:
    seen = 0
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            features = _clean_features(record.get("features"))
            labels = _clean_labels(record.get("labels"))
            if not features or not labels:
                continue
            yield features, labels
            seen += 1
            if max_examples is not None and seen >= max_examples:
                break


def build_reply_risk_stats(
    labels_path: str | Path,
    *,
    max_features: int = 512,
    min_feature_count: int = 2,
    max_examples: int | None = None,
) -> ReplyRiskStats:
    feature_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    examples = 0
    for features, labels in _iter_examples(labels_path, max_examples=max_examples):
        examples += 1
        feature_counts.update(features.keys())
        for target, value in labels.items():
            target_counts[target] += int(value > 0.0)
    ranked = [(key, count) for key, count in feature_counts.items() if count >= min_feature_count]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    if max_features > 0:
        ranked = ranked[:max_features]
    names = [key for key, _count in ranked]
    if not names:
        raise ValueError(f"No usable reply-risk features found in {labels_path}")
    return ReplyRiskStats(
        feature_names=names,
        feature_counts={key: int(value) for key, value in feature_counts.items()},
        example_count=examples,
        target_positive_counts={key: int(target_counts.get(key, 0)) for key in TARGET_NAMES},
    )


class ReplyRiskModel:
    """Sparse multi-label logistic model for hidden capability and reply-risk targets."""

    def __init__(
        self,
        *,
        feature_names: list[str],
        targets: list[str] | None = None,
        weights: dict[str, list[float]] | None = None,
        biases: dict[str, float] | None = None,
        input_clip: float = 8.0,
    ):
        self.feature_names = [str(name) for name in feature_names]
        self.feature_index = {name: index for index, name in enumerate(self.feature_names)}
        self.targets = [str(target) for target in (targets or list(TARGET_NAMES))]
        self.weights = {
            target: [float(value) for value in (weights or {}).get(target, [0.0] * len(self.feature_names))]
            for target in self.targets
        }
        self.biases = {target: float((biases or {}).get(target, 0.0)) for target in self.targets}
        self.input_clip = float(input_clip)

    @classmethod
    def initialize(cls, stats: ReplyRiskStats, *, input_clip: float = 8.0) -> "ReplyRiskModel":
        return cls(feature_names=stats.feature_names, input_clip=input_clip)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ReplyRiskModel":
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError(f"Not an {MODEL_TYPE} payload")
        return cls(
            feature_names=list(payload["feature_names"]),
            targets=list(payload.get("targets", TARGET_NAMES)),
            weights={str(key): list(value) for key, value in payload["weights"].items()},
            biases={str(key): float(value) for key, value in payload["biases"].items()},
            input_clip=float(payload.get("input_clip", 8.0)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ReplyRiskModel":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_payload(json.load(fh))

    def to_payload(self, summary: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_type": MODEL_TYPE,
            "feature_names": self.feature_names,
            "targets": self.targets,
            "weights": self.weights,
            "biases": self.biases,
            "input_clip": self.input_clip,
            "summary": summary or {},
        }

    def save(self, path: str | Path, summary: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self.to_payload(summary=summary), fh, indent=2, sort_keys=True)

    def active_vector_items(self, features: dict[str, float]) -> list[tuple[int, float]]:
        items: list[tuple[int, float]] = []
        for key, value in features.items():
            index = self.feature_index.get(str(key))
            if index is None:
                continue
            number = max(-self.input_clip, min(self.input_clip, float(value)))
            if number:
                items.append((index, number))
        return items

    def predict_proba(self, features: dict[str, float]) -> dict[str, float]:
        items = self.active_vector_items(features)
        output: dict[str, float] = {}
        for target in self.targets:
            score = self.biases[target]
            row = self.weights[target]
            for index, value in items:
                score += row[index] * value
            output[target] = _sigmoid(score)
        return output

    def train_step(
        self,
        features: dict[str, float],
        labels: dict[str, float],
        *,
        lr: float,
        l2: float = 0.0,
        grad_clip: float = 5.0,
    ) -> float:
        items = self.active_vector_items(features)
        total_loss = 0.0
        for target in self.targets:
            label = 1.0 if labels.get(target, 0.0) > 0.0 else 0.0
            row = self.weights[target]
            score = self.biases[target]
            for index, value in items:
                score += row[index] * value
            prob = _sigmoid(score)
            total_loss += -(label * math.log(max(1e-12, prob)) + (1.0 - label) * math.log(max(1e-12, 1.0 - prob)))
            gradient = max(-grad_clip, min(grad_clip, prob - label))
            self.biases[target] -= lr * gradient
            for index, value in items:
                row[index] -= lr * ((gradient * value) + (l2 * row[index]))
        return total_loss / float(max(1, len(self.targets)))


def extract_reply_risk_features(state: AgentState, config: dict[str, Any]) -> dict[str, float]:
    my_active = state.my_active
    opp_active = state.opp_active
    features = {
        "bias": 1.0,
        "turn_frac": min(1.0, float(state.turn) / 16.0),
        "turn_action_frac": min(1.0, float(state.turn_action_count) / 20.0),
        "turn_early": 1.0 if state.turn <= 4 else 0.0,
        "turn_late": 1.0 if state.my_prizes_remaining <= 2 or state.opp_prizes_remaining <= 2 else 0.0,
        "actor_prizes_remaining": float(state.my_prizes_remaining) / 6.0,
        "opponent_prizes_remaining": float(state.opp_prizes_remaining) / 6.0,
        "prize_diff": float(max(-6, min(6, state.prize_diff))) / 6.0,
        "actor_hand_size": min(1.0, float(len(state.my_hand)) / 12.0),
        "actor_deck_count": min(1.0, float(state.my_deck_count) / 60.0),
        "opponent_hand_size": min(1.0, float(state.opp_hand_count) / 12.0),
        "opponent_deck_count": min(1.0, float(state.opp_deck_count) / 60.0),
        "actor_energy_in_play": min(1.0, float(state.my_energy_in_play) / 10.0),
        "opponent_energy_in_play": min(1.0, float(state.opp_energy_in_play) / 10.0),
        "actor_bench_count": min(1.0, float(len(state.my_bench)) / max(1, state.my_bench_max)),
        "opponent_bench_count": min(1.0, float(len(state.opp_bench)) / max(1, state.opp_bench_max)),
        "supporter_played": 1.0 if state.supporter_played else 0.0,
        "energy_attached": 1.0 if state.energy_attached else 0.0,
        "retreated": 1.0 if state.retreated else 0.0,
    }
    features.update(_pokemon_public_features("actor_active", my_active))
    features.update(_pokemon_public_features("opponent_active", opp_active))
    for pokemon in state.my_bench:
        features[f"actor_bench_id_{pokemon.id}"] = features.get(f"actor_bench_id_{pokemon.id}", 0.0) + 1.0
        features["actor_bench_energy"] = features.get("actor_bench_energy", 0.0) + _norm(pokemon.energy_count, 5.0)
    for pokemon in state.opp_bench:
        features[f"opponent_bench_id_{pokemon.id}"] = features.get(f"opponent_bench_id_{pokemon.id}", 0.0) + 1.0
        features["opponent_bench_energy"] = features.get("opponent_bench_energy", 0.0) + _norm(pokemon.energy_count, 5.0)
    for card in state.my_discard:
        features[f"actor_discard_id_{card.id}"] = features.get(f"actor_discard_id_{card.id}", 0.0) + 1.0
    for card in state.opp_discard:
        features[f"opponent_discard_id_{card.id}"] = features.get(f"opponent_discard_id_{card.id}", 0.0) + 1.0
    for key in ("main_attackers", "main_line", "backup_attackers"):
        ids = _ids(config, key)
        if my_active and my_active.id in ids:
            features[f"actor_active_{key}"] = 1.0
        if opp_active and opp_active.id in ids:
            features[f"opponent_active_{key}"] = 1.0
    return features


def _pokemon_public_features(prefix: str, pokemon: PokemonView | None) -> dict[str, float]:
    if pokemon is None:
        return {f"{prefix}_missing": 1.0}
    return {
        f"{prefix}_id_{pokemon.id}": 1.0,
        f"{prefix}_hp_frac": _norm(pokemon.hp, max(1, pokemon.max_hp)),
        f"{prefix}_damage_frac": _norm(pokemon.damage, max(1, pokemon.max_hp)),
        f"{prefix}_energy": _norm(pokemon.energy_count, 5.0),
        f"{prefix}_prize_value": _norm(pokemon.prize_value, 3.0),
        f"{prefix}_is_ex": 1.0 if pokemon.is_ex else 0.0,
        f"{prefix}_is_mega_ex": 1.0 if pokemon.is_mega_ex else 0.0,
        f"{prefix}_is_basic": 1.0 if pokemon.is_basic else 0.0,
        f"{prefix}_is_stage1": 1.0 if pokemon.is_stage1 else 0.0,
        f"{prefix}_is_stage2": 1.0 if pokemon.is_stage2 else 0.0,
    }


def reply_risk_labels_from_state_and_actions(
    state: AgentState,
    selected_actions: list[ActionView],
    config: dict[str, Any],
) -> dict[str, float]:
    hand_ids = Counter(card.id for card in state.my_hand)
    main_attackers = _ids(config, "main_attackers")
    key_evolutions = _ids(config, "key_evolutions")
    labels = {target: 0.0 for target in TARGET_NAMES}
    labels["hand_has_gust"] = float(_has_group(hand_ids, _ids(config, "gust_cards")))
    labels["hand_has_switch"] = float(_has_named_or_group(hand_ids, _ids(config, "switch_cards"), ("switch", "cart", "rope")))
    labels["hand_has_energy"] = float(any(is_energy(card_id) or card_id in _ids(config, "energy_cards") for card_id in hand_ids))
    labels["hand_has_search"] = float(_has_group(hand_ids, _ids(config, "search_cards")))
    labels["hand_has_draw"] = float(_has_group(hand_ids, _ids(config, "draw_cards")))
    labels["hand_has_evolution"] = float(_has_group(hand_ids, key_evolutions))
    labels["hand_has_main_attacker"] = float(_has_group(hand_ids, main_attackers))

    action_feature_dicts = [extract_action_features(state, action, config) for action in selected_actions]
    for action, features in zip(selected_actions, action_feature_dicts):
        labels["selected_attack"] = max(labels["selected_attack"], float(action.is_attack))
        labels["selected_attach"] = max(labels["selected_attach"], float(action.is_attach))
        labels["selected_evolve"] = max(labels["selected_evolve"], float(action.is_evolve))
        labels["selected_ability"] = max(labels["selected_ability"], float(action.is_ability))
        labels["selected_retreat"] = max(labels["selected_retreat"], float(action.option_type == "RETREAT"))
        labels["selected_end"] = max(labels["selected_end"], float(action.is_end))
        labels["selected_gust"] = max(labels["selected_gust"], float(features.get("play_gust", 0.0) > 0.0))
        labels["selected_search"] = max(labels["selected_search"], float(features.get("play_search", 0.0) > 0.0 or features.get("select_search_card", 0.0) > 0.0))
        labels["selected_draw"] = max(labels["selected_draw"], float(features.get("play_draw", 0.0) > 0.0 or features.get("select_draw_card", 0.0) > 0.0))
        labels["selected_switch"] = max(labels["selected_switch"], float(_card_name_contains(action.source_card_id, ("switch", "cart", "rope"))))
        labels["takes_ko"] = max(labels["takes_ko"], float(features.get("attack_takes_ko", 0.0) > 0.0))
        labels["wins_game"] = max(labels["wins_game"], float(features.get("attack_wins_game", 0.0) > 0.0))
        labels["selects_setup_target"] = max(labels["selects_setup_target"], float(features.get("select_opponent_setup_target", 0.0) > 0.0))
        labels["selects_must_answer_target"] = max(labels["selects_must_answer_target"], float(features.get("select_must_answer_target", 0.0) > 0.0))
    return labels


def _has_group(counts: Counter[int], card_ids: set[int]) -> bool:
    return any(counts.get(card_id, 0) > 0 for card_id in card_ids)


def _has_named_or_group(counts: Counter[int], card_ids: set[int], parts: tuple[str, ...]) -> bool:
    return _has_group(counts, card_ids) or any(_card_name_contains(card_id, parts) for card_id in counts)


def _card_name_contains(card_id: int | None, parts: tuple[str, ...]) -> bool:
    data = card_data(card_id)
    name = (getattr(data, "name", "") or "").lower() if data else ""
    return any(part in name for part in parts)


def train_reply_risk_model(
    labels_path: str | Path,
    *,
    out_dir: str = "artifacts/checkpoints/reply_risk",
    epochs: int = 5,
    lr: float = 0.01,
    l2: float = 1e-6,
    max_features: int = 512,
    min_feature_count: int = 2,
    max_examples: int | None = None,
    input_clip: float = 8.0,
    grad_clip: float = 5.0,
) -> dict[str, Any]:
    stats = build_reply_risk_stats(
        labels_path,
        max_features=max_features,
        min_feature_count=min_feature_count,
        max_examples=max_examples,
    )
    model = ReplyRiskModel.initialize(stats, input_clip=input_clip)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    latest_path = out_path / "latest.json"
    best_path = out_path / "best.json"
    best_loss = math.inf
    summary: dict[str, Any] = {}
    for epoch in range(1, max(1, int(epochs)) + 1):
        examples = 0
        total_loss = 0.0
        correct = {target: 0 for target in model.targets}
        positives = {target: 0 for target in model.targets}
        for features, labels in _iter_examples(labels_path, max_examples=max_examples):
            probs = model.predict_proba(features)
            loss = model.train_step(features, labels, lr=lr, l2=l2, grad_clip=grad_clip)
            examples += 1
            total_loss += loss
            for target in model.targets:
                label = labels.get(target, 0.0) > 0.0
                prediction = probs.get(target, 0.0) > 0.5
                correct[target] += int(label == prediction)
                positives[target] += int(label)
        average_loss = total_loss / examples if examples else 0.0
        summary = {
            "epoch": epoch,
            "labels": str(labels_path),
            "out_dir": str(out_path),
            "examples": examples,
            "feature_count": len(model.feature_names),
            "target_positive_counts": positives,
            "target_accuracy": {target: correct[target] / examples if examples else 0.0 for target in model.targets},
            "average_loss": average_loss,
            "lr": lr,
            "l2": l2,
            "max_features": max_features,
            "min_feature_count": min_feature_count,
        }
        model.save(out_path / f"checkpoint_epoch_{epoch}.json", summary=summary)
        model.save(latest_path, summary=summary)
        if average_loss <= best_loss:
            best_loss = average_loss
            model.save(best_path, summary=summary)
    summary["latest_checkpoint"] = str(latest_path)
    summary["best_checkpoint"] = str(best_path)
    print(json.dumps(summary, indent=2))
    return summary
