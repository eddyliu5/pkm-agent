from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MODEL_TYPE = "mlp_sequence_ranker"
SCHEMA_VERSION = 2
DEFAULT_EXCLUDED_FEATURES = ("linear_score",)
FEATURE_SCHEMA_ALIASES = {
    "sequence_v3_full": "full",
    "sequence_v3_clean": "clean",
}


def canonical_feature_schema_version(value: str) -> str:
    schema = str(value)
    return FEATURE_SCHEMA_ALIASES.get(schema, schema)


@dataclass(frozen=True)
class FeatureStats:
    names: list[str]
    mean: list[float]
    scale: list[float]
    feature_counts: dict[str, int]
    example_count: int
    candidate_count: int


def _clean_features(features: Any, *, excluded_features: tuple[str, ...] = DEFAULT_EXCLUDED_FEATURES) -> dict[str, float]:
    if not isinstance(features, dict):
        return {}
    excluded = set(excluded_features)
    cleaned: dict[str, float] = {}
    for key, value in features.items():
        key = str(key)
        if key in excluded:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number and math.isfinite(number):
            cleaned[key] = number
    return cleaned


def _stable_unit_interval(value: str, *, seed: int) -> float:
    state = (int(seed) & 0xFFFFFFFF) or 1
    for char in value:
        state = ((state * 16777619) ^ ord(char)) & 0xFFFFFFFF
    return state / float(0xFFFFFFFF)


def _use_validation_split(index: int, *, validation_fraction: float, seed: int, key: str | None = None) -> bool:
    if validation_fraction <= 0.0:
        return False
    if validation_fraction >= 1.0:
        return True
    if key:
        return _stable_unit_interval(key, seed=seed) < validation_fraction
    rng = random.Random((int(seed) * 1_000_003) + index)
    return rng.random() < validation_fraction


def _iter_label_examples(
    labels_path: str | Path,
    max_examples: int | None = None,
    *,
    excluded_features: tuple[str, ...] = DEFAULT_EXCLUDED_FEATURES,
    split: str | None = None,
    validation_fraction: float = 0.0,
    seed: int = 13,
) -> Iterable[tuple[int, list[dict[str, float]]]]:
    seen = 0
    with Path(labels_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            candidates = record.get("candidates") or []
            if not isinstance(candidates, list) or not candidates:
                continue
            try:
                label_index = int(record.get("label_index", -1))
            except (TypeError, ValueError):
                continue
            if label_index < 0 or label_index >= len(candidates):
                continue
            if not bool(record.get("matched", label_index >= 0)):
                continue
            split_key = None
            if isinstance(record, dict):
                source = str(record.get("source", "") or "")
                episode_id = str(record.get("episode_id", "") or "")
                if source or episode_id:
                    split_key = f"{source}:{episode_id}"
            is_validation = _use_validation_split(seen, validation_fraction=validation_fraction, seed=seed, key=split_key)
            seen += 1
            if max_examples is not None and seen > max_examples:
                break
            if split == "train" and is_validation:
                continue
            if split == "validation" and not is_validation:
                continue
            candidate_features = [
                _clean_features(
                    candidate.get("features") if isinstance(candidate, dict) else {},
                    excluded_features=excluded_features,
                )
                for candidate in candidates
            ]
            if not candidate_features:
                continue
            yield label_index, candidate_features


def build_feature_stats(
    labels_path: str | Path,
    *,
    max_features: int = 256,
    min_feature_count: int = 1,
    max_examples: int | None = None,
    excluded_features: tuple[str, ...] = DEFAULT_EXCLUDED_FEATURES,
    validation_fraction: float = 0.0,
    seed: int = 13,
) -> FeatureStats:
    counts: Counter[str] = Counter()
    example_count = 0
    candidate_count = 0
    for _label_index, candidates in _iter_label_examples(
        labels_path,
        max_examples=max_examples,
        excluded_features=excluded_features,
        split="train",
        validation_fraction=validation_fraction,
        seed=seed,
    ):
        example_count += 1
        candidate_count += len(candidates)
        for features in candidates:
            counts.update(features.keys())

    ranked = [(key, count) for key, count in counts.items() if count >= min_feature_count]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    if max_features > 0:
        ranked = ranked[:max_features]
    names = [key for key, _count in ranked]
    if not names:
        raise ValueError(f"No usable candidate features found in {labels_path}")

    selected = set(names)
    sums = {name: 0.0 for name in names}
    sums_sq = {name: 0.0 for name in names}
    vector_count = 0
    for _label_index, candidates in _iter_label_examples(
        labels_path,
        max_examples=max_examples,
        excluded_features=excluded_features,
        split="train",
        validation_fraction=validation_fraction,
        seed=seed,
    ):
        for features in candidates:
            vector_count += 1
            for key, value in features.items():
                if key in selected:
                    sums[key] += value
                    sums_sq[key] += value * value

    if vector_count <= 0:
        raise ValueError(f"No matched label examples found in {labels_path}")

    mean: list[float] = []
    scale: list[float] = []
    for name in names:
        avg = sums[name] / float(vector_count)
        variance = max(0.0, (sums_sq[name] / float(vector_count)) - (avg * avg))
        std = math.sqrt(variance)
        mean.append(avg)
        scale.append(std if std >= 1e-6 else 1.0)

    return FeatureStats(
        names=names,
        mean=mean,
        scale=scale,
        feature_counts={key: int(value) for key, value in counts.items()},
        example_count=example_count,
        candidate_count=candidate_count,
    )


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    best = max(logits)
    values = [math.exp(max(-40.0, min(40.0, value - best))) for value in logits]
    total = sum(values) or 1.0
    return [value / total for value in values]


def _clip(value: float, limit: float) -> float:
    if limit <= 0:
        return value
    return max(-limit, min(limit, value))


def _default_hidden_dim(input_dim: int) -> int:
    if input_dim <= 128:
        return 128
    if input_dim <= 512:
        return 192
    return 256


def _evaluate_ranker(
    model: "MLPSequenceRanker",
    labels_path: str | Path,
    *,
    max_examples: int | None,
    excluded_features: tuple[str, ...],
    split: str | None,
    validation_fraction: float,
    seed: int,
    temperature: float,
) -> dict[str, float]:
    examples = 0
    total_loss = 0.0
    top1 = 0
    top3 = 0
    for label_index, candidates in _iter_label_examples(
        labels_path,
        max_examples=max_examples,
        excluded_features=excluded_features,
        split=split,
        validation_fraction=validation_fraction,
        seed=seed,
    ):
        scores = model.score_batch(candidates)
        probs = _softmax([score / max(1e-6, temperature) for score in scores])
        if not probs:
            continue
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        examples += 1
        total_loss -= math.log(max(1e-12, probs[label_index]))
        top1 += int(order[0] == label_index)
        top3 += int(label_index in order[:3])
    return {
        "examples": float(examples),
        "average_loss": total_loss / examples if examples else 0.0,
        "top1_accuracy": top1 / examples if examples else 0.0,
        "top3_accuracy": top3 / examples if examples else 0.0,
    }


class MLPSequenceRanker:
    """Tiny dependency-free MLP that scores sequence feature dictionaries."""

    def __init__(
        self,
        *,
        feature_names: list[str],
        input_mean: list[float],
        input_scale: list[float],
        hidden_dim: int | None = None,
        w1: list[list[float]] | None = None,
        b1: list[float] | None = None,
        w2: list[float] | None = None,
        b2: float | None = None,
        hidden_layers: list[dict[str, Any]] | None = None,
        output_weights: list[float] | None = None,
        output_bias: float | None = None,
        input_clip: float = 8.0,
        activation: str = "relu",
        normalization: str = "standard",
        feature_schema_version: str = "full",
    ):
        if activation != "relu":
            raise ValueError(f"Unsupported activation: {activation}")
        if normalization not in {"standard", "sparse"}:
            raise ValueError(f"Unsupported normalization: {normalization}")
        if len(feature_names) != len(input_mean) or len(feature_names) != len(input_scale):
            raise ValueError("Feature names, means, and scales must have the same length")
        self.feature_names = [str(name) for name in feature_names]
        self.feature_index = {name: index for index, name in enumerate(self.feature_names)}
        self.input_mean = [float(value) for value in input_mean]
        self.input_scale = [float(value) if float(value) else 1.0 for value in input_scale]
        if hidden_layers is None:
            if hidden_dim is None or hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            if w1 is None or b1 is None or w2 is None or b2 is None:
                raise ValueError("Legacy MLP payload requires w1, b1, w2, and b2")
            hidden_layers = [{"weights": w1, "bias": b1}]
            output_weights = w2
            output_bias = b2
        if not hidden_layers:
            raise ValueError("At least one hidden layer is required")
        if output_weights is None or output_bias is None:
            raise ValueError("output_weights and output_bias are required")
        self.layer_weights = [
            [[float(value) for value in row] for row in layer["weights"]]
            for layer in hidden_layers
        ]
        self.layer_biases = [[float(value) for value in layer["bias"]] for layer in hidden_layers]
        self.hidden_dims = [len(bias) for bias in self.layer_biases]
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("Hidden layer dimensions must be positive")
        self.hidden_dim = self.hidden_dims[0]
        self.output_weights = [float(value) for value in output_weights]
        self.output_bias = float(output_bias)
        self.w1 = self.layer_weights[0]
        self.b1 = self.layer_biases[0]
        self.w2 = self.output_weights
        self.b2 = self.output_bias
        self.input_clip = float(input_clip)
        self.activation = activation
        self.normalization = normalization
        if not feature_schema_version:
            raise ValueError("MLP feature_schema_version is required")
        self.feature_schema_version = canonical_feature_schema_version(feature_schema_version)
        previous_dim = len(self.feature_names)
        for index, (weights, bias) in enumerate(zip(self.layer_weights, self.layer_biases)):
            if len(weights) != len(bias):
                raise ValueError(f"Hidden layer {index} weights/bias shape mismatch")
            if any(len(row) != previous_dim for row in weights):
                raise ValueError(f"Hidden layer {index} input shape mismatch")
            previous_dim = len(bias)
        if len(self.output_weights) != previous_dim:
            raise ValueError("Output layer parameter shape does not match final hidden layer")

    @classmethod
    def initialize(
        cls,
        *,
        feature_stats: FeatureStats,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        hidden_dim: int | None = None,
        hidden_dim2: int | None = None,
        seed: int = 13,
        input_clip: float = 8.0,
        normalization: str = "sparse",
        feature_schema_version: str = "full",
    ) -> "MLPSequenceRanker":
        rng = random.Random(seed)
        input_dim = len(feature_stats.names)
        if hidden_dims is None:
            if hidden_dim is None or hidden_dim <= 0:
                hidden_dim = _default_hidden_dim(input_dim)
            if hidden_dim2 is None or hidden_dim2 <= 0:
                hidden_dim2 = max(16, int(hidden_dim) // 2)
            hidden_dims = [int(hidden_dim), int(hidden_dim2)]
        hidden_dims = [int(dim) for dim in hidden_dims if int(dim) > 0]
        if not hidden_dims:
            raise ValueError("At least one hidden dimension is required")
        input_mean = [0.0 for _ in feature_stats.names] if normalization == "sparse" else feature_stats.mean
        hidden_layers: list[dict[str, Any]] = []
        previous_dim = input_dim
        for dim in hidden_dims:
            limit = math.sqrt(6.0 / float(max(1, previous_dim + dim)))
            hidden_layers.append(
                {
                    "weights": [[rng.uniform(-limit, limit) for _ in range(previous_dim)] for _ in range(dim)],
                    "bias": [0.0 for _ in range(dim)],
                }
            )
            previous_dim = dim
        output_limit = math.sqrt(6.0 / float(max(1, previous_dim + 1)))
        output_weights = [rng.uniform(-output_limit, output_limit) for _ in range(previous_dim)]
        return cls(
            feature_names=feature_stats.names,
            input_mean=input_mean,
            input_scale=feature_stats.scale,
            hidden_layers=hidden_layers,
            output_weights=output_weights,
            output_bias=0.0,
            input_clip=input_clip,
            normalization=normalization,
            feature_schema_version=feature_schema_version,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MLPSequenceRanker":
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError(f"Not an {MODEL_TYPE} payload")
        if "hidden_layers" in payload:
            return cls(
                feature_names=list(payload["feature_names"]),
                input_mean=list(payload["input_mean"]),
                input_scale=list(payload["input_scale"]),
                hidden_layers=[
                    {"weights": [list(row) for row in layer["weights"]], "bias": list(layer["bias"])}
                    for layer in payload["hidden_layers"]
                ],
                output_weights=list(payload["output_weights"]),
                output_bias=float(payload["output_bias"]),
                input_clip=float(payload.get("input_clip", 8.0)),
                activation=str(payload.get("activation", "relu")),
                normalization=str(payload.get("normalization", "standard")),
                feature_schema_version=str(payload["feature_schema_version"]),
            )
        return cls(
            feature_names=list(payload["feature_names"]),
            input_mean=list(payload["input_mean"]),
            input_scale=list(payload["input_scale"]),
            hidden_dim=int(payload["hidden_dim"]),
            w1=[list(row) for row in payload["w1"]],
            b1=list(payload["b1"]),
            w2=list(payload["w2"]),
            b2=float(payload["b2"]),
            input_clip=float(payload.get("input_clip", 8.0)),
            activation=str(payload.get("activation", "relu")),
            normalization=str(payload.get("normalization", "standard")),
            feature_schema_version=str(payload["feature_schema_version"]),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "MLPSequenceRanker":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_payload(json.load(fh))

    def to_payload(self, summary: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_type": MODEL_TYPE,
            "feature_schema_version": self.feature_schema_version,
            "feature_names": self.feature_names,
            "input_mean": self.input_mean,
            "input_scale": self.input_scale,
            "hidden_dim": self.hidden_dim,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "normalization": self.normalization,
            "input_clip": self.input_clip,
            "hidden_layers": [
                {"weights": weights, "bias": bias}
                for weights, bias in zip(self.layer_weights, self.layer_biases)
            ],
            "output_weights": self.output_weights,
            "output_bias": self.output_bias,
            "summary": summary or {},
        }

    def validate_feature_schema(self, expected: str, *, allow_legacy: bool = False) -> None:
        actual = canonical_feature_schema_version(self.feature_schema_version)
        expected = canonical_feature_schema_version(expected)
        if allow_legacy and actual == "legacy":
            return
        if actual != expected:
            raise ValueError(f"MLP feature schema mismatch: expected {expected}, got {actual}")

    def save(self, path: str | Path, summary: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self.to_payload(summary=summary), fh, indent=2, sort_keys=True)

    def vectorize(self, features: dict[str, float]) -> list[float]:
        values: list[float] = []
        for name, mean, scale in zip(self.feature_names, self.input_mean, self.input_scale):
            raw = float(features.get(name, 0.0))
            normalized = (raw - mean) / scale
            values.append(_clip(normalized, self.input_clip))
        return values

    def active_vector_items(self, features: dict[str, float]) -> list[tuple[int, float]]:
        if self.normalization != "sparse":
            return [(index, value) for index, value in enumerate(self.vectorize(features)) if value]
        items: list[tuple[int, float]] = []
        for key, value in features.items():
            index = self.feature_index.get(str(key))
            if index is None:
                continue
            normalized = _clip(float(value) / self.input_scale[index], self.input_clip)
            if normalized:
                items.append((index, normalized))
        return items

    def _forward_vector(self, vector: list[float]) -> tuple[float, list[list[float]], list[list[float]]]:
        return self._forward_dense_layers(vector)

    def _forward_items(self, items: list[tuple[int, float]]) -> tuple[float, list[list[float]], list[list[float]]]:
        first_pre: list[float] = []
        first_hidden: list[float] = []
        for row, bias in zip(self.layer_weights[0], self.layer_biases[0]):
            value = bias
            for index, item in items:
                value += row[index] * item
            first_pre.append(value)
            first_hidden.append(value if value > 0.0 else 0.0)
        pre_activations = [first_pre]
        activations = [first_hidden]
        current = first_hidden
        for weights, biases in zip(self.layer_weights[1:], self.layer_biases[1:]):
            current_pre: list[float] = []
            current_hidden: list[float] = []
            for row, bias in zip(weights, biases):
                value = bias
                for weight, item in zip(row, current):
                    value += weight * item
                current_pre.append(value)
                current_hidden.append(value if value > 0.0 else 0.0)
            pre_activations.append(current_pre)
            activations.append(current_hidden)
            current = current_hidden
        score = self.output_bias
        for weight, item in zip(self.output_weights, activations[-1]):
            score += weight * item
        return score, pre_activations, activations

    def _forward_dense_layers(self, vector: list[float]) -> tuple[float, list[list[float]], list[list[float]]]:
        pre_activations: list[list[float]] = []
        activations: list[list[float]] = []
        current = vector
        for weights, biases in zip(self.layer_weights, self.layer_biases):
            current_pre: list[float] = []
            current_hidden: list[float] = []
            for row, bias in zip(weights, biases):
                value = bias
                for weight, item in zip(row, current):
                    value += weight * item
                current_pre.append(value)
                current_hidden.append(value if value > 0.0 else 0.0)
            pre_activations.append(current_pre)
            activations.append(current_hidden)
            current = current_hidden
        score = self.output_bias
        for weight, item in zip(self.output_weights, activations[-1]):
            score += weight * item
        return score, pre_activations, activations

    def score_features(self, features: dict[str, float]) -> float:
        if self.normalization == "sparse":
            score, _pre_activation, _hidden = self._forward_items(self.active_vector_items(features))
            return score
        score, _pre_activation, _hidden = self._forward_vector(self.vectorize(features))
        return score

    def score_batch(self, candidates: list[dict[str, float]]) -> list[float]:
        return [self.score_features(features) for features in candidates]

    def train_step(
        self,
        candidates: list[dict[str, float]],
        label_index: int,
        *,
        lr: float,
        temperature: float = 1.0,
        l2: float = 0.0,
        grad_clip: float = 5.0,
        loss_scale: float = 1.0,
    ) -> tuple[float, bool, bool]:
        if label_index < 0 or label_index >= len(candidates):
            raise ValueError("label_index is out of range")
        temperature = max(1e-6, float(temperature))
        sparse = self.normalization == "sparse"
        vectors = [self.active_vector_items(features) if sparse else self.vectorize(features) for features in candidates]
        caches = [self._forward_items(vector) if sparse else self._forward_vector(vector) for vector in vectors]
        scores = [cache[0] for cache in caches]
        probabilities = _softmax([score / temperature for score in scores])
        loss = -math.log(max(1e-12, probabilities[label_index]))

        input_dim = len(self.feature_names)
        layer_count = len(self.layer_weights)
        if sparse:
            grad_first_sparse: list[dict[int, float]] = [{} for _ in range(self.hidden_dim)]
            grad_first_dense: list[list[float]] = []
        else:
            grad_first_dense = [[0.0 for _ in range(input_dim)] for _ in range(self.hidden_dim)]
            grad_first_sparse = []
        grad_layer_weights: list[list[list[float]]] = [grad_first_dense]
        for layer_index in range(1, layer_count):
            previous_dim = self.hidden_dims[layer_index - 1]
            grad_layer_weights.append([[0.0 for _ in range(previous_dim)] for _ in range(self.hidden_dims[layer_index])])
        grad_layer_biases = [[0.0 for _ in range(dim)] for dim in self.hidden_dims]
        grad_output_weights = [0.0 for _ in self.output_weights]
        grad_output_bias = 0.0
        old_layer_weights = [
            [[value for value in row] for row in weights]
            for weights in self.layer_weights
        ]
        old_output_weights = list(self.output_weights)

        for cand_index, (vector, (_score, pre_activations, activations)) in enumerate(zip(vectors, caches)):
            dscore = loss_scale * (probabilities[cand_index] - (1.0 if cand_index == label_index else 0.0)) / temperature
            grad_output_bias += dscore
            dactivation = [0.0 for _ in activations[-1]]
            for hidden_index, hidden_value in enumerate(activations[-1]):
                grad_output_weights[hidden_index] += dscore * hidden_value
                dactivation[hidden_index] += dscore * old_output_weights[hidden_index]

            for layer_index in range(layer_count - 1, -1, -1):
                dpre = [
                    dactivation[index] if pre_activations[layer_index][index] > 0.0 else 0.0
                    for index in range(self.hidden_dims[layer_index])
                ]
                for hidden_index, value in enumerate(dpre):
                    grad_layer_biases[layer_index][hidden_index] += value
                if layer_index == 0:
                    if sparse:
                        for hidden_index, value in enumerate(dpre):
                            if not value:
                                continue
                            row_sparse = grad_first_sparse[hidden_index]
                            for input_index, input_value in vector:
                                row_sparse[input_index] = row_sparse.get(input_index, 0.0) + value * input_value
                    else:
                        input_vector = vector
                        for hidden_index, value in enumerate(dpre):
                            if not value:
                                continue
                            row = grad_layer_weights[0][hidden_index]
                            for input_index, input_value in enumerate(input_vector):
                                row[input_index] += value * input_value
                else:
                    input_vector = activations[layer_index - 1]
                    for hidden_index, value in enumerate(dpre):
                        if not value:
                            continue
                        row = grad_layer_weights[layer_index][hidden_index]
                        for input_index, input_value in enumerate(input_vector):
                            row[input_index] += value * input_value

                if layer_index > 0:
                    previous_dim = self.hidden_dims[layer_index - 1]
                    next_dactivation = [0.0 for _ in range(previous_dim)]
                    for hidden_index, value in enumerate(dpre):
                        if not value:
                            continue
                        for input_index, weight in enumerate(old_layer_weights[layer_index][hidden_index]):
                            next_dactivation[input_index] += value * weight
                    dactivation = next_dactivation

        for index, gradient in enumerate(grad_output_weights):
            self.output_weights[index] -= lr * (_clip(gradient, grad_clip) + l2 * self.output_weights[index])
        self.output_bias -= lr * _clip(grad_output_bias, grad_clip)
        self.w2 = self.output_weights
        self.b2 = self.output_bias

        for layer_index in range(layer_count):
            for hidden_index in range(self.hidden_dims[layer_index]):
                self.layer_biases[layer_index][hidden_index] -= lr * _clip(grad_layer_biases[layer_index][hidden_index], grad_clip)
                row = self.layer_weights[layer_index][hidden_index]
                if layer_index == 0 and sparse:
                    for input_index, gradient in grad_first_sparse[hidden_index].items():
                        row[input_index] -= lr * (_clip(gradient, grad_clip) + l2 * row[input_index])
                else:
                    grad_row = grad_layer_weights[layer_index][hidden_index]
                    for input_index, gradient in enumerate(grad_row):
                        row[input_index] -= lr * (_clip(gradient, grad_clip) + l2 * row[input_index])

        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return loss, order[0] == label_index, label_index in order[:3]


def pretrain_mlp_ranker(
    labels_path: str | Path,
    *,
    out_dir: str = "artifacts/checkpoints/mlp_ranker",
    epochs: int = 3,
    lr: float = 0.001,
    hidden_dim: int = 0,
    hidden_dim2: int = 0,
    max_features: int = 256,
    min_feature_count: int = 1,
    max_examples: int | None = None,
    seed: int = 13,
    temperature: float = 1.0,
    l2: float = 1e-5,
    input_clip: float = 8.0,
    grad_clip: float = 5.0,
    normalization: str = "sparse",
    feature_schema_version: str = "full",
    validation_fraction: float = 0.10,
    exclude_features: tuple[str, ...] = DEFAULT_EXCLUDED_FEATURES,
) -> dict[str, Any]:
    feature_schema_version = canonical_feature_schema_version(feature_schema_version)
    validation_fraction = max(0.0, min(0.9, float(validation_fraction)))
    stats = build_feature_stats(
        labels_path,
        max_features=max_features,
        min_feature_count=min_feature_count,
        max_examples=max_examples,
        excluded_features=exclude_features,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    hidden_dim = int(hidden_dim) if int(hidden_dim) > 0 else _default_hidden_dim(len(stats.names))
    hidden_dim2 = int(hidden_dim2) if int(hidden_dim2) > 0 else max(16, hidden_dim // 2)
    model = MLPSequenceRanker.initialize(
        feature_stats=stats,
        hidden_dim=hidden_dim,
        hidden_dim2=hidden_dim2,
        seed=seed,
        input_clip=input_clip,
        normalization=normalization,
        feature_schema_version=feature_schema_version,
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    latest_path = out_path / "latest.json"
    best_path = out_path / "best.json"
    best_score = -1.0
    summary: dict[str, Any] = {}

    for epoch in range(1, max(1, int(epochs)) + 1):
        examples = 0
        total_loss = 0.0
        top1 = 0
        top3 = 0
        for label_index, candidates in _iter_label_examples(
            labels_path,
            max_examples=max_examples,
            excluded_features=exclude_features,
            split="train",
            validation_fraction=validation_fraction,
            seed=seed,
        ):
            loss, is_top1, is_top3 = model.train_step(
                candidates,
                label_index,
                lr=lr,
                temperature=temperature,
                l2=l2,
                grad_clip=grad_clip,
            )
            examples += 1
            total_loss += loss
            top1 += int(is_top1)
            top3 += int(is_top3)

        train_metrics = {
            "examples": examples,
            "average_loss": total_loss / examples if examples else 0.0,
            "top1_accuracy": top1 / examples if examples else 0.0,
            "top3_accuracy": top3 / examples if examples else 0.0,
        }
        validation_metrics = _evaluate_ranker(
            model,
            labels_path,
            max_examples=max_examples,
            excluded_features=exclude_features,
            split="validation",
            validation_fraction=validation_fraction,
            seed=seed,
            temperature=temperature,
        ) if validation_fraction > 0.0 else {"examples": 0.0, "average_loss": 0.0, "top1_accuracy": 0.0, "top3_accuracy": 0.0}
        selection_metric = (
            validation_metrics["top1_accuracy"]
            if validation_metrics["examples"] > 0
            else train_metrics["top1_accuracy"]
        )
        summary = {
            "epoch": epoch,
            "labels": str(labels_path),
            "out_dir": str(out_path),
            "examples": examples,
            "candidate_vectors_for_stats": stats.candidate_count,
            "feature_count": len(stats.names),
            "hidden_dim": hidden_dim,
            "hidden_dim2": hidden_dim2,
            "hidden_dims": [hidden_dim, hidden_dim2],
            "normalization": normalization,
            "average_loss": train_metrics["average_loss"],
            "top1_accuracy": train_metrics["top1_accuracy"],
            "top3_accuracy": train_metrics["top3_accuracy"],
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_fraction": validation_fraction,
            "selection_metric": "validation_top1_accuracy" if validation_metrics["examples"] > 0 else "train_top1_accuracy",
            "selection_score": selection_metric,
            "lr": lr,
            "temperature": temperature,
            "l2": l2,
            "excluded_features": list(exclude_features),
            "feature_schema_version": feature_schema_version,
        }
        model.save(out_path / f"checkpoint_epoch_{epoch}.json", summary=summary)
        model.save(latest_path, summary=summary)
        if selection_metric >= best_score:
            best_score = selection_metric
            model.save(best_path, summary=summary)

    summary["latest_checkpoint"] = str(latest_path)
    summary["best_checkpoint"] = str(best_path)
    return summary
