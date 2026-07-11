from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import struct
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


SCHEMA_VERSION = 1
MODEL_TYPE = "xgb_immediate_root_ranker"
FEATURE_SCHEMA_VERSION = "full"
COMPACT_MODEL_TYPE = "compact_xgb_immediate_root_ranker"
COMPACT_SCHEMA_VERSION = 1
SPLIT_COUNTS = {"train": 160, "validation": 40, "test": 40}
MANIFEST_SEED = 13
MANIFEST_HASH_PREFIX = "xgb-root-v1:"
BOOTSTRAP_REPLICATES = 2_000
MIN_FEATURE_DECISIONS = 10
MAX_FEATURES = 512
MIN_EVAL_GROUPS = 2_000
MIN_EVAL_EPISODES = 25
LABEL_BUDGETS = {
    "beam_width": 8,
    "max_depth": 8,
    "max_nodes": 128,
    "time_budget_ms": 500,
}
GENERATED_FEATURES = frozenset(
    {
        "linear_score",
        "mlp_score",
        "xgb_score",
        "value_model_score",
        "risk_model_score",
        "plan_bonus_score",
        "opponent_counterplay_score",
        "hidden_root_mean_score",
        "hidden_root_min_score",
        "hidden_root_max_score",
        "hidden_root_score_std",
        "hidden_root_mean_value_score",
    }
)
GENERATED_PREFIXES = (
    "xgb_",
    "structured_critique_",
    "critique_",
    "opponent_response_mcts_",
)
XGB_PARAMS: dict[str, Any] = {
    "objective": "rank:ndcg",
    "eval_metric": ["ndcg@1", "ndcg@3"],
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 10.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 5.0,
    "alpha": 0.1,
    "lambdarank_pair_method": "topk",
    "lambdarank_num_pair_per_sample": 8,
    "tree_method": "hist",
    "seed": MANIFEST_SEED,
    "nthread": 1,
}
NUM_BOOST_ROUND = 800
EARLY_STOPPING_ROUNDS = 50
INFERENCE_FEATURE_COUNT = 389


@dataclass(frozen=True)
class RootGroup:
    episode_id: str
    group_id: str
    label_index: int
    features: tuple[dict[str, float], ...]
    base_scores: tuple[float, ...]

    @property
    def size(self) -> int:
        return len(self.features)


@dataclass(frozen=True)
class RootGroupInfo:
    episode_id: str
    group_id: str
    label_index: int
    size: int
    base_scores: tuple[float, ...]


@dataclass(frozen=True)
class MatrixData:
    matrix: Any
    labels: Any
    group_sizes: tuple[int, ...]
    groups: tuple[RootGroupInfo, ...]


@dataclass(frozen=True)
class LabelScan:
    records: int
    passive_end_groups: int
    eligible_records: int
    matched_records: int
    unmatched_records: int
    single_root_groups: int
    single_root_matched: int
    usable_groups: int
    episodes: int
    usable_episodes: int
    eligible_by_episode: dict[str, int]
    single_matched_by_episode: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "passive_end_groups": self.passive_end_groups,
            "eligible_records": self.eligible_records,
            "matched_records": self.matched_records,
            "unmatched_records": self.unmatched_records,
            "single_root_groups": self.single_root_groups,
            "single_root_matched": self.single_root_matched,
            "usable_groups": self.usable_groups,
            "episodes": self.episodes,
            "usable_episodes": self.usable_episodes,
            "natural_root_recall": self.matched_records / self.eligible_records if self.eligible_records else 0.0,
            "usable_group_rate": self.usable_groups / self.eligible_records if self.eligible_records else 0.0,
        }


def _scored_sequence(value: Any) -> Any:
    return getattr(value, "candidate", value)


def _candidate_path_key(value: Any) -> tuple[tuple[int, ...], ...]:
    sequence = _scored_sequence(value)
    steps = tuple(getattr(sequence, "steps", ()) or ())
    path: list[tuple[int, ...]] = []
    for step in steps:
        selected = getattr(step, "selected_indices", ())
        if not isinstance(selected, (list, tuple)):
            raise ValueError("Candidate step selected_indices must be a list or tuple")
        try:
            path.append(tuple(int(index) for index in selected))
        except (TypeError, ValueError) as exc:
            raise ValueError("Candidate path contains a non-integer index") from exc
    return tuple(path)


def _candidate_root_key(value: Any) -> tuple[int, ...]:
    sequence = _scored_sequence(value)
    selected = getattr(sequence, "first_selection", None)
    if selected is None:
        raise ValueError("Scored candidate has no first_selection")
    try:
        root = tuple(sorted(int(index) for index in selected))
    except (TypeError, ValueError) as exc:
        raise ValueError("Scored candidate root contains a non-integer index") from exc
    if len(root) != len(set(root)):
        raise ValueError("Scored candidate root contains duplicate indices")
    return root


def _base_candidate_key(value: Any) -> tuple[Any, ...]:
    try:
        score = float(getattr(value, "score"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Candidate must have a numeric base score") from exc
    if not math.isfinite(score):
        raise ValueError("Candidate base score must be finite")
    sequence = _scored_sequence(value)
    steps = tuple(getattr(sequence, "steps", ()) or ())
    path = _candidate_path_key(value)
    generation_tags = tuple(str(tag) for tag in (getattr(sequence, "generation_tags", ()) or ()))
    hidden_sample_id = str(getattr(sequence, "hidden_sample_id", "") or "")
    return (-score, len(steps), path, generation_tags, hidden_sample_id)


def dedupe_scored_roots(scored: Sequence[Any]) -> list[Any]:
    """Return one deterministic base-scored representative per canonical root."""

    by_root: dict[tuple[int, ...], Any] = {}
    for candidate in scored:
        root = _candidate_root_key(candidate)
        existing = by_root.get(root)
        if existing is None or _base_candidate_key(candidate) < _base_candidate_key(existing):
            by_root[root] = candidate
    return sorted(by_root.values(), key=lambda candidate: (*_base_candidate_key(candidate), _candidate_root_key(candidate)))


class XGBRootRanker:
    """Strict local inference wrapper for the native root-ranker Booster."""

    def __init__(
        self,
        *,
        booster: Any,
        metadata: dict[str, Any],
        feature_names: tuple[str, ...],
        best_iteration: int,
        model_dir: Path,
    ):
        self.booster = booster
        self.metadata = metadata
        self.feature_schema_version = str(metadata.get("feature_schema_version", ""))
        self.feature_names = feature_names
        self.best_iteration = int(best_iteration)
        self.model_dir = model_dir
        self._feature_index = {name: index for index, name in enumerate(feature_names)}

    def validate_feature_schema(self, expected: str) -> None:
        expected_schema = str(expected).strip().lower()
        actual_schema = self.feature_schema_version.strip().lower()
        if expected_schema != actual_schema:
            raise ValueError(
                f"XGBoost feature schema mismatch: model={actual_schema!r}, evaluator={expected_schema!r}"
            )

    @staticmethod
    def _load_xgboost() -> Any:
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover - depends on the local runtime.
            raise RuntimeError("XGBRootRanker inference requires xgboost") from exc
        return xgb

    @classmethod
    def from_directory(cls, path: str | Path) -> "XGBRootRanker":
        directory = Path(path)
        if not directory.is_dir():
            raise FileNotFoundError(f"XGBoost model directory does not exist: {directory}")
        return cls._from_metadata(directory / "metadata.json")

    @classmethod
    def from_dir(cls, path: str | Path) -> "XGBRootRanker":
        return cls.from_directory(path)

    @classmethod
    def from_file(cls, path: str | Path) -> "XGBRootRanker":
        model_path = Path(path)
        if model_path.is_dir():
            return cls.from_directory(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"XGBoost model file does not exist: {model_path}")
        if model_path.name == "metadata.json" or model_path.name.endswith(".metadata.json"):
            return cls._from_metadata(model_path)
        return cls._from_metadata(model_path.parent / "metadata.json", expected_booster=model_path)

    @classmethod
    def _from_metadata(
        cls,
        metadata_path: Path,
        *,
        expected_booster: Path | None = None,
    ) -> "XGBRootRanker":
        if not metadata_path.is_file():
            raise FileNotFoundError(f"XGBoost metadata does not exist: {metadata_path}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("XGBoost metadata must be a JSON object")
        if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported XGBoost root-ranker schema: {payload.get('schema_version')}")
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError(f"Unexpected XGBoost model type: {payload.get('model_type')!r}")
        feature_schema_version = payload.get("feature_schema_version")
        if feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "XGBoost metadata feature_schema_version must be "
                f"{FEATURE_SCHEMA_VERSION!r}; found {feature_schema_version!r}"
            )
        params = payload.get("params")
        if not isinstance(params, dict) or params.get("objective") != "rank:ndcg":
            raise ValueError("XGBoost metadata objective must be rank:ndcg")

        raw_names = payload.get("feature_names")
        if not isinstance(raw_names, list) or not all(isinstance(name, str) and name for name in raw_names):
            raise ValueError("XGBoost metadata feature_names must be non-empty strings")
        feature_names = tuple(raw_names)
        if len(feature_names) != INFERENCE_FEATURE_COUNT:
            raise ValueError(
                f"XGBoost root ranker requires exactly {INFERENCE_FEATURE_COUNT} features; found {len(feature_names)}"
            )
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("XGBoost metadata contains duplicate feature names")
        defaults = payload.get("feature_defaults")
        if not isinstance(defaults, dict) or set(defaults) != set(feature_names):
            raise ValueError("XGBoost feature_defaults must exactly match feature_names")
        for name in feature_names:
            try:
                default = float(defaults[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid feature default for {name!r}") from exc
            if not math.isfinite(default) or default != 0.0:
                raise ValueError(f"Feature {name!r} must have a finite zero default")

        booster_name = payload.get("booster_file")
        if not isinstance(booster_name, str) or not booster_name or Path(booster_name).name != booster_name:
            raise ValueError("XGBoost booster_file must be a local file name")
        booster_path = metadata_path.parent / booster_name
        if expected_booster is not None and booster_path.resolve() != expected_booster.resolve():
            raise ValueError(f"Metadata points to {booster_path}, not requested Booster {expected_booster}")
        if not booster_path.is_file():
            raise FileNotFoundError(f"XGBoost Booster does not exist: {booster_path}")
        expected_hash = payload.get("booster_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError("XGBoost metadata has no valid booster_sha256")
        actual_hash = _sha256_path(booster_path)
        if actual_hash != expected_hash:
            raise ValueError(f"XGBoost Booster hash mismatch: expected {expected_hash}, got {actual_hash}")

        try:
            best_iteration = int(payload.get("best_iteration"))
        except (TypeError, ValueError) as exc:
            raise ValueError("XGBoost metadata best_iteration must be an integer") from exc
        if best_iteration < 0:
            raise ValueError("XGBoost metadata best_iteration must be non-negative")
        xgb = cls._load_xgboost()
        booster = xgb.Booster()
        booster.load_model(booster_path)
        if tuple(booster.feature_names or ()) != feature_names:
            raise ValueError("Booster feature names do not match metadata")
        boosted_rounds = int(booster.num_boosted_rounds())
        if best_iteration >= boosted_rounds:
            raise ValueError(
                f"best_iteration {best_iteration} is outside Booster rounds {boosted_rounds}"
            )
        return cls(
            booster=booster,
            metadata=payload,
            feature_names=feature_names,
            best_iteration=best_iteration,
            model_dir=metadata_path.parent,
        )

    def score_batch(self, features: Sequence[Mapping[str, Any]]) -> list[float]:
        if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
            raise TypeError("features must be a sequence of mappings")
        if not features:
            return []
        import numpy as np

        matrix = np.zeros((len(features), len(self.feature_names)), dtype=np.float32)
        for row_index, candidate_features in enumerate(features):
            if not isinstance(candidate_features, Mapping):
                raise TypeError(f"features[{row_index}] must be a mapping")
            for raw_name, raw_value in candidate_features.items():
                name = str(raw_name)
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Feature {name!r} at row {row_index} is not numeric") from exc
                if not math.isfinite(value):
                    raise ValueError(f"Feature {name!r} at row {row_index} must be finite")
                feature_index = self._feature_index.get(name)
                if feature_index is None:
                    continue
                matrix[row_index, feature_index] = value
        xgb = self._load_xgboost()
        dmatrix = xgb.DMatrix(matrix, feature_names=list(self.feature_names), missing=float("nan"))
        predictions = self.booster.predict(
            dmatrix,
            iteration_range=(0, self.best_iteration + 1),
        )
        if len(predictions) != len(features):
            raise RuntimeError(
                f"XGBoost returned {len(predictions)} predictions for {len(features)} candidates"
            )
        result = [float(value) for value in predictions]
        if any(not math.isfinite(value) for value in result):
            raise RuntimeError("XGBoost returned a non-finite prediction")
        return result


def _float32(value: Any) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _compact_model_hash(payload: Mapping[str, Any]) -> str:
    core = {
        "base_score": payload.get("base_score"),
        "feature_names": payload.get("feature_names"),
        "trees": payload.get("trees"),
        "zero_default": payload.get("zero_default"),
    }
    encoded = json.dumps(core, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CompactXGBRootRanker:
    """Dependency-free evaluator for exported zero-default XGBoost trees."""

    def __init__(self, payload: dict[str, Any], model_path: Path):
        self.metadata = payload
        self.model_path = model_path
        self.feature_schema_version = str(payload["feature_schema_version"])
        self.feature_names = tuple(str(name) for name in payload["feature_names"])
        self.best_iteration = int(payload["best_iteration"])
        self.base_score = _float32(payload["base_score"])
        self._feature_index = {name: index for index, name in enumerate(self.feature_names)}
        self._trees = tuple(
            (
                tuple(int(value) for value in tree["f"]),
                tuple(_float32(value) for value in tree["t"]),
                tuple(int(value) for value in tree["l"]),
                tuple(int(value) for value in tree["r"]),
                tuple(_float32(value) for value in tree["v"]),
            )
            for tree in payload["trees"]
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "CompactXGBRootRanker":
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Compact XGBoost model does not exist: {model_path}")
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Compact XGBoost model must be a JSON object")
        if int(payload.get("schema_version", 0)) != COMPACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported compact XGBoost schema: {payload.get('schema_version')}")
        if payload.get("model_type") != COMPACT_MODEL_TYPE:
            raise ValueError(f"Unexpected compact XGBoost model type: {payload.get('model_type')!r}")
        if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "Compact XGBoost feature_schema_version must be "
                f"{FEATURE_SCHEMA_VERSION!r}; found {payload.get('feature_schema_version')!r}"
            )
        if payload.get("zero_default") is not True:
            raise ValueError("Compact XGBoost model must declare zero_default=true")
        raw_names = payload.get("feature_names")
        if not isinstance(raw_names, list) or len(raw_names) != INFERENCE_FEATURE_COUNT:
            raise ValueError(f"Compact XGBoost model requires exactly {INFERENCE_FEATURE_COUNT} features")
        if not all(isinstance(name, str) and name for name in raw_names) or len(set(raw_names)) != len(raw_names):
            raise ValueError("Compact XGBoost feature names must be unique non-empty strings")
        try:
            best_iteration = int(payload.get("best_iteration"))
            base_score = float(payload.get("base_score"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Compact XGBoost iteration/base score metadata is invalid") from exc
        trees = payload.get("trees")
        if best_iteration < 0 or not isinstance(trees, list) or len(trees) != best_iteration + 1:
            raise ValueError("Compact XGBoost tree count must equal best_iteration + 1")
        if payload.get("tree_count") != len(trees):
            raise ValueError("Compact XGBoost tree_count does not match trees")
        if not math.isfinite(base_score):
            raise ValueError("Compact XGBoost base score must be finite")
        for hash_name in ("source_booster_sha256", "source_metadata_sha256"):
            digest = payload.get(hash_name)
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"Compact XGBoost {hash_name} must be a lowercase SHA-256 digest")
        expected_hash = payload.get("model_sha256")
        actual_hash = _compact_model_hash(payload)
        if not isinstance(expected_hash, str) or expected_hash != actual_hash:
            raise ValueError(f"Compact XGBoost model hash mismatch: expected {expected_hash}, got {actual_hash}")
        for tree_index, tree in enumerate(trees):
            if not isinstance(tree, dict):
                raise ValueError(f"Compact XGBoost tree {tree_index} must be an object")
            arrays = [tree.get(key) for key in ("f", "t", "l", "r", "v")]
            if not all(isinstance(values, list) for values in arrays):
                raise ValueError(f"Compact XGBoost tree {tree_index} has missing node arrays")
            node_count = len(arrays[0])
            if node_count == 0 or any(len(values) != node_count for values in arrays[1:]):
                raise ValueError(f"Compact XGBoost tree {tree_index} node arrays differ in length")
            for node_index, (feature, threshold, left, right, leaf) in enumerate(zip(*arrays)):
                try:
                    feature = int(feature)
                    threshold = float(threshold)
                    left = int(left)
                    right = int(right)
                    leaf = float(leaf)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Compact XGBoost tree {tree_index} node {node_index} is invalid") from exc
                if not math.isfinite(threshold) or not math.isfinite(leaf):
                    raise ValueError(f"Compact XGBoost tree {tree_index} node {node_index} is non-finite")
                if feature == -1:
                    if left != -1 or right != -1:
                        raise ValueError(f"Compact XGBoost tree {tree_index} leaf {node_index} has children")
                elif not (0 <= feature < len(raw_names) and 0 <= left < node_count and 0 <= right < node_count):
                    raise ValueError(f"Compact XGBoost tree {tree_index} node {node_index} has invalid indices")
        return cls(payload, model_path)

    def validate_feature_schema(self, expected: str) -> None:
        expected_schema = str(expected).strip().lower()
        actual_schema = self.feature_schema_version.strip().lower()
        if expected_schema != actual_schema:
            raise ValueError(
                f"XGBoost feature schema mismatch: model={actual_schema!r}, evaluator={expected_schema!r}"
            )

    def score_batch(self, features: Sequence[Mapping[str, Any]]) -> list[float]:
        if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
            raise TypeError("features must be a sequence of mappings")
        result: list[float] = []
        for row_index, candidate_features in enumerate(features):
            if not isinstance(candidate_features, Mapping):
                raise TypeError(f"features[{row_index}] must be a mapping")
            values: dict[int, float] = {}
            for raw_name, raw_value in candidate_features.items():
                name = str(raw_name)
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Feature {name!r} at row {row_index} is not numeric") from exc
                if not math.isfinite(value):
                    raise ValueError(f"Feature {name!r} at row {row_index} must be finite")
                feature_index = self._feature_index.get(name)
                if feature_index is not None:
                    values[feature_index] = _float32(value)
            score = self.base_score
            for tree_features, thresholds, left, right, leaves in self._trees:
                node = 0
                while tree_features[node] >= 0:
                    feature_value = values.get(tree_features[node], 0.0)
                    node = left[node] if feature_value < thresholds[node] else right[node]
                score = _float32(score + leaves[node])
            result.append(score)
        return result


def export_compact(model_dir: str | Path, output: str | Path) -> dict[str, Any]:
    native = XGBRootRanker.from_directory(model_dir)
    config = json.loads(native.booster.save_config())
    raw_base_score = config["learner"]["learner_model_param"]["base_score"]
    parsed_base_score = json.loads(raw_base_score)
    base_score = parsed_base_score[0] if isinstance(parsed_base_score, list) else parsed_base_score
    feature_index = {name: index for index, name in enumerate(native.feature_names)}
    trees: list[dict[str, list[Any]]] = []
    dumps = native.booster.get_dump(dump_format="json")[: native.best_iteration + 1]
    for tree_index, raw_tree in enumerate(dumps):
        root = json.loads(raw_tree)
        nodes: dict[int, dict[str, Any]] = {}

        def collect(node: dict[str, Any]) -> None:
            nodes[int(node["nodeid"])] = node
            for child in node.get("children", ()):
                collect(child)

        collect(root)
        if set(nodes) != set(range(len(nodes))):
            raise ValueError(f"Native XGBoost tree {tree_index} has non-contiguous node ids")
        compact = {"f": [], "t": [], "l": [], "r": [], "v": []}
        for node_id in range(len(nodes)):
            node = nodes[node_id]
            if "leaf" in node:
                compact["f"].append(-1)
                compact["t"].append(0.0)
                compact["l"].append(-1)
                compact["r"].append(-1)
                compact["v"].append(_float32(node["leaf"]))
            else:
                compact["f"].append(feature_index[str(node["split"])])
                compact["t"].append(_float32(node["split_condition"]))
                compact["l"].append(int(node["yes"]))
                compact["r"].append(int(node["no"]))
                compact["v"].append(0.0)
        trees.append(compact)
    model_directory = Path(model_dir)
    payload: dict[str, Any] = {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "model_type": COMPACT_MODEL_TYPE,
        "feature_schema_version": native.feature_schema_version,
        "zero_default": True,
        "base_score": _float32(base_score),
        "best_iteration": native.best_iteration,
        "tree_count": len(trees),
        "feature_names": list(native.feature_names),
        "trees": trees,
        "source_booster_sha256": str(native.metadata["booster_sha256"]),
        "source_metadata_sha256": _sha256_path(model_directory / "metadata.json"),
    }
    payload["model_sha256"] = _compact_model_hash(payload)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite compact XGBoost model: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    return {
        "output": _normalized_path(destination),
        "bytes": destination.stat().st_size,
        "tree_count": len(trees),
        "model_sha256": payload["model_sha256"],
        "file_sha256": _sha256_path(destination),
    }


def load_root_ranker(path: str | Path) -> XGBRootRanker | CompactXGBRootRanker:
    model_path = Path(path)
    if model_path.is_dir():
        return XGBRootRanker.from_directory(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"XGBoost root-ranker path does not exist: {model_path}")
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict) and payload.get("model_type") == COMPACT_MODEL_TYPE:
        return CompactXGBRootRanker.from_file(model_path)
    return XGBRootRanker.from_file(model_path)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _high_elo_entries(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"episode_id", "path"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"High-Elo manifest must contain columns {sorted(required)}")
        for row in reader:
            episode_id = str(row.get("episode_id", "")).strip()
            replay_path = Path(str(row.get("path", "")).strip())
            if not episode_id or not str(replay_path):
                continue
            entries.append((episode_id, replay_path))
    return entries


def build_manifest(high_elo_manifest: str | Path, output: str | Path) -> dict[str, Any]:
    source_manifest = Path(high_elo_manifest)
    source_entries = _high_elo_entries(source_manifest)
    by_episode: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = defaultdict(list)
    for episode_id, path in source_entries:
        normalized = _normalized_path(path)
        if episode_id in by_episode:
            duplicates[episode_id].append(normalized)
            if normalized < _normalized_path(by_episode[episode_id]):
                by_episode[episode_id] = path
        else:
            by_episode[episode_id] = path

    required = sum(SPLIT_COUNTS.values())
    if len(by_episode) < required:
        raise ValueError(f"Need at least {required} unique replay episodes, found {len(by_episode)}")
    ranked = sorted(
        by_episode.items(),
        key=lambda item: (
            hashlib.sha256(f"{MANIFEST_HASH_PREFIX}{item[0]}".encode("utf-8")).hexdigest(),
            item[0],
        ),
    )[:required]
    splits: dict[str, list[dict[str, str]]] = {}
    offset = 0
    for split, count in SPLIT_COUNTS.items():
        entries = ranked[offset : offset + count]
        splits[split] = [
            {"episode_id": episode_id, "path": _normalized_path(path), "sha256": _sha256_path(path)}
            for episode_id, path in entries
        ]
        offset += count
    payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "xgb_root_ranker_replays",
        "selection_hash": "sha256('xgb-root-v1:' + episode_id)",
        "source_manifest": _normalized_path(source_manifest),
        "source_manifest_sha256": _sha256_path(source_manifest),
        "split_counts": dict(SPLIT_COUNTS),
        "available_unique_episodes": len(by_episode),
        "selected_episodes": required,
        "duplicate_episode_count": len(duplicates),
        "splits": splits,
    }
    _json_dump(Path(output), payload)
    return payload


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("manifest_type") != "xgb_root_ranker_replays":
        raise ValueError(f"Not an XGBoost root-ranker replay manifest: {manifest_path}")
    if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema: {payload.get('schema_version')}")
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Manifest has no splits")
    episode_ids: set[str] = set()
    for split, expected in SPLIT_COUNTS.items():
        entries = splits.get(split)
        if not isinstance(entries, list) or len(entries) != expected:
            raise ValueError(f"Manifest split {split!r} must contain exactly {expected} episodes")
        for entry in entries:
            if not isinstance(entry, dict) or not all(key in entry for key in ("episode_id", "path", "sha256")):
                raise ValueError(f"Manifest split {split!r} contains an invalid entry")
            episode_id = str(entry["episode_id"])
            if episode_id in episode_ids:
                raise ValueError(f"Episode {episode_id!r} occurs in more than one manifest split")
            episode_ids.add(episode_id)
    return payload


def _verified_manifest_paths(payload: dict[str, Any], split: str) -> list[str]:
    paths: list[str] = []
    for entry in payload["splits"][split]:
        path = Path(str(entry["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Manifest replay does not exist: {path}")
        actual = _sha256_path(path)
        if actual != str(entry["sha256"]):
            raise ValueError(f"Manifest replay hash changed: {path}")
        paths.append(str(path))
    return paths


def prepare_labels(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | None,
    weights_path: str | None,
    min_rating: float | None,
    workers: int,
) -> dict[str, Any]:
    from .replay_pipeline import build_label_dataset

    payload = _load_manifest(manifest)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    label_scans: dict[str, Any] = {}
    outputs: dict[str, str] = {}
    for split in SPLIT_COUNTS:
        output = destination / f"{split}.jsonl"
        inputs = _verified_manifest_paths(payload, split)
        summaries[split] = build_label_dataset(
            inputs,
            output=str(output),
            config_path=config_path,
            weights_path=weights_path,
            min_rating=min_rating,
            max_candidates=24,
            beam_width=LABEL_BUDGETS["beam_width"],
            max_depth=LABEL_BUDGETS["max_depth"],
            max_nodes=LABEL_BUDGETS["max_nodes"],
            time_budget_ms=LABEL_BUDGETS["time_budget_ms"],
            clean_evaluator=False,
            multi_self_decks=True,
            workers=workers,
            root_ranker_mode=True,
        )
        outputs[split] = str(output)
        label_scans[split] = scan_labels([output]).to_dict()
    result = {
        "schema_version": SCHEMA_VERSION,
        "labels_type": "xgb_root_ranker_labels",
        "replay_manifest": str(manifest),
        "replay_manifest_sha256": _sha256_path(Path(manifest)),
        "budgets": dict(LABEL_BUDGETS),
        "max_candidates": 24,
        "outputs": outputs,
        "summaries": summaries,
        "label_scans": label_scans,
    }
    _json_dump(destination / "labels_manifest.json", result)
    return result


def _root_key(selected: Any) -> tuple[int, ...]:
    if not isinstance(selected, list):
        raise ValueError("Candidate selected field must be a list")
    try:
        values = tuple(sorted(int(value) for value in selected))
    except (TypeError, ValueError) as exc:
        raise ValueError("Candidate selected field contains a non-integer") from exc
    if len(set(values)) != len(values):
        raise ValueError("Candidate selected field contains duplicate indices")
    return values


def _clean_features(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if key in GENERATED_FEATURES or any(key.startswith(prefix) for prefix in GENERATED_PREFIXES):
            continue
        try:
            number = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number != 0.0:
            cleaned[key] = number
    return cleaned


def _candidate_features(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, dict) or not isinstance(candidate.get("features"), dict):
        return {}
    return candidate["features"]


def _candidate_is_end(candidate: Any) -> bool:
    if not isinstance(candidate, dict):
        return False
    labels = candidate.get("labels")
    return isinstance(labels, (list, tuple)) and bool(labels) and str(labels[0]) == "end turn"


def _candidate_has_useful_non_end_action(candidate: Any) -> bool:
    if _candidate_is_end(candidate):
        return False
    features = _candidate_features(candidate)
    useful = (
        "attach_energy",
        "attach_to_main_line",
        "attach_to_backup",
        "evolve",
        "evolve_key",
        "evolve_main_attacker",
        "play_pokemon",
        "play_main_line",
        "play_setup_basic",
        "play_pokemon_to_avoid_empty_board_loss",
        "play_pokemon_behind_low_active",
        "play_pokemon_to_stabilize_board",
        "play_search",
        "play_draw",
        "attack_takes_ko",
        "attack_wins_game",
        "seq_my_prizes_gained",
        "seq_energy_delta",
        "seq_bench_delta",
    )
    return any(float(features.get(key, 0.0) or 0.0) > 0.0 for key in useful)


def _is_passive_end_group(label_index: int, candidates: list[Any]) -> bool:
    if not 0 <= label_index < len(candidates) or not _candidate_is_end(candidates[label_index]):
        return False
    label_features = _candidate_features(candidates[label_index])
    if float(label_features.get("end_with_useful_action", 0.0) or 0.0) > 0.0:
        return True
    return any(index != label_index and _candidate_has_useful_non_end_action(candidate) for index, candidate in enumerate(candidates))


def _iter_records(paths: Sequence[str | Path]) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: label record is not an object")
                if record.get("root_ranker_mode") is not True:
                    raise ValueError(f"{path}:{line_number}: root_ranker_mode=true is required")
                yield path, line_number, record


def scan_labels(paths: Sequence[str | Path]) -> LabelScan:
    records = passive = eligible = matched = unmatched = singles = single_matched = usable = 0
    eligible_by_episode: Counter[str] = Counter()
    single_by_episode: Counter[str] = Counter()
    usable_episode_ids: set[str] = set()
    for path, line_number, record in _iter_records(paths):
        records += 1
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"{path}:{line_number}: candidates must be a list")
        try:
            label_index = int(record.get("label_index", -1))
        except (TypeError, ValueError):
            label_index = -1
        is_matched = bool(record.get("matched", label_index >= 0)) and 0 <= label_index < len(candidates)
        if is_matched and _is_passive_end_group(label_index, candidates):
            passive += 1
            continue
        episode_id = str(record.get("episode_id", "") or f"{path.name}:{line_number}")
        eligible += 1
        eligible_by_episode[episode_id] += 1
        if is_matched:
            matched += 1
        else:
            unmatched += 1
        if len(candidates) < 2:
            singles += 1
            if is_matched and len(candidates) == 1:
                single_matched += 1
                single_by_episode[episode_id] += 1
            continue
        if is_matched:
            usable += 1
            usable_episode_ids.add(episode_id)
    return LabelScan(
        records=records,
        passive_end_groups=passive,
        eligible_records=eligible,
        matched_records=matched,
        unmatched_records=unmatched,
        single_root_groups=singles,
        single_root_matched=single_matched,
        usable_groups=usable,
        episodes=len(eligible_by_episode),
        usable_episodes=len(usable_episode_ids),
        eligible_by_episode=dict(eligible_by_episode),
        single_matched_by_episode=dict(single_by_episode),
    )


def iter_groups(paths: Sequence[str | Path]) -> Iterator[RootGroup]:
    for path, line_number, record in _iter_records(paths):
        candidates = record.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            continue
        try:
            label_index = int(record.get("label_index", -1))
        except (TypeError, ValueError):
            continue
        if not bool(record.get("matched", label_index >= 0)) or not 0 <= label_index < len(candidates):
            continue
        if _is_passive_end_group(label_index, candidates):
            continue
        roots: set[tuple[int, ...]] = set()
        features: list[dict[str, float]] = []
        base_scores: list[float] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"{path}:{line_number}: candidate is not an object")
            root = _root_key(candidate.get("selected"))
            if root in roots:
                raise ValueError(f"{path}:{line_number}: duplicate immediate root {root}")
            roots.add(root)
            features.append(_clean_features(candidate.get("features")))
            try:
                score = float(candidate.get("score", 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid base score") from exc
            if not math.isfinite(score):
                raise ValueError(f"{path}:{line_number}: non-finite base score")
            base_scores.append(score)
        episode_id = str(record.get("episode_id", "") or f"{path.name}:{line_number}")
        player_index = str(record.get("player_index", ""))
        step_index = str(record.get("step_index", line_number))
        yield RootGroup(
            episode_id=episode_id,
            group_id=f"{episode_id}:{player_index}:{step_index}",
            label_index=label_index,
            features=tuple(features),
            base_scores=tuple(base_scores),
        )


def build_feature_vocabulary(train_paths: Sequence[str | Path]) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for group in iter_groups(train_paths):
        present = {key for features in group.features for key in features}
        counts.update(present)
    eligible = [key for key, count in counts.items() if count >= MIN_FEATURE_DECISIONS]
    selected = sorted(eligible, key=lambda key: (-counts[key], key))[:MAX_FEATURES]
    if not selected:
        raise ValueError(
            f"Training labels contained no feature present in at least {MIN_FEATURE_DECISIONS} decisions"
        )
    return tuple(selected)


def build_matrix(paths: Sequence[str | Path], feature_names: Sequence[str]) -> MatrixData:
    import numpy as np

    group_infos: list[RootGroupInfo] = []
    for group in iter_groups(paths):
        group_infos.append(
            RootGroupInfo(
                episode_id=group.episode_id,
                group_id=group.group_id,
                label_index=group.label_index,
                size=group.size,
                base_scores=group.base_scores,
            )
        )
    if not group_infos:
        raise ValueError(f"No valid root-ranking groups in: {', '.join(map(str, paths))}")
    group_ids = [group.group_id for group in group_infos]
    if len(set(group_ids)) != len(group_ids):
        duplicates = [group_id for group_id, count in Counter(group_ids).items() if count > 1]
        raise ValueError(f"Duplicate decision groups in label split: {duplicates[:5]}")
    row_count = sum(group.size for group in group_infos)
    feature_index = {name: index for index, name in enumerate(feature_names)}
    matrix = np.zeros((row_count, len(feature_names)), dtype=np.float32)
    labels = np.zeros(row_count, dtype=np.float32)
    offset = 0
    for expected, group in zip(group_infos, iter_groups(paths)):
        if group.group_id != expected.group_id or group.size != expected.size:
            raise ValueError("Label files changed while constructing the dense matrix")
        labels[offset + group.label_index] = 1.0
        for local_index, features in enumerate(group.features):
            row = matrix[offset + local_index]
            for key, value in features.items():
                index = feature_index.get(key)
                if index is not None:
                    row[index] = value
        offset += group.size
    return MatrixData(
        matrix=matrix,
        labels=labels,
        group_sizes=tuple(group.size for group in group_infos),
        groups=tuple(group_infos),
    )


def _ranks(scores: Sequence[float]) -> list[int]:
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    ranks = [0] * len(scores)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def _metric_sums(groups: Sequence[RootGroupInfo], predictions: Sequence[Sequence[float]]) -> dict[str, float]:
    values = {"groups": 0.0, "top1": 0.0, "top3": 0.0, "mrr": 0.0, "ndcg3": 0.0}
    for group, scores in zip(groups, predictions):
        rank = _ranks(scores)[group.label_index]
        values["groups"] += 1.0
        values["top1"] += float(rank == 1)
        values["top3"] += float(rank <= 3)
        values["mrr"] += 1.0 / rank
        values["ndcg3"] += (1.0 / math.log2(rank + 1.0)) if rank <= 3 else 0.0
    return values


def _mean_metrics(sums: dict[str, float]) -> dict[str, float]:
    count = sums["groups"] or 1.0
    return {key: (value if key == "groups" else value / count) for key, value in sums.items()}


def _split_predictions(flat: Sequence[float], groups: Sequence[RootGroupInfo]) -> list[list[float]]:
    result: list[list[float]] = []
    offset = 0
    for group in groups:
        result.append([float(value) for value in flat[offset : offset + group.size]])
        offset += group.size
    if offset != len(flat):
        raise ValueError("Prediction row count does not match query groups")
    return result


def _bootstrap_delta(
    groups: Sequence[RootGroupInfo],
    model_predictions: Sequence[Sequence[float]],
    base_predictions: Sequence[Sequence[float]],
    scan: LabelScan,
) -> dict[str, Any]:
    episode_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        episode_indices[group.episode_id].append(index)
    episode_ids = sorted(scan.eligible_by_episode)
    if not episode_ids:
        return {}
    episode_sums: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for episode_id, indices in episode_indices.items():
        episode_sums[episode_id] = (
            _metric_sums([groups[index] for index in indices], [model_predictions[index] for index in indices]),
            _metric_sums([groups[index] for index in indices], [base_predictions[index] for index in indices]),
        )
    rng = random.Random(MANIFEST_SEED)
    samples: dict[str, list[float]] = {key: [] for key in ("top1", "top3", "mrr", "ndcg3")}
    for _ in range(BOOTSTRAP_REPLICATES):
        chosen = [episode_ids[rng.randrange(len(episode_ids))] for _ in episode_ids]
        model_sums = {"groups": 0.0, "top1": 0.0, "top3": 0.0, "mrr": 0.0, "ndcg3": 0.0}
        base_sums = dict(model_sums)
        eligible_count = 0.0
        for episode_id in chosen:
            model_episode, base_episode = episode_sums.get(
                episode_id,
                (
                    {"groups": 0.0, "top1": 0.0, "top3": 0.0, "mrr": 0.0, "ndcg3": 0.0},
                    {"groups": 0.0, "top1": 0.0, "top3": 0.0, "mrr": 0.0, "ndcg3": 0.0},
                ),
            )
            for key in model_sums:
                model_sums[key] += model_episode[key]
                base_sums[key] += base_episode[key]
            singleton = float(scan.single_matched_by_episode.get(episode_id, 0))
            for key in ("top1", "top3", "mrr", "ndcg3"):
                model_sums[key] += singleton
                base_sums[key] += singleton
            eligible_count += float(scan.eligible_by_episode[episode_id])
        for key in samples:
            samples[key].append((model_sums[key] - base_sums[key]) / max(1.0, eligible_count))

    result: dict[str, Any] = {}
    for key, values in samples.items():
        ordered = sorted(values)
        low = ordered[int(0.025 * (len(ordered) - 1))]
        high = ordered[int(0.975 * (len(ordered) - 1))]
        result[key] = {"ci95": [low, high], "bootstrap_replicates": BOOTSTRAP_REPLICATES}
    return result


def evaluate_predictions(
    groups: Sequence[RootGroupInfo],
    flat_predictions: Sequence[float],
    *,
    scan: LabelScan,
    include_bootstrap: bool = True,
) -> dict[str, Any]:
    model_predictions = _split_predictions(flat_predictions, groups)
    base_predictions = [list(group.base_scores) for group in groups]
    model_sums = _metric_sums(groups, model_predictions)
    base_sums = _metric_sums(groups, base_predictions)
    model = _mean_metrics(model_sums)
    base = _mean_metrics(base_sums)
    delta = {key: model[key] - base[key] for key in ("top1", "top3", "mrr", "ndcg3")}
    unconditional_model = {
        key: (model_sums[key] + scan.single_root_matched) / max(1, scan.eligible_records)
        for key in ("top1", "top3", "mrr", "ndcg3")
    }
    unconditional_base = {
        key: (base_sums[key] + scan.single_root_matched) / max(1, scan.eligible_records)
        for key in ("top1", "top3", "mrr", "ndcg3")
    }
    unconditional_delta = {
        key: unconditional_model[key] - unconditional_base[key]
        for key in ("top1", "top3", "mrr", "ndcg3")
    }
    result: dict[str, Any] = {
        "conditional": {"model": model, "base": base, "delta": delta},
        "unconditional": {
            "model": unconditional_model,
            "base": unconditional_base,
            "delta": unconditional_delta,
            "denominator": scan.eligible_records,
        },
        "label_scan": scan.to_dict(),
    }
    if include_bootstrap:
        bootstrap = _bootstrap_delta(groups, model_predictions, base_predictions, scan)
        for key, point in unconditional_delta.items():
            bootstrap.setdefault(key, {})["point"] = point
        result["episode_bootstrap_unconditional_delta"] = bootstrap
    return result


def _dmat(xgb: Any, data: MatrixData, feature_names: Sequence[str]) -> Any:
    matrix = xgb.DMatrix(data.matrix, label=data.labels, feature_names=list(feature_names), missing=float("nan"))
    matrix.set_group(list(data.group_sizes))
    return matrix


def train(
    train_labels: Sequence[str | Path],
    validation_labels: Sequence[str | Path],
    test_labels: Sequence[str | Path],
    output_dir: str | Path,
    *,
    _enforce_minimums: bool = True,
) -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - depends on optional training environment.
        raise RuntimeError("Training requires the optional xgboost dependency") from exc

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    booster_path = destination / "booster.json"
    metadata_path = destination / "metadata.json"
    if booster_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing model in {destination}")

    feature_names = build_feature_vocabulary(train_labels)
    label_scans = {
        "train": scan_labels(train_labels),
        "validation": scan_labels(validation_labels),
        "test": scan_labels(test_labels),
    }
    if _enforce_minimums:
        for split in ("validation", "test"):
            scan = label_scans[split]
            if scan.usable_groups < MIN_EVAL_GROUPS:
                raise ValueError(
                    f"{split} requires at least {MIN_EVAL_GROUPS} usable groups; found {scan.usable_groups}"
                )
            if scan.usable_episodes < MIN_EVAL_EPISODES:
                raise ValueError(
                    f"{split} requires at least {MIN_EVAL_EPISODES} usable episodes; found {scan.usable_episodes}"
                )
    train_data = build_matrix(train_labels, feature_names)
    validation_data = build_matrix(validation_labels, feature_names)
    test_data = build_matrix(test_labels, feature_names)
    split_episodes = {
        "train": {group.episode_id for group in train_data.groups},
        "validation": {group.episode_id for group in validation_data.groups},
        "test": {group.episode_id for group in test_data.groups},
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_episodes[left] & split_episodes[right]
        if overlap:
            raise ValueError(f"Episode leakage between {left} and {right}: {sorted(overlap)[:5]}")
    dtrain = _dmat(xgb, train_data, feature_names)
    dvalidation = _dmat(xgb, validation_data, feature_names)
    dtest = _dmat(xgb, test_data, feature_names)
    booster = xgb.train(
        dict(XGB_PARAMS),
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dvalidation, "validation")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    best_iteration = int(getattr(booster, "best_iteration", NUM_BOOST_ROUND - 1))
    iteration_range = (0, best_iteration + 1)
    validation_predictions = booster.predict(dvalidation, iteration_range=iteration_range)
    test_predictions = booster.predict(dtest, iteration_range=iteration_range)
    booster.save_model(booster_path)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "booster_file": booster_path.name,
        "booster_sha256": _sha256_path(booster_path),
        "xgboost_version": str(xgb.__version__),
        "params": dict(XGB_PARAMS),
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": best_iteration,
        "feature_names": list(feature_names),
        "feature_defaults": {name: 0.0 for name in feature_names},
        "feature_vocabulary": {
            "source_split": "train",
            "presence_unit": "decision",
            "minimum_decisions": MIN_FEATURE_DECISIONS,
            "maximum_features": MAX_FEATURES,
        },
        "excluded_generated_features": sorted(GENERATED_FEATURES),
        "excluded_generated_prefixes": list(GENERATED_PREFIXES),
        "label_filters": {"require_root_ranker_mode": True, "exclude_passive_end_groups": True},
        "evaluation_minimums": {"usable_groups": MIN_EVAL_GROUPS, "usable_episodes": MIN_EVAL_EPISODES},
        "groups": {
            "train": len(train_data.groups),
            "validation": len(validation_data.groups),
            "test": len(test_data.groups),
        },
        "rows": {
            "train": int(train_data.matrix.shape[0]),
            "validation": int(validation_data.matrix.shape[0]),
            "test": int(test_data.matrix.shape[0]),
        },
        "label_scans": {split: scan.to_dict() for split, scan in label_scans.items()},
        "validation_metrics_vs_base": evaluate_predictions(
            validation_data.groups,
            validation_predictions,
            scan=label_scans["validation"],
            include_bootstrap=False,
        ),
        "locked_test_metrics_vs_base": evaluate_predictions(
            test_data.groups,
            test_predictions,
            scan=label_scans["test"],
        ),
        "inputs": {
            "train": [{"path": str(path), "sha256": _sha256_path(Path(path))} for path in train_labels],
            "validation": [{"path": str(path), "sha256": _sha256_path(Path(path))} for path in validation_labels],
            "test": [{"path": str(path), "sha256": _sha256_path(Path(path))} for path in test_labels],
        },
    }
    _json_dump(metadata_path, metadata)
    return metadata


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline XGBoost immediate-root ranking experiment.", allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Select deterministic 160/40/40 replay splits.", allow_abbrev=False)
    manifest.add_argument("--high-elo-manifest", required=True, help="Path to high_elo_manifest.csv.")
    manifest.add_argument("--output", required=True)

    labels = subparsers.add_parser("prepare-labels", help="Generate root-ranker labels for a replay manifest.", allow_abbrev=False)
    labels.add_argument("--manifest", required=True)
    labels.add_argument("--output-dir", required=True)
    labels.add_argument("--config", default=None)
    labels.add_argument("--weights", default=None)
    labels.add_argument("--min-rating", type=float, default=None)
    labels.add_argument("--workers", type=_positive_int, default=1)

    trainer = subparsers.add_parser("train", help="Train and evaluate the fixed native XGBoost ranker.", allow_abbrev=False)
    trainer.add_argument("--train-labels", action="append", required=True)
    trainer.add_argument("--validation-labels", action="append", required=True)
    trainer.add_argument("--test-labels", action="append", required=True)
    trainer.add_argument("--output-dir", required=True)

    exporter = subparsers.add_parser(
        "export-compact",
        help="Export an early-stopped native Booster as a dependency-free compact JSON model.",
        allow_abbrev=False,
    )
    exporter.add_argument("--model", required=True, help="Native XGBoost model directory.")
    exporter.add_argument("--output", required=True, help="Destination compact JSON file.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "manifest":
        result = build_manifest(args.high_elo_manifest, args.output)
    elif args.command == "prepare-labels":
        result = prepare_labels(
            args.manifest,
            args.output_dir,
            config_path=args.config,
            weights_path=args.weights,
            min_rating=args.min_rating,
            workers=args.workers,
        )
    elif args.command == "train":
        result = train(args.train_labels, args.validation_labels, args.test_labels, args.output_dir)
    elif args.command == "export-compact":
        result = export_compact(args.model, args.output)
    else:  # pragma: no cover - argparse enforces a subcommand.
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
