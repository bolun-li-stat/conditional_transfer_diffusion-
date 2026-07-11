"""Deterministic split and training-subset manifests for image experiments.

The two manifest layers deliberately represent different sources of
randomness.  A split manifest fixes validation/evaluation holdouts, while a
training-subset manifest fixes nested training orderings inside the remaining
candidate pools.  Keeping the layers separate lets low-data repetitions vary
the training sample without changing the evaluation set.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence


SPLIT_MANIFEST_SCHEMA_VERSION = "2.0"
SUBSET_MANIFEST_SCHEMA_VERSION = "2.0"
COMBINED_MANIFEST_SCHEMA_VERSION = "2.0"
# Kept for callers that only need the current data-manifest schema generation.
MANIFEST_SCHEMA_VERSION = COMBINED_MANIFEST_SCHEMA_VERSION


class ManifestError(RuntimeError):
    """Base class for malformed or incompatible manifests."""


class ManifestInsufficientDataError(ManifestError):
    """Raised when a requested study split or training subset is unavailable."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


def canonical_sha256(value: Any) -> str:
    """Hash JSON data with stable key ordering and no platform-dependent spaces."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    """Return the canonical hash used in job identifiers and provenance."""

    return canonical_sha256(config)


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\0".join([str(seed), *parts]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _refs(source: str, values: Sequence[int | str]) -> list[str]:
    return [f"{source}:{value}" for value in values]


def parse_sample_ref(reference: str) -> tuple[str, int | str]:
    """Split a manifest reference into its source name and dataset index/path."""

    source, separator, raw = str(reference).partition(":")
    if not separator:
        raise ManifestError(f"Invalid sample reference {reference!r}")
    try:
        value: int | str = int(raw)
    except ValueError:
        value = raw
    return source, value


def _shuffled(values: Sequence[str], seed: int, namespace: str) -> list[str]:
    result = list(values)
    random.Random(_stable_seed(seed, namespace)).shuffle(result)
    return result


def _reserve_exact(
    values: Sequence[str],
    requested: int,
    *,
    what: str,
    mode: str,
    issues: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    requested = max(int(requested), 0)
    available = len(values)
    if available < requested:
        details = {"what": what, "requested": requested, "available": available, "shortfall": requested - available}
        if mode == "strict":
            raise ManifestInsufficientDataError(
                f"{what} has {available} samples after earlier reservations; need {requested}",
                details=details,
            )
        issues.append(details)
    actual = min(requested, available)
    return list(values[:actual]), list(values[actual:])


def _dataset_fingerprint(
    dataset_name: str,
    train_pools: Mapping[str, Sequence[int | str]],
    eval_pools: Mapping[str, Sequence[int | str]] | None,
    supplied: str | None,
) -> str:
    if supplied:
        return str(supplied)
    payload = {
        "dataset": dataset_name,
        "train": {str(key): list(values) for key, values in sorted(train_pools.items())},
        "evaluation": {str(key): list(values) for key, values in sorted((eval_pools or {}).items())},
    }
    return canonical_sha256(payload)


def resolve_manifest_seeds(
    *,
    holdout_seed: int | None = None,
    training_subset_seed: int | None = None,
    data_split_seed: int | None = None,
) -> tuple[int, int]:
    """Resolve v2 seeds, accepting the legacy combined seed with a warning."""

    if data_split_seed is not None:
        legacy = int(data_split_seed)
        if holdout_seed is not None and int(holdout_seed) != legacy:
            raise ValueError("data_split_seed conflicts with holdout_seed")
        if training_subset_seed is not None and int(training_subset_seed) != legacy:
            raise ValueError("data_split_seed conflicts with training_subset_seed")
        holdout_seed = legacy if holdout_seed is None else int(holdout_seed)
        training_subset_seed = legacy if training_subset_seed is None else int(training_subset_seed)
        warnings.warn(
            "data_split_seed is deprecated; use holdout_seed and training_subset_seed",
            DeprecationWarning,
            stacklevel=2,
        )
    return int(0 if holdout_seed is None else holdout_seed), int(
        0 if training_subset_seed is None else training_subset_seed
    )


def build_split_manifest(
    *,
    dataset_name: str,
    target_class: str,
    holdout_seed: int,
    train_pools: Mapping[str, Sequence[int | str]],
    auxiliary_classes: Sequence[str] = (),
    eval_pools: Mapping[str, Sequence[int | str]] | None = None,
    eval_source: str = "train_holdout",
    target_eval_size: int = 500,
    target_val_size: int = 100,
    auxiliary_eval_size: int = 100,
    target_similarity_reference_size: int = 0,
    auxiliary_similarity_reference_size: int = 0,
    dataset_fingerprint: str | None = None,
    mode: str = "strict",
) -> dict[str, Any]:
    """Create the holdout layer, independent of experiment and training subset.

    Candidate pools are stored in canonical order.  Only the holdout selection
    depends on ``holdout_seed``; a separate subset manifest later permutes the
    remaining training candidates.
    """

    if mode not in {"strict", "debug"}:
        raise ValueError(f"Unknown manifest mode {mode!r}; expected 'strict' or 'debug'")
    if target_class not in train_pools:
        raise ManifestError(f"Target class {target_class!r} is absent from train_pools")
    eval_source = str(eval_source)
    uses_train_holdout = eval_source == "train_holdout"
    if not uses_train_holdout and (not eval_pools or target_class not in eval_pools):
        raise ManifestError(f"eval_source={eval_source!r} requires a target pool in eval_pools")

    seed = int(holdout_seed)
    issues: list[dict[str, Any]] = []
    train_refs = {str(label): _refs("train", values) for label, values in train_pools.items()}
    eval_refs = {str(label): _refs(eval_source, values) for label, values in (eval_pools or {}).items()}

    if uses_train_holdout:
        remaining = _shuffled(train_refs[target_class], seed, f"target:{target_class}:holdout")
        target_eval, remaining = _reserve_exact(
            remaining, target_eval_size, what="target evaluation split", mode=mode, issues=issues
        )
        target_val, target_training = _reserve_exact(
            remaining, target_val_size, what="target validation split", mode=mode, issues=issues
        )
        target_similarity, target_training = _reserve_exact(
            target_training,
            target_similarity_reference_size,
            what="target similarity reference split",
            mode=mode,
            issues=issues,
        )
        target_training = sorted(target_training)
    else:
        target_training = sorted(train_refs[target_class])
        remaining = _shuffled(eval_refs[target_class], seed, f"target:{target_class}:{eval_source}:holdout")
        target_eval, remaining = _reserve_exact(
            remaining, target_eval_size, what="target evaluation split", mode=mode, issues=issues
        )
        target_val, remaining = _reserve_exact(
            remaining, target_val_size, what="target validation split", mode=mode, issues=issues
        )
        target_similarity, _ = _reserve_exact(
            remaining,
            target_similarity_reference_size,
            what="target similarity reference split",
            mode=mode,
            issues=issues,
        )

    auxiliary_pools: dict[str, dict[str, list[str]]] = {}
    for auxiliary in sorted(set(map(str, auxiliary_classes))):
        if auxiliary not in train_refs:
            raise ManifestError(f"Auxiliary class {auxiliary!r} is absent from train_pools")
        if uses_train_holdout:
            shuffled = _shuffled(train_refs[auxiliary], seed, f"aux:{auxiliary}:holdout")
            aux_eval, aux_train = _reserve_exact(
                shuffled,
                auxiliary_eval_size,
                what=f"auxiliary evaluation split for {auxiliary}",
                mode=mode,
                issues=issues,
            )
            aux_similarity, aux_train = _reserve_exact(
                aux_train,
                auxiliary_similarity_reference_size,
                what=f"auxiliary similarity reference split for {auxiliary}",
                mode=mode,
                issues=issues,
            )
            aux_train = sorted(aux_train)
        else:
            aux_train = sorted(train_refs[auxiliary])
            candidates = _shuffled(eval_refs.get(auxiliary, []), seed, f"aux:{auxiliary}:{eval_source}:holdout")
            aux_eval, candidates = _reserve_exact(
                candidates,
                auxiliary_eval_size,
                what=f"auxiliary evaluation split for {auxiliary}",
                mode=mode,
                issues=issues,
            )
            aux_similarity, _ = _reserve_exact(
                candidates,
                auxiliary_similarity_reference_size,
                what=f"auxiliary similarity reference split for {auxiliary}",
                mode=mode,
                issues=issues,
            )
        auxiliary_pools[auxiliary] = {
            "train_candidate_pool": aux_train,
            "eval_candidate_pool": aux_eval,
            "similarity_reference": aux_similarity,
        }

    fingerprint = _dataset_fingerprint(dataset_name, train_pools, eval_pools, dataset_fingerprint)
    payload: dict[str, Any] = {
        "manifest_kind": "split",
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "dataset_name": str(dataset_name),
        "dataset_fingerprint": fingerprint,
        "target_class": str(target_class),
        "holdout_seed": seed,
        "eval_source": eval_source,
        "holdout_specification": {
            "target_eval_size": int(target_eval_size),
            "target_validation_size": int(target_val_size),
            "auxiliary_eval_size_per_class": int(auxiliary_eval_size),
            "target_similarity_reference_size": int(target_similarity_reference_size),
            "auxiliary_similarity_reference_size_per_class": int(auxiliary_similarity_reference_size),
        },
        "target": {
            "eval": target_eval,
            "validation": target_val,
            "similarity_reference": target_similarity,
            "train_candidate_pool": target_training,
        },
        "auxiliary": auxiliary_pools,
        "split_sizes": {
            "requested_target_eval": int(target_eval_size),
            "requested_target_validation": int(target_val_size),
            "actual_target_eval": len(target_eval),
            "actual_target_validation": len(target_val),
            "actual_target_similarity_reference": len(target_similarity),
            "target_training_available": len(target_training),
            "requested_auxiliary_eval_per_class": int(auxiliary_eval_size),
            "requested_target_similarity_reference": int(target_similarity_reference_size),
            "requested_auxiliary_similarity_reference_per_class": int(auxiliary_similarity_reference_size),
            "auxiliary_training_available": {
                label: len(pools["train_candidate_pool"]) for label, pools in auxiliary_pools.items()
            },
            "auxiliary_eval_available": {
                label: len(pools["eval_candidate_pool"]) for label, pools in auxiliary_pools.items()
            },
            "auxiliary_similarity_reference_available": {
                label: len(pools["similarity_reference"]) for label, pools in auxiliary_pools.items()
            },
        },
        "feasibility_issues": issues,
    }
    payload["target_similarity_reference_hash"] = canonical_sha256(target_similarity)
    payload["auxiliary_similarity_reference_hashes"] = {
        label: canonical_sha256(pools["similarity_reference"])
        for label, pools in sorted(auxiliary_pools.items())
    }
    payload["split_manifest_hash"] = canonical_sha256(payload)
    validate_split_manifest(payload)
    return payload


def build_subset_manifest(
    split_manifest: Mapping[str, Any],
    *,
    training_subset_seed: int,
    nested_training_subsets: bool = True,
) -> dict[str, Any]:
    """Create deterministic per-class training orders for one split manifest."""

    validate_split_manifest(split_manifest)
    if not nested_training_subsets:
        raise ValueError("Only the nested-prefix training-subset policy is supported")
    seed = int(training_subset_seed)
    target_class = str(split_manifest["target_class"])
    target_candidates = list(split_manifest["target"]["train_candidate_pool"])
    auxiliary_orders = {
        str(label): _shuffled(
            list(pools["train_candidate_pool"]), seed, f"aux:{label}:training-subset-order:v1"
        )
        for label, pools in sorted(split_manifest.get("auxiliary", {}).items())
    }
    payload: dict[str, Any] = {
        "manifest_kind": "training_subset",
        "schema_version": SUBSET_MANIFEST_SCHEMA_VERSION,
        "split_manifest_hash": str(split_manifest["split_manifest_hash"]),
        "training_subset_seed": seed,
        "nested_subset_policy": {"name": "prefix", "version": 1},
        "candidate_pool_hashes": {
            "target": canonical_sha256(target_candidates),
            "auxiliary": {
                str(label): canonical_sha256(list(pools["train_candidate_pool"]))
                for label, pools in sorted(split_manifest.get("auxiliary", {}).items())
            },
        },
        "target_training_order": _shuffled(
            target_candidates, seed, f"target:{target_class}:training-subset-order:v1"
        ),
        "auxiliary_training_order": auxiliary_orders,
    }
    payload["subset_manifest_hash"] = canonical_sha256(payload)
    validate_subset_manifest(payload, split_manifest=split_manifest)
    return payload


def combine_manifests(
    split_manifest: Mapping[str, Any], subset_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Return an in-memory compatibility view used by legacy callers."""

    validate_split_manifest(split_manifest)
    validate_subset_manifest(subset_manifest, split_manifest=split_manifest)
    auxiliary = {
        str(label): {
            "train_candidate_pool": list(subset_manifest["auxiliary_training_order"][label]),
            "eval_candidate_pool": list(pools["eval_candidate_pool"]),
            "similarity_reference": list(pools["similarity_reference"]),
        }
        for label, pools in split_manifest.get("auxiliary", {}).items()
    }
    payload: dict[str, Any] = {
        "manifest_kind": "combined",
        "schema_version": COMBINED_MANIFEST_SCHEMA_VERSION,
        "dataset_name": split_manifest["dataset_name"],
        "dataset_fingerprint": split_manifest["dataset_fingerprint"],
        "target_class": split_manifest["target_class"],
        "holdout_seed": split_manifest["holdout_seed"],
        "training_subset_seed": subset_manifest["training_subset_seed"],
        "eval_source": split_manifest["eval_source"],
        "nested_training_subsets": True,
        "split_manifest_hash": split_manifest["split_manifest_hash"],
        "subset_manifest_hash": subset_manifest["subset_manifest_hash"],
        "split_manifest": dict(split_manifest),
        "subset_manifest": dict(subset_manifest),
        "target": {
            "eval": list(split_manifest["target"]["eval"]),
            "validation": list(split_manifest["target"]["validation"]),
            "similarity_reference": list(split_manifest["target"]["similarity_reference"]),
            "train_candidate_pool": list(subset_manifest["target_training_order"]),
        },
        "auxiliary": auxiliary,
        "split_sizes": dict(split_manifest["split_sizes"]),
        "target_similarity_reference_hash": split_manifest["target_similarity_reference_hash"],
        "auxiliary_similarity_reference_hashes": dict(
            split_manifest["auxiliary_similarity_reference_hashes"]
        ),
        "feasibility_issues": list(split_manifest.get("feasibility_issues", [])),
    }
    payload["manifest_hash"] = canonical_sha256(
        {
            "schema_version": COMBINED_MANIFEST_SCHEMA_VERSION,
            "split_manifest_hash": payload["split_manifest_hash"],
            "subset_manifest_hash": payload["subset_manifest_hash"],
        }
    )
    validate_manifest(payload)
    return payload


def build_data_manifest(
    *,
    dataset_name: str,
    target_class: str,
    train_pools: Mapping[str, Sequence[int | str]],
    data_split_seed: int | None = None,
    holdout_seed: int | None = None,
    training_subset_seed: int | None = None,
    auxiliary_classes: Sequence[str] = (),
    eval_pools: Mapping[str, Sequence[int | str]] | None = None,
    eval_source: str = "train_holdout",
    target_eval_size: int = 500,
    target_val_size: int = 100,
    auxiliary_eval_size: int = 100,
    target_similarity_reference_size: int = 0,
    auxiliary_similarity_reference_size: int = 0,
    experiment_family: str = "image_transfer",
    dataset_fingerprint: str | None = None,
    mode: str = "strict",
    nested_training_subsets: bool = True,
) -> dict[str, Any]:
    """Compatibility constructor returning a combined v2 manifest view.

    ``experiment_family`` is accepted for old call sites but intentionally does
    not affect either manifest identity.
    """

    del experiment_family
    resolved_holdout, resolved_subset = resolve_manifest_seeds(
        holdout_seed=holdout_seed,
        training_subset_seed=training_subset_seed,
        data_split_seed=data_split_seed,
    )
    split = build_split_manifest(
        dataset_name=dataset_name,
        target_class=target_class,
        holdout_seed=resolved_holdout,
        train_pools=train_pools,
        auxiliary_classes=auxiliary_classes,
        eval_pools=eval_pools,
        eval_source=eval_source,
        target_eval_size=target_eval_size,
        target_val_size=target_val_size,
        auxiliary_eval_size=auxiliary_eval_size,
        target_similarity_reference_size=target_similarity_reference_size,
        auxiliary_similarity_reference_size=auxiliary_similarity_reference_size,
        dataset_fingerprint=dataset_fingerprint,
        mode=mode,
    )
    subset = build_subset_manifest(
        split,
        training_subset_seed=resolved_subset,
        nested_training_subsets=nested_training_subsets,
    )
    return combine_manifests(split, subset)


def _validate_hash(manifest: Mapping[str, Any], hash_field: str) -> None:
    supplied = str(manifest.get(hash_field, ""))
    unhashed = {key: value for key, value in manifest.items() if key != hash_field}
    expected = canonical_sha256(unhashed)
    if supplied != expected:
        raise ManifestError(f"Manifest hash mismatch: stored={supplied!r}, expected={expected!r}")


def validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("manifest_kind") != "split" or manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ManifestError("Unsupported split-manifest schema")
    _validate_hash(manifest, "split_manifest_hash")
    target = manifest.get("target", {})
    train = set(target.get("train_candidate_pool", []))
    validation = set(target.get("validation", []))
    evaluation = set(target.get("eval", []))
    similarity = set(target.get("similarity_reference", []))
    target_parts = [train, validation, evaluation, similarity]
    if any(left & right for index, left in enumerate(target_parts) for right in target_parts[index + 1 :]):
        raise ManifestError("Target train/validation/evaluation/similarity splits overlap")
    if manifest.get("target_similarity_reference_hash") != canonical_sha256(
        list(target.get("similarity_reference", []))
    ):
        raise ManifestError("Target similarity-reference hash mismatch")
    expected_auxiliary_hashes: dict[str, str] = {}
    for label, pools in manifest.get("auxiliary", {}).items():
        train_pool = set(pools.get("train_candidate_pool", []))
        eval_pool = set(pools.get("eval_candidate_pool", []))
        similarity_pool = set(pools.get("similarity_reference", []))
        if train_pool & eval_pool or train_pool & similarity_pool or eval_pool & similarity_pool:
            raise ManifestError(f"Auxiliary train/evaluation/similarity splits overlap for {label}")
        expected_auxiliary_hashes[str(label)] = canonical_sha256(
            list(pools.get("similarity_reference", []))
        )
    if manifest.get("auxiliary_similarity_reference_hashes") != expected_auxiliary_hashes:
        raise ManifestError("Auxiliary similarity-reference hashes mismatch")


def validate_subset_manifest(
    manifest: Mapping[str, Any], *, split_manifest: Mapping[str, Any] | None = None
) -> None:
    if (
        manifest.get("manifest_kind") != "training_subset"
        or manifest.get("schema_version") != SUBSET_MANIFEST_SCHEMA_VERSION
    ):
        raise ManifestError("Unsupported training-subset-manifest schema")
    _validate_hash(manifest, "subset_manifest_hash")
    if manifest.get("nested_subset_policy") != {"name": "prefix", "version": 1}:
        raise ManifestError("Unsupported training-subset policy")
    if split_manifest is None:
        return
    validate_split_manifest(split_manifest)
    if manifest.get("split_manifest_hash") != split_manifest.get("split_manifest_hash"):
        raise ManifestError("Training-subset manifest references a different split manifest")
    expected_target = list(split_manifest["target"]["train_candidate_pool"])
    actual_target = list(manifest.get("target_training_order", []))
    if len(actual_target) != len(expected_target) or set(actual_target) != set(expected_target):
        raise ManifestError("Target training order is not a permutation of the split candidate pool")
    expected_auxiliary = {
        str(label): list(pools["train_candidate_pool"])
        for label, pools in split_manifest.get("auxiliary", {}).items()
    }
    actual_auxiliary = manifest.get("auxiliary_training_order", {})
    if set(actual_auxiliary) != set(expected_auxiliary):
        raise ManifestError("Auxiliary training-order classes do not match the split manifest")
    for label, expected in expected_auxiliary.items():
        actual = list(actual_auxiliary[label])
        if len(actual) != len(expected) or set(actual) != set(expected):
            raise ManifestError(f"Auxiliary training order for {label} is not a candidate-pool permutation")


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate a split, subset, or combined manifest."""

    kind = manifest.get("manifest_kind")
    if kind == "split":
        validate_split_manifest(manifest)
        return
    if kind == "training_subset":
        validate_subset_manifest(manifest)
        return
    if kind != "combined" or manifest.get("schema_version") != COMBINED_MANIFEST_SCHEMA_VERSION:
        raise ManifestError("Unsupported combined-manifest schema")
    split = manifest.get("split_manifest", {})
    subset = manifest.get("subset_manifest", {})
    validate_split_manifest(split)
    validate_subset_manifest(subset, split_manifest=split)
    expected = canonical_sha256(
        {
            "schema_version": COMBINED_MANIFEST_SCHEMA_VERSION,
            "split_manifest_hash": split["split_manifest_hash"],
            "subset_manifest_hash": subset["subset_manifest_hash"],
        }
    )
    if manifest.get("manifest_hash") != expected:
        raise ManifestError("Combined manifest hash mismatch")


def _training_order(manifest: Mapping[str, Any], *, auxiliary: str | None = None) -> list[str]:
    kind = manifest.get("manifest_kind")
    if kind == "training_subset":
        if auxiliary is None:
            return list(manifest["target_training_order"])
        return list(manifest.get("auxiliary_training_order", {}).get(str(auxiliary), []))
    if auxiliary is None:
        return list(manifest["target"]["train_candidate_pool"])
    return list(manifest.get("auxiliary", {}).get(str(auxiliary), {}).get("train_candidate_pool", []))


def target_training_subset(manifest: Mapping[str, Any], count: int) -> list[str]:
    """Return the nested prefix for ``count`` target training samples."""

    pool = _training_order(manifest)
    requested = int(count)
    if requested < 0:
        raise ValueError("Target training count must be non-negative")
    if len(pool) < requested:
        details = {
            "what": "target training subset",
            "requested": requested,
            "available": len(pool),
            "shortfall": requested - len(pool),
        }
        raise ManifestInsufficientDataError(
            f"Target training pool has {len(pool)} samples after validation/evaluation reservations; need {requested}",
            details=details,
        )
    return pool[:requested]


def auxiliary_training_subset(manifest: Mapping[str, Any], auxiliary: str, count: int) -> list[str]:
    pool = _training_order(manifest, auxiliary=str(auxiliary))
    requested = int(count)
    if requested < 0:
        raise ValueError("Auxiliary training count must be non-negative")
    if len(pool) < requested:
        details = {
            "what": f"auxiliary training subset for {auxiliary}",
            "requested": requested,
            "available": len(pool),
            "shortfall": requested - len(pool),
        }
        raise ManifestInsufficientDataError(
            f"Auxiliary class {auxiliary} has {len(pool)} training samples after reservations; need {requested}",
            details=details,
        )
    return pool[:requested]


def _split_view(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest.get("split_manifest", manifest)


def equal_total_feasibility(manifest: Mapping[str, Any], *, n0: int, m_per_aux: int, k_aux: int) -> dict[str, Any]:
    """Check Experiment-B target-only feasibility after all reservations."""

    split = _split_view(manifest)
    required = int(n0) + int(m_per_aux) * int(k_aux)
    available = len(split["target"]["train_candidate_pool"])
    feasible = available >= required
    return {
        "feasible": feasible,
        "requested_target_train": required,
        "available_target_train_after_reservations": available,
        "shortfall": max(required - available, 0),
        "reason": "" if feasible else "insufficient_target_images_after_manifest_reservations",
    }


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return cleaned or "unnamed"


def split_manifest_path(
    manifest_root: str | Path,
    *,
    dataset_name: str,
    target_class: str,
    holdout_seed: int,
    split_manifest_hash: str,
) -> Path:
    filename = (
        f"{_safe_component(target_class)}__holdout{int(holdout_seed)}"
        f"__{str(split_manifest_hash)[:16]}.split.json"
    )
    return Path(manifest_root) / _safe_component(dataset_name) / _safe_component(target_class) / filename


def subset_manifest_path(
    manifest_root: str | Path,
    *,
    dataset_name: str,
    target_class: str,
    holdout_seed: int,
    training_subset_seed: int,
    split_manifest_hash: str,
    subset_manifest_hash: str,
) -> Path:
    filename = (
        f"{_safe_component(target_class)}__holdout{int(holdout_seed)}"
        f"__subset{int(training_subset_seed)}__{str(subset_manifest_hash)[:16]}.subset.json"
    )
    return (
        Path(manifest_root)
        / _safe_component(dataset_name)
        / _safe_component(target_class)
        / str(split_manifest_hash)[:16]
        / filename
    )


def manifest_path(
    manifest_root: str | Path,
    *,
    dataset_name: str,
    target_class: str,
    data_split_seed: int,
    experiment_family: str = "image_transfer",
) -> Path:
    """Legacy logical path helper; new code uses the hash-addressed v2 paths."""

    del experiment_family
    filename = f"{_safe_component(target_class)}__split{int(data_split_seed)}.manifest.json"
    return Path(manifest_root) / _safe_component(dataset_name) / _safe_component(target_class) / filename


def write_manifest_atomic(manifest: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically persist a manifest so concurrent workers cannot truncate it."""

    validate_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(manifest), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def _identity_hash(manifest: Mapping[str, Any]) -> str:
    kind = manifest.get("manifest_kind")
    if kind == "split":
        return str(manifest["split_manifest_hash"])
    if kind == "training_subset":
        return str(manifest["subset_manifest_hash"])
    return str(manifest["manifest_hash"])


def persist_or_validate_manifest(manifest: Mapping[str, Any], path: str | Path) -> tuple[dict[str, Any], Path]:
    """Reuse an identical manifest or atomically create it."""

    destination = Path(path)
    if destination.exists():
        existing = load_manifest(destination)
        if _identity_hash(existing) != _identity_hash(manifest):
            raise ManifestError(
                f"Existing manifest at {destination} does not match the current manifest identity "
                f"({_identity_hash(existing)} != {_identity_hash(manifest)})"
            )
        return existing, destination
    write_manifest_atomic(manifest, destination)
    final = load_manifest(destination)
    if _identity_hash(final) != _identity_hash(manifest):
        raise ManifestError(f"Concurrent manifest creation produced a different manifest at {destination}")
    return final, destination


def unique_combination_count(group_sizes: Mapping[str, tuple[int, int]]) -> int:
    """Return the number of class draws implied by ``(pool_size, draw_size)`` pairs."""

    total = 1
    for pool_size, draw_size in group_sizes.values():
        total *= math.comb(int(pool_size), int(draw_size))
    return total
