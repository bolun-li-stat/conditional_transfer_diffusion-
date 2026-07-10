"""Deterministic, model-independent data manifests for image experiments.

The manifest is deliberately a plain JSON-compatible dictionary.  Dataset
builders may therefore create and validate splits before constructing a model,
and evaluation code can use ``manifest_hash`` as part of every cache key.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = "1.0"


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


def build_data_manifest(
    *,
    dataset_name: str,
    target_class: str,
    data_split_seed: int,
    train_pools: Mapping[str, Sequence[int | str]],
    auxiliary_classes: Sequence[str] = (),
    eval_pools: Mapping[str, Sequence[int | str]] | None = None,
    eval_source: str = "train_holdout",
    target_eval_size: int = 500,
    target_val_size: int = 100,
    auxiliary_eval_size: int = 100,
    experiment_family: str = "image_transfer",
    dataset_fingerprint: str | None = None,
    mode: str = "strict",
    nested_training_subsets: bool = True,
) -> dict[str, Any]:
    """Create a canonical split manifest without any model-specific input.

    ``train_pools`` and ``eval_pools`` map class identifiers to stable dataset
    indices or relative paths.  In ``train_holdout`` mode evaluation and
    validation are reserved, in that order, before the training candidate pool
    is exposed.  The remaining target order defines all nested ``n0`` subsets.
    """

    if mode not in {"strict", "debug"}:
        raise ValueError(f"Unknown manifest mode {mode!r}; expected 'strict' or 'debug'")
    if target_class not in train_pools:
        raise ManifestError(f"Target class {target_class!r} is absent from train_pools")
    eval_source = str(eval_source)
    uses_train_holdout = eval_source == "train_holdout"
    if not uses_train_holdout and (not eval_pools or target_class not in eval_pools):
        raise ManifestError(f"eval_source={eval_source!r} requires a target pool in eval_pools")

    seed = int(data_split_seed)
    issues: list[dict[str, Any]] = []
    train_refs = {str(label): _refs("train", values) for label, values in train_pools.items()}
    eval_refs = {str(label): _refs(eval_source, values) for label, values in (eval_pools or {}).items()}

    if uses_train_holdout:
        target_remaining = _shuffled(train_refs[target_class], seed, f"target:{target_class}:train")
        target_eval, target_remaining = _reserve_exact(
            target_remaining, target_eval_size, what="target evaluation split", mode=mode, issues=issues
        )
        target_val, target_training = _reserve_exact(
            target_remaining, target_val_size, what="target validation split", mode=mode, issues=issues
        )
    else:
        target_training = _shuffled(train_refs[target_class], seed, f"target:{target_class}:train")
        target_eval_remaining = _shuffled(eval_refs[target_class], seed, f"target:{target_class}:{eval_source}")
        target_eval, target_eval_remaining = _reserve_exact(
            target_eval_remaining, target_eval_size, what="target evaluation split", mode=mode, issues=issues
        )
        target_val, _ = _reserve_exact(
            target_eval_remaining, target_val_size, what="target validation split", mode=mode, issues=issues
        )

    auxiliary_pools: dict[str, dict[str, list[str]]] = {}
    for auxiliary in sorted(set(map(str, auxiliary_classes))):
        if auxiliary not in train_refs:
            raise ManifestError(f"Auxiliary class {auxiliary!r} is absent from train_pools")
        shuffled_train = _shuffled(train_refs[auxiliary], seed, f"aux:{auxiliary}:train")
        if uses_train_holdout:
            aux_eval, aux_train = _reserve_exact(
                shuffled_train,
                auxiliary_eval_size,
                what=f"auxiliary evaluation split for {auxiliary}",
                mode=mode,
                issues=issues,
            )
        else:
            aux_train = shuffled_train
            candidates = _shuffled(eval_refs.get(auxiliary, []), seed, f"aux:{auxiliary}:{eval_source}")
            aux_eval, _ = _reserve_exact(
                candidates,
                auxiliary_eval_size,
                what=f"auxiliary evaluation split for {auxiliary}",
                mode=mode,
                issues=issues,
            )
        auxiliary_pools[auxiliary] = {
            "train_candidate_pool": aux_train,
            "eval_candidate_pool": aux_eval,
        }

    fingerprint = _dataset_fingerprint(dataset_name, train_pools, eval_pools, dataset_fingerprint)
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_name": str(dataset_name),
        "dataset_fingerprint": fingerprint,
        "target_class": str(target_class),
        "experiment_family": str(experiment_family),
        "data_split_seed": seed,
        "eval_source": eval_source,
        "nested_training_subsets": bool(nested_training_subsets),
        "target": {
            "eval": target_eval,
            "validation": target_val,
            "train_candidate_pool": target_training,
        },
        "auxiliary": auxiliary_pools,
        "split_sizes": {
            "requested_target_eval": int(target_eval_size),
            "requested_target_validation": int(target_val_size),
            "actual_target_eval": len(target_eval),
            "actual_target_validation": len(target_val),
            "target_training_available": len(target_training),
            "requested_auxiliary_eval_per_class": int(auxiliary_eval_size),
            "auxiliary_training_available": {
                label: len(pools["train_candidate_pool"]) for label, pools in auxiliary_pools.items()
            },
            "auxiliary_eval_available": {
                label: len(pools["eval_candidate_pool"]) for label, pools in auxiliary_pools.items()
            },
        },
        "feasibility_issues": issues,
    }
    payload["manifest_hash"] = canonical_sha256(payload)
    validate_manifest(payload)
    return payload


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate schema, content hash, and split non-overlap."""

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported manifest schema {manifest.get('schema_version')!r}; expected {MANIFEST_SCHEMA_VERSION!r}"
        )
    supplied_hash = str(manifest.get("manifest_hash", ""))
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    expected_hash = canonical_sha256(unhashed)
    if supplied_hash != expected_hash:
        raise ManifestError(f"Manifest hash mismatch: stored={supplied_hash!r}, expected={expected_hash!r}")

    target = manifest.get("target", {})
    train = set(target.get("train_candidate_pool", []))
    validation = set(target.get("validation", []))
    evaluation = set(target.get("eval", []))
    if train & validation or train & evaluation or validation & evaluation:
        raise ManifestError("Target train/validation/evaluation splits overlap")
    for label, pools in manifest.get("auxiliary", {}).items():
        if set(pools.get("train_candidate_pool", [])) & set(pools.get("eval_candidate_pool", [])):
            raise ManifestError(f"Auxiliary train/evaluation splits overlap for {label}")


def target_training_subset(manifest: Mapping[str, Any], count: int) -> list[str]:
    """Return the nested prefix for ``count`` target training samples."""

    pool = list(manifest["target"]["train_candidate_pool"])
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
    pool = list(manifest.get("auxiliary", {}).get(str(auxiliary), {}).get("train_candidate_pool", []))
    requested = int(count)
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


def equal_total_feasibility(manifest: Mapping[str, Any], *, n0: int, m_per_aux: int, k_aux: int) -> dict[str, Any]:
    """Check Experiment-B target-only feasibility after all reservations."""

    required = int(n0) + int(m_per_aux) * int(k_aux)
    available = len(manifest["target"]["train_candidate_pool"])
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


def manifest_path(
    manifest_root: str | Path,
    *,
    dataset_name: str,
    target_class: str,
    data_split_seed: int,
    experiment_family: str,
) -> Path:
    filename = (
        f"{_safe_component(experiment_family)}__{_safe_component(target_class)}"
        f"__split{int(data_split_seed)}.manifest.json"
    )
    return Path(manifest_root) / _safe_component(dataset_name) / filename


def write_manifest_atomic(manifest: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically persist a manifest so concurrent array workers cannot truncate it."""

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
            # A hard link publishes the fully fsynced temporary file only if
            # the logical manifest name is still absent.  Unlike os.replace,
            # this cannot overwrite a concurrently created reference split.
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


def persist_or_validate_manifest(manifest: Mapping[str, Any], path: str | Path) -> tuple[dict[str, Any], Path]:
    """Reuse an identical manifest or atomically create it.

    A different manifest at the same logical path signals a changed dataset or
    split configuration and is rejected rather than silently replacing the
    reference images used by prior runs.
    """

    destination = Path(path)
    if destination.exists():
        existing = load_manifest(destination)
        if existing["manifest_hash"] != manifest["manifest_hash"]:
            raise ManifestError(
                f"Existing manifest at {destination} does not match the current dataset/split configuration "
                f"({existing['manifest_hash']} != {manifest['manifest_hash']})"
            )
        return existing, destination
    write_manifest_atomic(manifest, destination)
    # A concurrent writer may have replaced the same path.  Validate the final
    # file and require canonical identity in either case.
    final = load_manifest(destination)
    if final["manifest_hash"] != manifest["manifest_hash"]:
        raise ManifestError(f"Concurrent manifest creation produced a different split at {destination}")
    return final, destination


def unique_combination_count(group_sizes: Mapping[str, tuple[int, int]]) -> int:
    """Return the number of class draws implied by ``(pool_size, draw_size)`` pairs."""

    total = 1
    for pool_size, draw_size in group_sizes.values():
        total *= math.comb(int(pool_size), int(draw_size))
    return total
