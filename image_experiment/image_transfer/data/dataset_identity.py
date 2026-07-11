"""Frozen, content-addressed identities for image datasets.

Strict experiment grids are allowed to run only when the bytes that make up
the configured dataset have been inventoried ahead of time.  Paths stored in
the identity are relative to ``data_root`` so the same identity remains usable
after staging the dataset on another machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from image_transfer.utils.io import resolve_env_path

from .manifests import canonical_sha256


DATASET_IDENTITY_SCHEMA_VERSION = "1.0"


class DatasetIdentityError(RuntimeError):
    """Raised when a frozen dataset identity is missing, malformed, or stale."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configured_classes(config: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    for target in config.get("targets", []):
        target_label = target.get("synset") or target.get("name")
        if target_label:
            labels.append(str(target_label))
        for candidates in (target.get("auxiliary_sets") or {}).values():
            labels.extend(map(str, candidates))
    return sorted(set(labels))


def _inventory_files(root: Path, config: Mapping[str, Any]) -> list[Path]:
    dataset_name = str(config.get("dataset", "")).lower()
    classes = _configured_classes(config)
    files: list[Path] = []
    if dataset_name.startswith("imagenet") or any((root / split).is_dir() for split in ("train", "val")):
        for split in ("train", "val"):
            split_root = root / split
            if not split_root.is_dir():
                continue
            for label in classes:
                class_root = split_root / label
                if class_root.is_dir():
                    files.extend(path for path in class_root.rglob("*") if path.is_file())
    else:
        files.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def _identity_payload(
    config: Mapping[str, Any],
    *,
    root: Path | None,
    inventory: Sequence[Mapping[str, Any]],
    identity_kind: str,
) -> dict[str, Any]:
    inventory_payload = [dict(item) for item in inventory]
    content_hash = canonical_sha256(inventory_payload)
    payload: dict[str, Any] = {
        "schema_version": DATASET_IDENTITY_SCHEMA_VERSION,
        "identity_kind": str(identity_kind),
        "dataset_name": str(config.get("dataset", "unknown")),
        "classes": _configured_classes(config),
        "inventory": inventory_payload,
        "inventory_count": len(inventory_payload),
        "dataset_content_hash": content_hash,
        "portable_root": "data_root",
    }
    if identity_kind == "synthetic":
        payload["synthetic_specification"] = {
            "fake_data_size": int(config.get("fake_data_size", 100000)),
            "image_size": int(config.get("image_size", 32)),
            "fake_data_seed": int(config.get("fake_data_seed", 0)),
        }
        payload["dataset_content_hash"] = canonical_sha256(payload["synthetic_specification"] | {
            "dataset_name": payload["dataset_name"],
            "classes": payload["classes"],
        })
    payload["dataset_identity_hash"] = canonical_sha256(payload)
    return payload


def build_dataset_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compute the current dataset identity without writing it."""

    if bool(config.get("use_fake_data", False)):
        return _identity_payload(config, root=None, inventory=[], identity_kind="synthetic")
    expanded = resolve_env_path(config.get("data_root"), "data")
    root = Path(expanded).expanduser().resolve()
    if not root.is_dir():
        raise DatasetIdentityError(f"Dataset root does not exist or is not a directory: {root}")
    paths = _inventory_files(root, config)
    if not paths:
        raise DatasetIdentityError(
            "No files were found for the configured dataset classes beneath data_root"
        )
    dataset_name = str(config.get("dataset", "")).lower()
    if dataset_name.startswith("imagenet") or (root / "train").is_dir():
        relative_paths = [path.relative_to(root).parts for path in paths]
        missing_train = [
            label
            for label in _configured_classes(config)
            if not any(len(parts) >= 3 and parts[0] == "train" and parts[1] == label for parts in relative_paths)
        ]
        if missing_train:
            raise DatasetIdentityError(
                f"Configured classes have no training files in data_root: {missing_train}"
            )
        eval_source = str(config.get("data_split", {}).get("eval_source", "train_holdout"))
        if eval_source != "train_holdout":
            missing_eval = [
                label
                for label in _configured_classes(config)
                if not any(
                    len(parts) >= 3 and parts[0] == eval_source and parts[1] == label
                    for parts in relative_paths
                )
            ]
            if missing_eval:
                raise DatasetIdentityError(
                    f"Configured classes have no {eval_source} files in data_root: {missing_eval}"
                )
    inventory = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]
    return _identity_payload(config, root=root, inventory=inventory, identity_kind="file_inventory")


def validate_dataset_identity(identity: Mapping[str, Any]) -> None:
    """Validate an identity's schema and both content-addressed hashes."""

    if identity.get("schema_version") != DATASET_IDENTITY_SCHEMA_VERSION:
        raise DatasetIdentityError("Unsupported dataset-identity schema")
    kind = str(identity.get("identity_kind", ""))
    if kind not in {"file_inventory", "synthetic"}:
        raise DatasetIdentityError(f"Unsupported dataset identity kind {kind!r}")
    inventory = identity.get("inventory")
    if not isinstance(inventory, list):
        raise DatasetIdentityError("Dataset identity inventory must be a list")
    if int(identity.get("inventory_count", -1)) != len(inventory):
        raise DatasetIdentityError("Dataset identity inventory_count is inconsistent")
    if kind == "file_inventory":
        seen: set[str] = set()
        for item in inventory:
            if not isinstance(item, Mapping):
                raise DatasetIdentityError("Dataset identity inventory entries must be mappings")
            relative = str(item.get("relative_path", ""))
            candidate = Path(relative)
            if not relative or candidate.is_absolute() or ".." in candidate.parts:
                raise DatasetIdentityError(f"Unsafe dataset inventory path {relative!r}")
            if relative in seen:
                raise DatasetIdentityError(f"Duplicate dataset inventory path {relative!r}")
            seen.add(relative)
            if int(item.get("size_bytes", -1)) < 0 or len(str(item.get("sha256", ""))) != 64:
                raise DatasetIdentityError(f"Invalid dataset inventory entry for {relative!r}")
        expected_content = canonical_sha256([dict(item) for item in inventory])
    else:
        specification = identity.get("synthetic_specification")
        if not isinstance(specification, Mapping):
            raise DatasetIdentityError("Synthetic dataset identity is missing its specification")
        expected_content = canonical_sha256(dict(specification) | {
            "dataset_name": str(identity.get("dataset_name", "unknown")),
            "classes": list(identity.get("classes", [])),
        })
    if identity.get("dataset_content_hash") != expected_content:
        raise DatasetIdentityError("Dataset content hash does not match the frozen inventory")
    unhashed = {key: value for key, value in identity.items() if key != "dataset_identity_hash"}
    expected_identity = canonical_sha256(unhashed)
    if identity.get("dataset_identity_hash") != expected_identity:
        raise DatasetIdentityError("Dataset identity hash mismatch")


def load_dataset_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise DatasetIdentityError(f"Could not read dataset identity {source}: {exception}") from exception
    if not isinstance(payload, Mapping):
        raise DatasetIdentityError(f"Dataset identity {source} must contain a JSON object")
    identity = dict(payload)
    validate_dataset_identity(identity)
    return identity


def verify_dataset_identity(config: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute current bytes/specification and require an exact frozen match."""

    validate_dataset_identity(identity)
    current = build_dataset_identity(config)
    if str(identity.get("dataset_name")) != str(current.get("dataset_name")):
        raise DatasetIdentityError("Frozen dataset name does not match the configured dataset")
    if list(identity.get("classes", [])) != list(current.get("classes", [])):
        raise DatasetIdentityError("Frozen dataset classes do not match the configured target/auxiliary classes")
    if identity.get("dataset_identity_hash") != current.get("dataset_identity_hash"):
        raise DatasetIdentityError(
            "Current dataset bytes or inventory do not match the frozen dataset identity"
        )
    return current


def verify_dataset_identity_file(config: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    return verify_dataset_identity(config, load_dataset_identity(path))


def write_dataset_identity_atomic(identity: Mapping[str, Any], path: str | Path) -> Path:
    validate_dataset_identity(identity)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(dict(identity), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return destination


def freeze_dataset_identity(config: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    identity = build_dataset_identity(config)
    write_dataset_identity_atomic(identity, path)
    return identity
