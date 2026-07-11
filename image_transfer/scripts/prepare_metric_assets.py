"""Prepare and verify the two pretrained evaluation backends used by strict runs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import socket
from typing import Any, Iterator

import torch

from image_transfer.evaluation.classifier_fidelity import preflight_classifier_fidelity
from image_transfer.evaluation.feature_metrics import preflight_feature_metric_backend
from image_transfer.utils.io import atomic_write_json


ASSET_MANIFEST_SCHEMA_VERSION = "2.0"
RUNTIME_PACKAGES = ("torch", "torchvision", "torchmetrics", "torch-fidelity")
REQUIRED_ASSET_PATHS = {
    "inception_features": "hub/checkpoints/weights-inception-2015-12-05-6726825d.pth",
    "imagenet_classifier": "hub/checkpoints/resnet50-11ad3fa6.pth",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _versions() -> dict[str, str]:
    result = {}
    for name in RUNTIME_PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def _repository_root() -> Path | None:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / ".git").exists():
            return candidate
    return None


def _validated_asset_root(value: str | Path) -> Path:
    if not str(value).strip():
        raise ValueError("asset root must be an explicit non-empty path")
    root = Path(value).expanduser().resolve()
    repository = _repository_root()
    forbidden = {Path(root.anchor).resolve(), Path.cwd().resolve()}
    if repository is not None:
        forbidden.add(repository.resolve())
    if root in forbidden:
        raise ValueError("asset root must be a dedicated directory, not the filesystem or repository root")
    return root


def _validated_manifest_path(value: str | Path) -> Path:
    if not str(value).strip():
        raise ValueError("manifest path must be an explicit non-empty path")
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_file():
        raise ValueError(f"manifest path is not a file: {path}")
    return path


def _safe_relative_path(value: Any) -> PurePosixPath:
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or text != path.as_posix()
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe asset manifest path: {text!r}")
    return path


def _asset_entry(root: Path, *, role: str, relative_path: str) -> dict[str, Any]:
    relative = _safe_relative_path(relative_path)
    path = root.joinpath(*relative.parts)
    if not path.is_file():
        raise FileNotFoundError(f"required metric asset for {role!r} is missing: {path}")
    return {
        "role": role,
        "path": relative.as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": _sha(path),
    }


def build_manifest(asset_root: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    """Build a portable manifest for the exact known Torch cache layout.

    ``manifest_path`` is mandatory so a custom-named manifest can never be
    accidentally included among the files it describes.
    """

    root = _validated_asset_root(asset_root)
    destination = _validated_manifest_path(manifest_path)
    assets = [
        _asset_entry(root, role=role, relative_path=relative_path)
        for role, relative_path in sorted(REQUIRED_ASSET_PATHS.items())
    ]
    if any((root / entry["path"]).resolve() == destination for entry in assets):
        raise ValueError("manifest path must not replace a required metric asset")
    payload: dict[str, Any] = {
        "schema_version": ASSET_MANIFEST_SCHEMA_VERSION,
        "asset_root_layout": "torch_home",
        "runtime_packages": _versions(),
        "required_roles": sorted(REQUIRED_ASSET_PATHS),
        "assets": assets,
        # Kept as a read-only convenience for older tooling. Both fields are
        # validated to be identical when loading schema 2.0.
        "files": assets,
    }
    payload["manifest_hash"] = _canonical_hash(payload)
    return payload


def _validate_manifest_structure(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise RuntimeError("Metric asset manifest must be a JSON object")
    if manifest.get("schema_version") != ASSET_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"Metric asset manifest schema must be {ASSET_MANIFEST_SCHEMA_VERSION!r}; "
            f"got {manifest.get('schema_version')!r}"
        )
    recorded_hash = manifest.get("manifest_hash")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if not isinstance(recorded_hash, str) or recorded_hash != _canonical_hash(unhashed):
        raise RuntimeError("Metric asset manifest self-hash does not match its content")
    if manifest.get("asset_root_layout") != "torch_home":
        raise RuntimeError("Metric asset manifest has an unsupported asset-root layout")
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise RuntimeError("Metric asset manifest contains no assets")
    if manifest.get("files") != assets:
        raise RuntimeError("Metric asset manifest files/assets fields disagree")
    roles = manifest.get("required_roles")
    if roles != sorted(REQUIRED_ASSET_PATHS):
        raise RuntimeError("Metric asset manifest does not declare every required backend role")
    return manifest


def verify_manifest(
    asset_root: str | Path,
    manifest_path: str | Path,
    *,
    verify_runtime: bool = False,
) -> dict[str, Any]:
    """Verify manifest integrity, safe paths, assets, roles, and optionally packages."""

    root = _validated_asset_root(asset_root)
    source = _validated_manifest_path(manifest_path)
    try:
        manifest = _validate_manifest_structure(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exception:
        raise RuntimeError(f"Metric asset manifest could not be read: {exception}") from exception

    failures: list[str] = []
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    for raw_entry in manifest["assets"]:
        if not isinstance(raw_entry, dict):
            failures.append("invalid asset entry")
            continue
        role = str(raw_entry.get("role", ""))
        try:
            relative = _safe_relative_path(raw_entry.get("path", ""))
        except ValueError as exception:
            failures.append(str(exception))
            continue
        expected_path = REQUIRED_ASSET_PATHS.get(role)
        if expected_path is None:
            failures.append(f"unknown-role:{role}")
        elif relative.as_posix() != expected_path:
            failures.append(f"role-path:{role}")
        if role in seen_roles:
            failures.append(f"duplicate-role:{role}")
        if relative.as_posix() in seen_paths:
            failures.append(f"duplicate-path:{relative.as_posix()}")
        seen_roles.add(role)
        seen_paths.add(relative.as_posix())
        path = root.joinpath(*relative.parts)
        if path.resolve() == source:
            failures.append(f"manifest-self-inclusion:{relative.as_posix()}")
        elif not path.is_file():
            failures.append(f"missing:{relative.as_posix()}")
        elif path.stat().st_size != int(raw_entry.get("bytes", -1)):
            failures.append(f"size:{relative.as_posix()}")
        elif _sha(path) != raw_entry.get("sha256"):
            failures.append(f"sha256:{relative.as_posix()}")
    missing_roles = set(REQUIRED_ASSET_PATHS) - seen_roles
    failures.extend(f"missing-role:{role}" for role in sorted(missing_roles))

    recorded_versions = manifest.get("runtime_packages")
    if not isinstance(recorded_versions, dict) or set(recorded_versions) != set(RUNTIME_PACKAGES):
        failures.append("runtime package version set is incomplete")
    elif verify_runtime:
        runtime_versions = _versions()
        failures.extend(
            f"runtime-version:{package}:expected={recorded_versions[package]}:actual={runtime_versions[package]}"
            for package in RUNTIME_PACKAGES
            if runtime_versions[package] != recorded_versions[package]
        )
    if failures:
        raise RuntimeError("Metric asset verification failed: " + ", ".join(failures))
    return manifest


@contextmanager
def _network_disabled() -> Iterator[None]:
    original_connect = socket.socket.connect

    def blocked_connect(self, address):  # noqa: ANN001, ARG001
        raise RuntimeError("network access is disabled during metric backend initialization")

    socket.socket.connect = blocked_connect
    try:
        yield
    finally:
        socket.socket.connect = original_connect


def initialize_backends_offline(asset_root: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    """Initialize both strict backends while all socket connections are blocked."""

    root = _validated_asset_root(asset_root)
    os.environ["TORCH_HOME"] = str(root)
    torch.hub.set_dir(str(root / "hub"))
    with _network_disabled():
        feature = preflight_feature_metric_backend(device=device)
        classifier = preflight_classifier_fidelity("n02108915", [], dataset_name="imagenet64")
    return {"feature_backend": feature, "classifier_backend": classifier}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True, help="Dedicated TORCH_HOME directory")
    parser.add_argument("--manifest", required=True, help="Explicit output/input manifest JSON path")
    parser.add_argument("--offline-check", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = _validated_asset_root(args.asset_root)
    manifest_path = _validated_manifest_path(args.manifest)
    root.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(root)
    torch.hub.set_dir(str(root / "hub"))
    if args.offline_check:
        report = verify_manifest(root, manifest_path, verify_runtime=True)
        report = {**report, "offline_initialization": initialize_backends_offline(root, device=args.device)}
    else:
        preflight_feature_metric_backend(device=args.device)
        preflight_classifier_fidelity("n02108915", [], dataset_name="imagenet64")
        report = build_manifest(root, manifest_path)
        atomic_write_json(report, manifest_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
