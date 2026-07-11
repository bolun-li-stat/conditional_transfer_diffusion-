"""Emit a machine-readable runtime, lock, and device audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version
import torch
import yaml

from image_transfer.utils.io import atomic_write_json, canonical_json_hash, get_git_sha


ENVIRONMENT_REPORT_SCHEMA_VERSION = "2.0"
PACKAGES = [
    "torch", "torchvision", "torchmetrics", "torch-fidelity", "numpy", "scipy", "pandas",
    "matplotlib", "Pillow", "PyYAML", "pytest", "scikit-learn", "tqdm",
    "filelock", "fsspec", "jinja2", "networkx", "sympy", "typing-extensions", "packaging",
    "lightning-utilities", "mpmath", "MarkupSafe", "python-dateutil", "pytz", "tzdata", "contourpy",
    "cycler", "fonttools", "kiwisolver", "pyparsing", "pluggy", "iniconfig", "Pygments",
    "joblib", "threadpoolctl", "six", "setuptools",
]
_CONDA_NAME_MAP = {"pytorch": "torch", "python": "python", "pytorch-cuda": "pytorch-cuda"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_exact_requirement(line: str) -> tuple[str, str] | None:
    value = line.split("#", 1)[0].strip()
    if not value or value.startswith(("-", "http:", "https:")):
        return None
    match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^;\s]+)(?:\s*;.*)?", value)
    if match is None:
        return None
    return _canonical_name(match.group(1)), match.group(2)


def _parse_environment_definition(path: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Return exact versions, optional conda build constraints, and unparsed entries."""

    expected: dict[str, str] = {}
    builds: dict[str, str] = {}
    unparsed: list[str] = []
    if path.suffix.lower() not in {".yaml", ".yml"}:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            value = raw_line.split("#", 1)[0].strip()
            if not value or value.startswith(("--index-url", "--extra-index-url", "-f", "--find-links")):
                continue
            parsed = _parse_exact_requirement(raw_line)
            if parsed is None:
                unparsed.append(value)
            else:
                expected[parsed[0]] = parsed[1]
        return expected, builds, unparsed

    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    dependencies = document.get("dependencies", []) if isinstance(document, dict) else []
    if not isinstance(dependencies, list):
        return {}, {}, ["dependencies must be a list"]
    for item in dependencies:
        if isinstance(item, str):
            value = item.strip()
            if _canonical_name(value) in {"pip"}:
                continue
            parts = value.split("=", 2)
            if len(parts) < 2 or not parts[1]:
                unparsed.append(value)
                continue
            raw_name, version = parts[0], parts[1]
            name = _CONDA_NAME_MAP.get(_canonical_name(raw_name), _canonical_name(raw_name))
            expected[name] = version
            if len(parts) == 3 and parts[2]:
                builds[name] = parts[2]
        elif isinstance(item, dict) and set(item) == {"pip"} and isinstance(item["pip"], list):
            for raw_line in item["pip"]:
                parsed = _parse_exact_requirement(str(raw_line))
                if parsed is None:
                    unparsed.append(f"pip:{raw_line}")
                else:
                    expected[parsed[0]] = parsed[1]
        else:
            unparsed.append(str(item))
    return expected, builds, unparsed


def _installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "missing"
    return packages


def _public_version(value: str) -> str:
    try:
        return Version(value).public
    except InvalidVersion:
        return value


def _compare_lock(
    expected: dict[str, str],
    builds: dict[str, str],
    packages: dict[str, str],
) -> list[str]:
    runtime_by_name = {_canonical_name(name): version for name, version in packages.items()}
    mismatches: list[str] = []
    for name, required in sorted(expected.items()):
        if name == "python":
            actual = platform.python_version()
            if actual == required or _public_version(actual) == _public_version(required):
                continue
        elif name == "pytorch-cuda":
            actual = str(torch.version.cuda or "missing")
            if actual != "missing" and (actual == required or actual.startswith(required + ".")):
                continue
        else:
            actual = runtime_by_name.get(name, "missing")
            build = builds.get(name, "")
            if build:
                public_matches = _public_version(actual) == _public_version(required)
                build_matches = ("cpu" not in build.lower()) or ("cpu" in actual.lower())
                if public_matches and build_matches:
                    continue
            elif actual == required or _public_version(actual) == _public_version(required):
                continue
        mismatches.append(f"{name}: expected {required}, found {actual}")
    return mismatches


def inspect_environment(lock_path: str | Path) -> dict[str, Any]:
    lock = Path(lock_path).expanduser().resolve()
    packages = _installed_packages()
    driver = "unavailable"
    if torch.cuda.is_available():
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip().splitlines()[0]
        except (OSError, IndexError, subprocess.SubprocessError):
            driver = "unavailable"

    expected: dict[str, str] = {}
    builds: dict[str, str] = {}
    unparsed: list[str] = []
    if lock.is_file():
        try:
            expected, builds, unparsed = _parse_environment_definition(lock)
        except (OSError, yaml.YAMLError) as exception:
            unparsed = [f"definition parse failed: {type(exception).__name__}: {exception}"]
    mismatches = _compare_lock(expected, builds, packages) if expected else ["no exact package pins found"]
    if unparsed:
        mismatches.extend(f"unpinned or unsupported definition entry: {entry}" for entry in unparsed)

    lock_hash = file_sha256(lock) if lock.is_file() else "missing"
    runtime_identity = {
        "schema_version": ENVIRONMENT_REPORT_SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "packages": packages,
        "os": platform.platform(),
        "environment_lock_hash": lock_hash,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "driver_version": driver,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "tf32_matmul_allowed": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
        "amp_dtype": str(torch.get_autocast_dtype("cuda")) if torch.cuda.is_available() else "not_applicable",
    }
    report = {
        **runtime_identity,
        "python_executable": sys.executable,
        "os": platform.platform(),
        "git_sha": get_git_sha(),
        "git_dirty": bool(
            subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        ),
        "environment_lock_path": str(lock),
        "lock_expected_versions": expected,
        "lock_expected_builds": builds,
        "lock_unparsed_entries": unparsed,
        "lock_matches_runtime": lock_hash != "missing" and not mismatches,
        "lock_mismatches": mismatches,
        "driver_version": driver,
        "torch_home": os.environ.get("TORCH_HOME", ""),
        "environment_runtime_hash": canonical_json_hash(runtime_identity),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, help="Exact requirements lock or conda environment definition")
    parser.add_argument("--out")
    args = parser.parse_args()
    report = inspect_environment(args.lock)
    if args.out:
        atomic_write_json(report, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
