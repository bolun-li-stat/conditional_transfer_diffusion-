"""Exact, portable runtime-lock creation and verification helpers.

The checked-in environment YAML is an installation specification.  It is not
an observation of what a scheduler node actually installed.  This module
defines a small JSON lock format which records the complete conda and pip
package sets exposed by the environment in which it was generated.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from image_transfer.utils.io import canonical_json_hash, utc_timestamp


EXACT_ENVIRONMENT_LOCK_SCHEMA_VERSION = "1.0"


def canonical_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def installed_pip_packages() -> list[dict[str, str]]:
    """Return the complete installed distribution set in a stable form."""

    packages: dict[str, str] = {}
    # Distribution discovery follows sys.path precedence. Preserve the first
    # occurrence so an isolated target directory is not overwritten by a
    # duplicate package in the base interpreter.
    for distribution in importlib.metadata.distributions():
        if not distribution.metadata.get("Name"):
            continue
        packages.setdefault(
            canonical_package_name(distribution.metadata["Name"]),
            str(distribution.version),
        )
    return [{"name": name, "version": packages[name]} for name in sorted(packages)]


def conda_explicit_packages() -> list[str]:
    """Return exact conda package URLs, or an empty list outside conda."""

    try:
        completed = subprocess.run(
            ["conda", "list", "--explicit"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode:
        return []
    return sorted(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def build_exact_environment_lock(
    *,
    source_spec_path: str | Path | None = None,
    source_spec_hash: str = "missing",
    conda_packages: Sequence[str] | None = None,
    pip_packages: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    conda_values = sorted(str(value) for value in (conda_packages if conda_packages is not None else conda_explicit_packages()))
    pip_values = [
        {"name": canonical_package_name(str(item["name"])), "version": str(item["version"])}
        for item in (pip_packages if pip_packages is not None else installed_pip_packages())
    ]
    pip_values.sort(key=lambda item: item["name"])
    payload: dict[str, Any] = {
        "schema_version": EXACT_ENVIRONMENT_LOCK_SCHEMA_VERSION,
        "created_at": utc_timestamp(),
        "source_environment_spec_path": str(source_spec_path or ""),
        "source_environment_spec_hash": str(source_spec_hash),
        "python_version": platform.python_version(),
        "conda_explicit_packages": conda_values,
        "pip_packages": pip_values,
    }
    payload["lock_payload_hash"] = canonical_json_hash(
        {key: value for key, value in payload.items() if key not in {"created_at", "lock_payload_hash"}}
    )
    validate_exact_environment_lock(payload)
    return payload


def validate_exact_environment_lock(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != EXACT_ENVIRONMENT_LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported exact environment lock schema")
    pip_values = payload.get("pip_packages")
    conda_values = payload.get("conda_explicit_packages")
    if not isinstance(pip_values, list) or not pip_values:
        raise ValueError("exact environment lock must contain the complete pip package set")
    if not isinstance(conda_values, list):
        raise ValueError("exact environment lock conda package set must be a list")
    names: list[str] = []
    for entry in pip_values:
        if not isinstance(entry, Mapping) or not entry.get("name") or not entry.get("version"):
            raise ValueError("invalid pip package entry in exact environment lock")
        name = canonical_package_name(str(entry["name"]))
        if name != entry["name"]:
            raise ValueError("pip package names in exact environment lock must be canonical")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("duplicate pip package in exact environment lock")
    if list(conda_values) != sorted(str(value) for value in conda_values):
        raise ValueError("conda package URLs in exact environment lock must be sorted")
    expected_hash = canonical_json_hash(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_at", "lock_payload_hash"}
        }
    )
    if payload.get("lock_payload_hash") != expected_hash:
        raise ValueError("exact environment lock payload hash mismatch")


def load_exact_environment_lock(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("exact environment lock must be a JSON object")
    validate_exact_environment_lock(payload)
    return payload


def compare_exact_environment_lock(
    payload: Mapping[str, Any],
    *,
    pip_packages: Sequence[Mapping[str, Any]] | None = None,
    conda_packages: Sequence[str] | None = None,
) -> list[str]:
    """Compare complete package sets, including unexpected runtime packages."""

    validate_exact_environment_lock(payload)
    expected_pip = {
        canonical_package_name(str(item["name"])): str(item["version"])
        for item in payload["pip_packages"]
    }
    actual_values = pip_packages if pip_packages is not None else installed_pip_packages()
    actual_pip = {
        canonical_package_name(str(item["name"])): str(item["version"])
        for item in actual_values
    }
    mismatches: list[str] = []
    for name in sorted(set(expected_pip) | set(actual_pip)):
        expected = expected_pip.get(name)
        actual = actual_pip.get(name)
        if expected is None:
            mismatches.append(f"unexpected pip package {name}=={actual}")
        elif actual is None:
            mismatches.append(f"missing pip package {name}=={expected}")
        elif actual != expected:
            mismatches.append(f"pip package {name}: expected {expected}, found {actual}")

    expected_conda = set(str(value) for value in payload["conda_explicit_packages"])
    actual_conda = set(str(value) for value in (conda_packages if conda_packages is not None else conda_explicit_packages()))
    if expected_conda or actual_conda:
        for value in sorted(expected_conda - actual_conda):
            mismatches.append(f"missing conda package URL {value}")
        for value in sorted(actual_conda - expected_conda):
            mismatches.append(f"unexpected conda package URL {value}")
    if platform.python_version() != str(payload.get("python_version")):
        mismatches.append(
            f"python: expected {payload.get('python_version')}, found {platform.python_version()}"
        )
    return mismatches


__all__ = [
    "EXACT_ENVIRONMENT_LOCK_SCHEMA_VERSION",
    "build_exact_environment_lock",
    "canonical_package_name",
    "compare_exact_environment_lock",
    "conda_explicit_packages",
    "installed_pip_packages",
    "load_exact_environment_lock",
    "validate_exact_environment_lock",
]
