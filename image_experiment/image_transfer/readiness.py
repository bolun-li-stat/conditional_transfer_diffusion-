"""Readiness status loading and main-stage gate checks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from image_transfer.config import load_resolved_config
from image_transfer.models.model_factory import model_config_hash
from image_transfer.utils.io import get_git_sha


def load_readiness_status(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {"status": "not_run", "reason": f"missing status file: {source}"}
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("status") not in {"not_run", "failed", "passed"}:
        raise ValueError(f"Invalid readiness status in {source}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enforce_readiness_gate(
    config: Mapping[str, Any],
    *,
    override: bool = False,
    current_git_sha: str | None = None,
    config_source_path: str | Path | None = None,
) -> dict[str, Any]:
    if str(config.get("study_stage", "pilot")) != "main":
        return {
            "required": False,
            "override": False,
            "status": "not_applicable",
            "passed": True,
            "mismatches": [],
        }

    base_dir = Path(config_source_path).resolve().parent if config_source_path else Path.cwd()

    def resolve_path(value: Any) -> Path:
        candidate = Path(str(value)).expanduser()
        return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()

    path = resolve_path(config.get("readiness_status_path", "readiness/pilot_status.json"))
    pilot_config_value = config.get("readiness_pilot_config_path")
    if not pilot_config_value:
        raise RuntimeError("Main-stage readiness gate requires readiness_pilot_config_path")
    pilot_path = resolve_path(pilot_config_value)
    try:
        pilot = load_resolved_config(pilot_path)
    except Exception as exception:
        raise RuntimeError(f"Unable to resolve linked release-pilot config {pilot_path}: {exception}") from exception
    if str(pilot.resolved.get("study_stage")) != "pilot":
        raise RuntimeError("readiness_pilot_config_path must identify a pilot-stage config")

    status = load_readiness_status(path)
    status_hash = _file_sha256(path) if path.exists() else "missing"
    resolved_git_sha = str(current_git_sha or get_git_sha())
    current_model_hash = model_config_hash(dict(config.get("model") or {}))
    expected = {
        "git_sha": resolved_git_sha,
        "model_config_hash": pilot.model_hash,
        "target_set_hash": pilot.target_set_hash,
        "environment_lock_hash": pilot.environment_lock_hash,
        "study_plan_hash": pilot.study_plan_hash,
        "pilot_config_hash": pilot.resolved_hash,
    }
    mismatches = [key for key, value in expected.items() if status.get(key) != value]
    if status.get("status") == "passed":
        expected_jobs = status.get("expected_jobs")
        evidence_valid = (
            status.get("schema_version") == "3.0"
            and type(expected_jobs) is int
            and expected_jobs > 0
            and status.get("validated_jobs") == expected_jobs
            and status.get("resume_state_validated_jobs") == expected_jobs
            and status.get("checkpoint_artifacts_validated_jobs") == expected_jobs
            and status.get("sample_artifacts_validated_jobs") == expected_jobs
            and status.get("metric_artifacts_validated_jobs") == expected_jobs
            and status.get("nearest_neighbor_artifacts_validated_jobs") == expected_jobs
            and status.get("figure_artifacts_validated_jobs") == expected_jobs
            and status.get("provenance_artifacts_validated_jobs") == expected_jobs
            and type(status.get("validated_pairs")) is int
            and status.get("validated_pairs", 0) > 0
            and isinstance(status.get("validated_result_hashes"), Mapping)
            and len(status["validated_result_hashes"]) == expected_jobs
            and all(
                re.fullmatch(r"[0-9a-fA-F]{64}", str(value))
                for value in status["validated_result_hashes"].values()
            )
            and status.get("failures") == []
            and bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(status.get("expected_job_grid_hash", ""))))
            and bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(status.get("jobs_csv_hash", ""))))
            and bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(status.get("environment_runtime_hash", ""))))
            and bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(status.get("environment_report_hash", ""))))
            and bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(status.get("exact_environment_lock_hash", ""))))
            and bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(status.get("gpu_load_probe_hash", ""))))
            and bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(status.get("resume_probe_hash", ""))))
        )
        if not evidence_valid:
            mismatches.append("validation_evidence")
    if current_model_hash != pilot.model_hash:
        mismatches.append("main_model_config_hash")
    if config.get("environment_lock_hash") != pilot.environment_lock_hash:
        mismatches.append("main_environment_lock_hash")
    if config.get("study_plan_hash") != pilot.study_plan_hash:
        mismatches.append("main_study_plan_hash")
    ready = status.get("status") == "passed" and not mismatches
    if not ready and not override:
        raise RuntimeError(
            f"Main-stage readiness gate failed: status={status.get('status')}, mismatches={mismatches}; "
            "complete the real-data pilot or use the explicit override"
        )
    return {
        "required": True,
        "override": bool(override),
        "status": status.get("status"),
        "passed": bool(ready),
        "mismatches": mismatches,
        "status_path": str(config.get("readiness_status_path", "readiness/pilot_status.json")),
        "status_file_hash": status_hash,
        "pilot_config_path": str(pilot_config_value),
        "pilot_config_hash": pilot.resolved_hash,
        "pilot_target_set_hash": pilot.target_set_hash,
        "pilot_model_config_hash": pilot.model_hash,
        "pilot_environment_lock_hash": pilot.environment_lock_hash,
        "validated_git_sha": status.get("git_sha", ""),
        "current_git_sha": resolved_git_sha,
    }
