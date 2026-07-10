from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import traceback as traceback_module
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

RESULT_SCHEMA_VERSION = 1
FAILURE_SCHEMA_VERSION = 1


def ensure_dir(path: str | Path) -> Path:
    value = Path(path)
    value.mkdir(parents=True, exist_ok=True)
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except (ValueError, TypeError):
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(obj: Any, path: str | Path) -> Path:
    """Write standards-compliant JSON by fsync + same-directory atomic rename."""

    destination = Path(path)
    ensure_dir(destination.parent)
    payload = _json_safe(obj)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temp_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
        temp_name = None
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:  # pragma: no cover - filesystem dependent
            pass
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
    return destination


def write_json(obj: Any, path: str | Path) -> None:
    """Backward-compatible alias; JSON writes are now atomic."""

    atomic_write_json(obj, path)


def canonical_json_hash(value: Any) -> str:
    serialized = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_sha(cwd: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def validate_result_record(record: Mapping[str, Any], *, expected_run_id: str | None = None) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("Run result must be a JSON object")
    version = record.get("schema_version", record.get("result_schema_version"))
    if version != RESULT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported run-result schema version {version!r}")
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Run result has no non-empty run_id")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError(f"Run result id {run_id!r} does not match expected {expected_run_id!r}")
    status = record.get("status")
    if status not in {"completed", "skipped"}:
        raise ValueError("Run-result status must be 'completed' or 'skipped'")
    if not isinstance(record.get("job"), Mapping) or not isinstance(record.get("metadata"), Mapping):
        raise ValueError("Run result must contain job and metadata objects")
    job = record["job"]
    metadata = record["metadata"]
    for key in ("experiment", "target_synset", "model_type", "training_protocol"):
        if job.get(key) in {None, ""}:
            raise ValueError(f"Run-result job is missing required field {key!r}")
    if status == "skipped":
        if metadata.get("skip_reason") in {None, ""}:
            raise ValueError("Skipped result must record skip_reason")
        return

    if not isinstance(record.get("training"), Mapping) or not isinstance(record.get("metrics"), Mapping):
        raise ValueError("Completed result must contain training and metrics objects")
    training = record["training"]
    metrics = record["metrics"]
    for key in (
        "manifest_hash",
        "config_hash",
        "git_sha",
        "selected_checkpoint_path",
        "target_eval_indices_hash",
    ):
        if metadata.get(key) in {None, ""}:
            raise ValueError(f"Completed result metadata is missing required field {key!r}")
    for key in ("optimizer_steps", "target_examples_seen", "auxiliary_examples_seen", "total_examples_seen"):
        if key not in training:
            raise ValueError(f"Completed result training section is missing required field {key!r}")
    for key in (
        "evaluation_mode",
        "num_generated",
        "num_real_eval",
        "test_epsilon_mse_target",
        "test_corruption_bank_hash",
    ):
        if key not in metrics:
            raise ValueError(f"Completed result metrics section is missing required field {key!r}")
    if metrics.get("evaluation_mode") == "debug" and "debug_pooled_pixel_distance" not in metrics:
        raise ValueError("Debug result is missing debug_pooled_pixel_distance")
    if metrics.get("evaluation_mode") == "paper" and metrics.get("metric_backend") in {None, ""}:
        raise ValueError("Paper result is missing metric_backend provenance")


def normalize_result_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten canonical or legacy-flat result JSON for tabular analysis."""

    flat: dict[str, Any] = {}
    for section in ("job", "metadata", "training", "metrics"):
        value = record.get(section)
        if isinstance(value, Mapping):
            flat.update(value)
    for key, value in record.items():
        if key not in {"job", "metadata", "training", "metrics", "config"}:
            flat[key] = value
    flat.setdefault("schema_version", record.get("result_schema_version", RESULT_SCHEMA_VERSION))
    return flat


def run_result_path(results_root: str | Path, run_id: str) -> Path:
    return Path(results_root) / "run_results" / f"{run_id}.json"


def failure_result_path(results_root: str | Path, run_id: str) -> Path:
    return Path(results_root) / "failures" / f"{run_id}.json"


def write_run_result(results_root: str | Path, run_id: str, result: Mapping[str, Any]) -> Path:
    record = dict(result)
    record["schema_version"] = RESULT_SCHEMA_VERSION
    record["run_id"] = run_id
    record.setdefault("status", "completed")
    record.setdefault("completed_at", utc_timestamp())
    validate_result_record(record, expected_run_id=run_id)
    return atomic_write_json(record, run_result_path(results_root, run_id))


def write_failure_result(
    results_root: str | Path,
    run_id: str,
    exception: BaseException,
    *,
    config: Mapping[str, Any] | None = None,
    job: Mapping[str, Any] | None = None,
    git_sha: str | None = None,
    traceback_text: str | None = None,
) -> Path:
    record = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "status": "failed",
        "run_id": run_id,
        "exception_type": type(exception).__name__,
        "message": str(exception),
        "traceback": traceback_text or "".join(traceback_module.format_exception(exception)),
        "config": dict(config or {}),
        "job": dict(job or {}),
        "git_sha": git_sha or get_git_sha(),
        "timestamp": utc_timestamp(),
    }
    return atomic_write_json(record, failure_result_path(results_root, run_id))


def load_valid_result(path: str | Path, *, expected_run_id: str | None = None) -> dict[str, Any]:
    record = load_json(path)
    validate_result_record(record, expected_run_id=expected_run_id)
    return record


def append_csv_row(path: str | Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    """Legacy single-process CSV helper.

    SLURM workers must use :func:`write_run_result`; aggregation is the only
    supported concurrent workflow.
    """

    destination = Path(path)
    ensure_dir(destination.parent)
    exists = destination.exists()
    with open(destination, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def resolve_env_path(value: str | None, default: str | None = None) -> str | None:
    if value is None:
        value = default
    if value is None:
        return None
    text = str(value)

    def repl(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        return os.environ.get(name, fallback or "")

    text = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}", repl, text)
    return os.path.expandvars(os.path.expanduser(text))
