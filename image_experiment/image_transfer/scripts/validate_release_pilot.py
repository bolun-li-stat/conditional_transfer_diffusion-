"""Validate the exact engineering contract of a completed real-data pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from image_transfer.config import ResolvedConfig, load_resolved_config
from image_transfer.data.manifests import canonical_sha256
from image_transfer.evaluation.corruption_bank import load_corruption_bank
from image_transfer.scripts.make_job_grid import JOB_FIELDS, RUN_SPEC_FIELDS, rows_for_experiment
from image_transfer.utils.io import atomic_write_json, load_valid_result, resolve_env_path


_RESULT_JOB_EXCLUSIONS = {"config_path", "output_dir", "manifest_key"}
_RESULT_JOB_FIELDS = tuple(field for field in JOB_FIELDS if field not in _RESULT_JOB_EXCLUSIONS)
_FIXED_ARTIFACTS = {
    "last_checkpoint_path": ("checkpoints", "_last.pt"),
    "best_checkpoint_path": ("checkpoints", "_best.pt"),
    "selected_checkpoint_path": ("checkpoints", "_best.pt"),
    "sample_path": ("samples", "_samples.pt"),
    "run_config_path": ("configs", ".yaml"),
    "nearest_neighbor_grid_path": ("figures", "_nearest_neighbors.png"),
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_text(value: Any) -> str:
    return "" if value is None else str(value)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _enabled_experiments(config: Mapping[str, Any]) -> list[str]:
    declared = config.get("experiments", {})
    return [
        name
        for name in ("A", "B", "C")
        if name in declared and declared[name].get("enabled") is True
    ]


def rebuild_expected_jobs(config_path: str | Path, resolved: ResolvedConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in _enabled_experiments(resolved.resolved):
        rows.extend(
            rows_for_experiment(
                experiment,
                resolved.raw,
                str(config_path),
                resolved_info=resolved,
            )
        )
    return rows


def _read_and_compare_grid(
    jobs_csv: str | Path,
    expected_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[str]]:
    failures: list[str] = []
    with Path(jobs_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        jobs = list(reader)
    if fieldnames != JOB_FIELDS:
        failures.append("job_grid_header")
    if any(None in row for row in jobs):
        failures.append("job_grid_extra_columns")

    expected_by_id = {str(row["run_id"]): row for row in expected_rows}
    supplied_ids = [str(row.get("run_id", "")) for row in jobs]
    counts = Counter(supplied_ids)
    for run_id, count in sorted(counts.items()):
        if not run_id:
            failures.append("job_grid_missing_run_id")
        elif count != 1:
            failures.append(f"job_grid_duplicate:{run_id}")
    for run_id in sorted(set(expected_by_id) - set(supplied_ids)):
        failures.append(f"job_grid_missing:{run_id}")
    for run_id in sorted(set(supplied_ids) - set(expected_by_id) - {""}):
        failures.append(f"job_grid_unexpected:{run_id}")

    supplied_by_id = {str(row.get("run_id", "")): row for row in jobs if counts[str(row.get("run_id", ""))] == 1}
    for run_id in sorted(set(expected_by_id) & set(supplied_by_id)):
        expected = expected_by_id[run_id]
        supplied = supplied_by_id[run_id]
        for field in JOB_FIELDS:
            if supplied.get(field, "") != _csv_text(expected.get(field, "")):
                failures.append(f"job_grid_altered:{run_id}:{field}")
    if len(jobs) != len(expected_rows):
        failures.append(f"job_grid_count:{len(jobs)}:{len(expected_rows)}")
    return jobs, failures


def _check_file(path_value: Any, expected: Path | None, label: str, run_id: str, failures: list[str]) -> Path | None:
    if not path_value:
        failures.append(f"missing_artifact:{run_id}:{label}")
        return None
    path = Path(str(path_value))
    if expected is not None and path.resolve() != expected.resolve():
        failures.append(f"artifact_path:{run_id}:{label}")
        return None
    if not path.is_file() or path.stat().st_size <= 0:
        failures.append(f"missing_artifact:{run_id}:{label}")
        return None
    return path


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        return torch.load(path, map_location="cpu")


def _validate_checkpoint(
    path: Path,
    *,
    run_id: str,
    expected: Mapping[str, Any],
    metadata: Mapping[str, Any],
    git_sha: str,
    failures: list[str],
) -> bool:
    try:
        checkpoint = _load_torch(path)
    except Exception as exception:
        failures.append(f"invalid_checkpoint:{run_id}:{type(exception).__name__}")
        return False
    checks = {
        "step": int(expected["training_steps"]),
        "config_hash": expected["resolved_config_hash"],
        "manifest_hash": metadata.get("manifest_hash"),
        "git_sha": git_sha,
        "training_protocol": expected["training_protocol"],
    }
    for field, value in checks.items():
        if checkpoint.get(field) != value:
            failures.append(f"checkpoint_provenance:{run_id}:{field}")
    model_metadata = checkpoint.get("model_metadata") or {}
    if model_metadata.get("model_config_hash") != expected["model_config_hash"]:
        failures.append(f"checkpoint_provenance:{run_id}:model_config_hash")
    if int(checkpoint.get("checkpoint_schema_version", 0)) < 3:
        failures.append(f"checkpoint_schema:{run_id}")
    for field in ("raw_model_state", "ema_model_state", "optimizer_state", "rng_states", "data_state"):
        value = checkpoint.get(field)
        if not isinstance(value, Mapping) or not value:
            failures.append(f"checkpoint_resume_state:{run_id}:{field}")
    if str(expected.get("training_protocol")) and not isinstance(
        checkpoint.get("training_protocol_metadata"), Mapping
    ):
        failures.append(f"checkpoint_resume_state:{run_id}:training_protocol_metadata")
    rng = checkpoint.get("rng_states") or {}
    for name in ("python", "numpy", "torch_cpu", "torch_cuda"):
        if name not in rng:
            failures.append(f"checkpoint_resume_state:{run_id}:rng:{name}")
    data_state = checkpoint.get("data_state") or {}
    if int(data_state.get("global_step", -1)) != int(expected["training_steps"]):
        failures.append(f"checkpoint_resume_state:{run_id}:data_position")
    if "sampler_seed" not in data_state or "num_workers" not in data_state:
        failures.append(f"checkpoint_resume_state:{run_id}:sampler")
    if str(expected.get("training_protocol")) and checkpoint.get("grad_scaler_state") is None:
        failures.append(f"checkpoint_resume_state:{run_id}:grad_scaler_state")
    protocol = checkpoint.get("training_protocol_metadata") or {}
    for field in ("optimizer_steps", "target_examples_seen", "auxiliary_examples_seen", "total_examples_seen"):
        if field not in protocol:
            failures.append(f"checkpoint_resume_state:{run_id}:protocol:{field}")
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping):
        failures.append(f"checkpoint_provenance:{run_id}:provenance")
        provenance = {}
    for field in (
        "raw_config_hash",
        "resolved_config_hash",
        "model_config_hash",
        "study_plan_hash",
        "target_set_hash",
        "environment_lock_hash",
        "split_manifest_hash",
        "subset_manifest_hash",
        "run_id",
    ):
        expected_value = run_id if field == "run_id" else (
            metadata.get(field) if field in {"split_manifest_hash", "subset_manifest_hash"} else expected.get(field)
        )
        if provenance.get(field) != expected_value:
            failures.append(f"checkpoint_provenance:{run_id}:{field}")
    for field in ("raw_config", "resolved_config", "split_manifest", "subset_manifest", "environment_report"):
        if not isinstance(provenance.get(field), Mapping):
            failures.append(f"checkpoint_provenance:{run_id}:{field}")
    return not any(
        item.startswith((f"checkpoint_resume_state:{run_id}", f"checkpoint_provenance:{run_id}", f"checkpoint_schema:{run_id}"))
        for item in failures
    )


def _validate_artifacts(
    expected: Mapping[str, Any],
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    failures: list[str],
) -> bool:
    run_id = str(expected["run_id"])
    metadata = record["metadata"]
    output_dir = Path(str(expected["output_dir"]))
    paths: dict[str, Path | None] = {}
    neighbors_required = bool(config.get("evaluation", {}).get("make_nearest_neighbors", False))
    for field, (directory, suffix) in _FIXED_ARTIFACTS.items():
        if field == "nearest_neighbor_grid_path" and not neighbors_required:
            continue
        expected_path = output_dir / directory / f"{run_id}{suffix}"
        paths[field] = _check_file(metadata.get(field), expected_path, field, run_id, failures)

    sample_path = paths.get("sample_path")
    if sample_path is not None:
        try:
            samples = _load_torch(sample_path)
            expected_shape = (
                int(expected["num_generated"]),
                3,
                int(expected["image_size"]),
                int(expected["image_size"]),
            )
            if not isinstance(samples, torch.Tensor) or tuple(samples.shape) != expected_shape:
                failures.append(f"sample_shape:{run_id}")
            elif (
                not torch.isfinite(samples).all()
                or float(samples.std()) == 0.0
                or float(samples.min()) < -1.001
                or float(samples.max()) > 1.001
            ):
                failures.append(f"invalid_samples:{run_id}")
        except Exception as exception:
            failures.append(f"invalid_samples:{run_id}:{type(exception).__name__}")

    last_path = paths.get("last_checkpoint_path")
    resume_state_valid = False
    if last_path is not None:
        resume_state_valid = _validate_checkpoint(
            last_path,
            run_id=run_id,
            expected=expected,
            metadata=metadata,
            git_sha=str(metadata.get("git_sha", "")),
            failures=failures,
        )
    best_path = paths.get("best_checkpoint_path")
    if best_path is not None:
        _validate_checkpoint(
            best_path,
            run_id=run_id,
            expected=expected,
            metadata=metadata,
            git_sha=str(metadata.get("git_sha", "")),
            failures=failures,
        )

    run_config_path = paths.get("run_config_path")
    if run_config_path is not None:
        try:
            run_config = yaml.safe_load(run_config_path.read_text(encoding="utf-8")) or {}
            for field, value in {
                "run_id": run_id,
                "resolved_config_hash": expected["resolved_config_hash"],
                "model_config_hash": expected["model_config_hash"],
            }.items():
                if run_config.get(field) != value:
                    failures.append(f"run_config_provenance:{run_id}:{field}")
        except Exception as exception:
            failures.append(f"invalid_run_config:{run_id}:{type(exception).__name__}")

    for field, hash_field in (
        ("split_manifest_path", "split_manifest_hash"),
        ("subset_manifest_path", "subset_manifest_hash"),
    ):
        path = _check_file(metadata.get(field), None, field, run_id, failures)
        if path is not None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get(hash_field) != metadata.get(hash_field):
                    failures.append(f"manifest_artifact:{run_id}:{field}")
                if str(metadata.get(hash_field, ""))[:16] not in path.name:
                    failures.append(f"manifest_artifact_path:{run_id}:{field}")
            except Exception as exception:
                failures.append(f"invalid_manifest_artifact:{run_id}:{field}:{type(exception).__name__}")

    metrics = record["metrics"]
    for split in ("validation", "test", "train"):
        path_field = f"{split}_corruption_bank_path"
        hash_field = f"{split}_corruption_bank_hash"
        path = _check_file(metadata.get(path_field), None, path_field, run_id, failures)
        if path is not None:
            try:
                bank = load_corruption_bank(path)
                if bank.bank_hash != metrics.get(hash_field):
                    failures.append(f"corruption_artifact:{run_id}:{split}")
            except Exception as exception:
                failures.append(f"invalid_corruption_artifact:{run_id}:{split}:{type(exception).__name__}")
    return resume_state_valid


def _validate_provenance(
    expected: Mapping[str, Any],
    record: Mapping[str, Any],
    failures: list[str],
) -> None:
    run_id = str(expected["run_id"])
    job = record["job"]
    metadata = record["metadata"]
    for field in _RESULT_JOB_FIELDS:
        if _csv_text(job.get(field, "")) != _csv_text(expected.get(field, "")):
            failures.append(f"result_job_provenance:{run_id}:{field}")
    for field in (
        "raw_config_hash",
        "resolved_config_hash",
        "study_plan_hash",
        "target_set_hash",
        "environment_lock_hash",
        "model_config_hash",
        "split_manifest_key",
        "subset_manifest_key",
        "resolved_run_spec_hash",
        "config_hash",
    ):
        if metadata.get(field) != expected.get(field):
            failures.append(f"result_metadata_provenance:{run_id}:{field}")
    for field in ("architecture", "architecture_profile"):
        if metadata.get(field) != expected.get(field):
            failures.append(f"result_metadata_provenance:{run_id}:{field}")
    git_sha = str(metadata.get("git_sha", ""))
    if not re.fullmatch(r"[0-9a-fA-F]{40}", git_sha) or git_sha.lower() == "0" * 40:
        failures.append(f"result_metadata_provenance:{run_id}:git_sha")
    for field in (
        "manifest_hash",
        "split_manifest_hash",
        "subset_manifest_hash",
        "target_training_subset_hash",
        "paired_target_prefix_hash",
        "target_eval_indices_hash",
        "target_validation_indices_hash",
    ):
        if not metadata.get(field):
            failures.append(f"result_metadata_provenance:{run_id}:{field}")
    environment = metadata.get("environment_report")
    if not isinstance(environment, Mapping):
        failures.append(f"result_metadata_provenance:{run_id}:environment_report")
    else:
        if environment.get("environment_lock_hash") != expected.get("environment_lock_hash"):
            failures.append(f"result_metadata_provenance:{run_id}:environment_report_lock")
        if not bool(environment.get("lock_matches_runtime", False)):
            failures.append(f"result_metadata_provenance:{run_id}:environment_runtime_mismatch")
        if environment.get("environment_runtime_hash") != metadata.get("environment_runtime_hash"):
            failures.append(f"result_metadata_provenance:{run_id}:environment_runtime_hash")
    if not metadata.get("environment_runtime_hash"):
        failures.append(f"result_metadata_provenance:{run_id}:environment_runtime_hash")
    for field in ("wallclock_total_seconds", "peak_gpu_memory_bytes"):
        if not _finite(metadata.get(field)) or float(metadata.get(field, 0)) <= 0:
            failures.append(f"runtime_metadata:{run_id}:{field}")


def _validate_metrics_and_counters(
    expected: Mapping[str, Any],
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    failures: list[str],
) -> None:
    run_id = str(expected["run_id"])
    training = record["training"]
    metrics = record["metrics"]
    evaluation = config.get("evaluation", {})
    training_config = config.get("training", {})
    steps = int(expected["training_steps"])
    if int(training.get("optimizer_steps", -1)) != steps:
        failures.append(f"exposure_counters:{run_id}:optimizer_steps")
    target_seen = int(training.get("target_examples_seen", -1))
    auxiliary_seen = int(training.get("auxiliary_examples_seen", -1))
    total_seen = int(training.get("total_examples_seen", -1))
    if total_seen != target_seen + auxiliary_seen:
        failures.append(f"exposure_counters:{run_id}:total")
    is_baseline = str(expected["model_type"]).startswith(("conditional_target_only", "unconditional"))
    protocol = str(expected["training_protocol"])
    if protocol == "target_exposure_matched":
        expected_target = steps * int(training_config.get("target_batch_size", training_config.get("batch_size", 1)))
        expected_auxiliary = 0 if is_baseline else steps * int(
            training_config.get("auxiliary_batch_size", training_config.get("batch_size", 1))
        )
        if target_seen != expected_target or auxiliary_seen != expected_auxiliary:
            failures.append(f"exposure_counters:{run_id}:protocol")
    else:
        expected_total = steps * int(training_config.get("batch_size", 1))
        if total_seen != expected_total or target_seen <= 0 or (is_baseline and auxiliary_seen != 0):
            failures.append(f"exposure_counters:{run_id}:protocol")
        if not is_baseline and auxiliary_seen <= 0:
            failures.append(f"exposure_counters:{run_id}:auxiliary")

    for field in ("final_objective_train_loss", "final_pooled_train_loss", "final_target_batch_train_loss"):
        if not _finite(training.get(field)):
            failures.append(f"nonfinite:{run_id}:{field}")
    for field in ("wallclock_train_seconds", "images_processed_per_second"):
        if not _finite(training.get(field)) or float(training.get(field, 0)) <= 0:
            failures.append(f"runtime_metadata:{run_id}:{field}")
    if not is_baseline and not _finite(training.get("final_auxiliary_batch_train_loss")):
        failures.append(f"nonfinite:{run_id}:final_auxiliary_batch_train_loss")
    for field in ("validation_epsilon_mse_target", "test_epsilon_mse_target"):
        source = training if field.startswith("validation") else metrics
        if not _finite(source.get(field)):
            failures.append(f"nonfinite:{run_id}:{field}")

    if metrics.get("evaluation_mode") != "strict" or metrics.get("metric_backend") != "torchmetrics_inception_features":
        failures.append(f"metric_backend:{run_id}")
    for field in ("metric_backend_version", "feature_extractor_name", "metric_implementation"):
        if metrics.get(field) in {None, "", "unknown", "disabled"}:
            failures.append(f"metric_backend_provenance:{run_id}:{field}")
    if int(metrics.get("num_generated", -1)) != int(expected["num_generated"]):
        failures.append(f"metric_count:{run_id}:num_generated")
    expected_real = min(
        int(config.get("data_split", {}).get("target_eval_size", 0)),
        int(evaluation.get("real_eval_max", config.get("data_split", {}).get("target_eval_size", 0))),
    )
    if int(metrics.get("num_real_eval", -1)) != expected_real:
        failures.append(f"metric_count:{run_id}:num_real_eval")

    configured_metrics = {
        "compute_fid": (("fid_target",), "fid_target_status"),
        "compute_kid": (("kid_target_mean", "kid_target_std"), "kid_target_status"),
        "compute_prdc": (
            ("precision_target", "recall_target", "density_target", "coverage_target"),
            "prdc_status",
        ),
    }
    for option, (fields, status_field) in configured_metrics.items():
        if bool(evaluation.get(option, False)):
            if metrics.get(status_field) != "ok":
                failures.append(f"metric_status:{run_id}:{status_field}")
            for field in fields:
                if not _finite(metrics.get(field)):
                    failures.append(f"nonfinite:{run_id}:{field}")
    if bool(evaluation.get("compute_fid", False)):
        fid_minimum = int(evaluation.get("fid_reliable_min_real", 1000))
        warning = str(metrics.get("fid_reliability_warning", ""))
        if expected_real < fid_minimum and not warning:
            failures.append(f"metric_provenance:{run_id}:fid_reliability_warning")
    if bool(evaluation.get("compute_classifier_fidelity", False)):
        if metrics.get("classifier_fidelity_status") != "ok":
            failures.append(f"metric_status:{run_id}:classifier_fidelity_status")
        for field in ("classifier_target_top1_acc", "classifier_target_top5_acc"):
            if not _finite(metrics.get(field)):
                failures.append(f"nonfinite:{run_id}:{field}")
        if not is_baseline and not _finite(metrics.get("auxiliary_leakage_rate")):
            failures.append(f"nonfinite:{run_id}:auxiliary_leakage_rate")
        for field in ("classifier_architecture", "classifier_weights", "classifier_preprocessing"):
            if metrics.get(field) in {None, "", "unknown", "disabled"}:
                failures.append(f"metric_backend_provenance:{run_id}:{field}")
    if bool(evaluation.get("compute_feature_similarity", False)) and not is_baseline:
        if not _finite(metrics.get("average_auxiliary_similarity")):
            failures.append(f"nonfinite:{run_id}:average_auxiliary_similarity")
    if bool(evaluation.get("make_nearest_neighbors", False)):
        for name in ("target_train", "target_eval"):
            if metrics.get(f"nearest_neighbor_{name}_status") != "ok":
                failures.append(f"metric_status:{run_id}:nearest_neighbor_{name}")

    split = config.get("data_split", {})
    expected_counts = {
        "num_validation_images": int(split.get("target_val_size", 0)),
        "num_test_images": int(split.get("target_eval_size", 0)),
        "num_train_images": min(
            int(expected["n0"]),
            int(evaluation.get("train_diagnostic_max_images", int(expected["n0"]))),
        ),
        "num_validation_corruptions": int(split.get("target_val_size", 0))
        * int(evaluation.get("validation_corruptions_per_image", 1)),
        "num_test_corruptions": int(split.get("target_eval_size", 0))
        * int(evaluation.get("test_corruptions_per_image", 1)),
        "num_train_corruptions": min(
            int(expected["n0"]),
            int(evaluation.get("train_diagnostic_max_images", int(expected["n0"]))),
        )
        * int(evaluation.get("train_diagnostic_corruptions_per_image", 1)),
    }
    for field, value in expected_counts.items():
        if int(metrics.get(field, -1)) != value:
            failures.append(f"metric_count:{run_id}:{field}")


def _validate_pairing(
    expected_rows: list[dict[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    failures: list[str],
) -> int:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    pair_fields = (
        "experiment",
        "target_synset",
        "n0",
        "holdout_seed",
        "training_subset_seed",
        "model_initialization_seed",
        "training_seed",
        "sampling_seed",
        "evaluation_seed",
        "training_protocol",
        "architecture_profile",
        "model_config_hash",
    )
    for row in expected_rows:
        groups[tuple(row[field] for field in pair_fields)].append(row)

    pair_count = 0
    for rows in groups.values():
        baselines = [row for row in rows if str(row["model_type"]).startswith("conditional_target_only")]
        candidates = [row for row in rows if row not in baselines and not str(row["model_type"]).startswith("unconditional")]
        if len(baselines) != 1 or not candidates:
            failures.append(f"pair_design:{rows[0]['run_id']}")
            continue
        baseline_record = records.get(str(baselines[0]["run_id"]))
        if baseline_record is None:
            continue
        baseline_metadata = baseline_record["metadata"]
        for candidate in candidates:
            candidate_record = records.get(str(candidate["run_id"]))
            if candidate_record is None:
                continue
            pair_count += 1
            candidate_metadata = candidate_record["metadata"]
            for field in (
                "split_manifest_hash",
                "subset_manifest_hash",
                "target_training_subset_hash",
                "paired_target_prefix_hash",
                "target_eval_indices_hash",
                "target_validation_indices_hash",
            ):
                if candidate_metadata.get(field) != baseline_metadata.get(field):
                    failures.append(f"pair_identity:{candidate['run_id']}:{field}")
            for field in ("validation_corruption_bank_hash", "test_corruption_bank_hash"):
                if candidate_record["metrics"].get(field) != baseline_record["metrics"].get(field):
                    failures.append(f"pair_identity:{candidate['run_id']}:{field}")
    return pair_count


def validate_pilot(config_path: str | Path, jobs_csv: str | Path, results_root: str | Path) -> dict[str, Any]:
    resolved = load_resolved_config(config_path)
    if str(resolved.resolved.get("study_stage")) != "pilot":
        raise ValueError("Release-pilot validation requires a pilot-stage config")
    if bool(resolved.resolved.get("use_fake_data", False)):
        raise ValueError("Release-pilot validation refuses synthetic-data configurations")
    if str(resolved.resolved.get("evaluation", {}).get("mode")) != "strict":
        raise ValueError("Release-pilot validation requires strict evaluation mode")
    expected_rows = rebuild_expected_jobs(config_path, resolved)
    if not expected_rows:
        raise ValueError("Release-pilot config produces no explicitly enabled jobs")
    jobs, failures = _read_and_compare_grid(jobs_csv, expected_rows)
    root = Path(results_root)
    configured_root = Path(str(resolve_env_path(resolved.resolved.get("output_root"), "image_transfer_results")))
    if root.resolve() != configured_root.resolve():
        failures.append("results_root_mismatch")

    expected_by_id = {str(row["run_id"]): row for row in expected_rows}
    records: dict[str, Mapping[str, Any]] = {}
    result_hashes: dict[str, str] = {}
    result_dir = root / "run_results"
    existing_ids = {path.stem for path in result_dir.glob("*.json")} if result_dir.exists() else set()
    for run_id in sorted(existing_ids - set(expected_by_id)):
        failures.append(f"unexpected_result:{run_id}")
    failure_dir = root / "failures"
    for path in sorted(failure_dir.glob("*.json")) if failure_dir.exists() else []:
        failures.append(f"failure_record_present:{path.stem}")

    resume_state_validated = 0

    for run_id, expected in expected_by_id.items():
        result_path = result_dir / f"{run_id}.json"
        if not result_path.exists():
            failures.append(f"missing_result:{run_id}")
            continue
        try:
            record = load_valid_result(result_path, expected_run_id=run_id)
        except Exception as exception:
            failures.append(f"invalid_result:{run_id}:{type(exception).__name__}:{exception}")
            continue
        result_hashes[run_id] = _hash(result_path)
        if record.get("status") != "completed":
            failures.append(f"not_completed:{run_id}")
            continue
        records[run_id] = record
        _validate_provenance(expected, record, failures)
        _validate_metrics_and_counters(expected, record, resolved.resolved, failures)
        if _validate_artifacts(expected, record, resolved.resolved, failures):
            resume_state_validated += 1

    pair_count = _validate_pairing(expected_rows, records, failures)
    git_shas = {str(record["metadata"].get("git_sha", "")) for record in records.values()}
    if len(git_shas) != 1:
        failures.append("result_git_sha_mismatch")
        validated_git_sha = ""
    else:
        validated_git_sha = next(iter(git_shas))
    failures = list(dict.fromkeys(failures))
    status = "passed" if not failures and bool(expected_rows) else "failed"
    return {
        "schema_version": "2.0",
        "status": status,
        "meaning": "engineering health only; this does not validate a scientific hypothesis",
        "git_sha": validated_git_sha,
        "model_config_hash": resolved.model_hash,
        "target_set_hash": resolved.target_set_hash,
        "environment_lock_hash": resolved.environment_lock_hash,
        "study_plan_hash": resolved.study_plan_hash,
        "pilot_config_hash": resolved.resolved_hash,
        "expected_job_grid_hash": canonical_sha256(expected_rows),
        "jobs_csv_hash": _hash(Path(jobs_csv)),
        "validated_result_hashes": result_hashes,
        "expected_jobs": len(expected_rows),
        "supplied_jobs": len(jobs),
        "validated_jobs": len(records),
        "validated_pairs": pair_count,
        "resume_state_validated_jobs": resume_state_validated,
        "failures": failures,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--jobs-csv", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--status-out", default="readiness/pilot_status.json")
    args = parser.parse_args()
    report = validate_pilot(args.config, args.jobs_csv, args.results_root)
    atomic_write_json(report, args.status_out)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
