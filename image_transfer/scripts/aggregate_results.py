from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from image_transfer.utils.io import (
    atomic_write_json,
    canonical_json_hash,
    load_json,
    normalize_result_record,
    validate_result_record,
)
from image_transfer.evaluation.classifier_fidelity import imagenet_synset_to_index

PRIMARY_BASELINES = {
    "A": "conditional_target_only_n0",
    "B": "conditional_target_only_equal_total",
    "C": "conditional_target_only_n0",
}
LEGACY_BASELINES = {
    "A": "unconditional_n0",
    "B": "unconditional_equal_total",
    "C": "unconditional_n0",
}
BASELINE_MODEL_TYPES = set(PRIMARY_BASELINES.values()) | set(LEGACY_BASELINES.values())

LOWER_IS_BETTER = {
    "test_epsilon_mse_target",
    "test_epsilon_mse_low_noise",
    "test_epsilon_mse_mid_noise",
    "test_epsilon_mse_high_noise",
    "validation_epsilon_mse_target",
    "validation_epsilon_mse_low_noise",
    "validation_epsilon_mse_mid_noise",
    "validation_epsilon_mse_high_noise",
    "fid_target",
    "kid_target_mean",
    "auxiliary_leakage_rate",
    "debug_pooled_pixel_distance",
}
HIGHER_IS_BETTER = {
    "classifier_target_top1_acc",
    "classifier_target_top5_acc",
    "precision_target",
    "recall_target",
    "density_target",
    "coverage_target",
    "inception_score_mean",
}
METRIC_DIRECTIONS = {**{name: "lower" for name in LOWER_IS_BETTER}, **{name: "higher" for name in HIGHER_IS_BETTER}}

PAIR_KEY_COLUMNS = [
    "experiment",
    "target_synset",
    "n0",
    "m_per_aux",
    "K_aux",
    "baseline_target_count",
    "architecture",
    "architecture_profile",
    "model_config_hash",
    "holdout_seed",
    "training_subset_seed",
    "split_manifest_hash",
    "subset_manifest_hash",
    "paired_target_prefix_hash",
    "model_initialization_seed",
    "training_seed",
    "training_protocol",
    "sampling_seed",
    "evaluation_seed",
    "sampler",
    "sampling_steps",
    "resolved_run_spec_hash",
    "resolved_config_hash",
    "study_plan_hash",
    "target_set_hash",
    "environment_lock_hash",
    "environment_runtime_hash",
]
BASELINE_PAIR_KEY_COLUMNS = [
    "experiment",
    "target_synset",
    "n0",
    "baseline_target_count",
    "architecture",
    "architecture_profile",
    "model_config_hash",
    "holdout_seed",
    "training_subset_seed",
    "split_manifest_hash",
    "subset_manifest_hash",
    "paired_target_prefix_hash",
    "model_initialization_seed",
    "training_seed",
    "training_protocol",
    "sampling_seed",
    "evaluation_seed",
    "sampler",
    "sampling_steps",
    "resolved_config_hash",
    "study_plan_hash",
    "target_set_hash",
    "environment_lock_hash",
    "environment_runtime_hash",
]
GROUP_COLUMNS = [
    "experiment",
    "target_synset",
    "model_type",
    "aux_set",
    "n0",
    "m_per_aux",
    "K_aux",
    "training_protocol",
    "architecture",
    "architecture_profile",
    "model_config_hash",
    "target_set_hash",
    "environment_lock_hash",
    "environment_runtime_hash",
]

HIERARCHY_ID_COLUMNS = [
    "target_synset",
    "split_manifest_hash",
    "subset_manifest_hash",
    "holdout_seed",
    "training_subset_seed",
]

OPTIMIZATION_ID_COLUMNS = [
    "model_initialization_seed",
    "training_seed",
]


def _atomic_to_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temp_name = handle.name
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
    return destination


def _discover(root: Path, directory: str) -> list[Path]:
    direct = sorted((root / directory).glob("*.json"))
    nested = sorted(root.glob(f"*/{directory}/*.json"))
    return sorted(set(direct + nested))


def _load_run_records(root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    records: dict[str, tuple[float, dict[str, Any], Path]] = {}
    invalid: list[dict[str, Any]] = []
    duplicate_counts: dict[str, int] = {}
    for path in _discover(root, "run_results"):
        try:
            record = load_json(path)
            validate_result_record(record)
            flat = normalize_result_record(record)
        except Exception as exception:
            invalid.append(
                {
                    "run_id": path.stem,
                    "status": "invalid_result",
                    "exception_type": type(exception).__name__,
                    "message": str(exception),
                    "source_path": str(path),
                }
            )
            continue
        run_id = str(flat["run_id"])
        modified = path.stat().st_mtime
        if run_id in records:
            duplicate_counts[run_id] = duplicate_counts.get(run_id, 1) + 1
        if run_id not in records or modified >= records[run_id][0]:
            flat["source_path"] = str(path)
            records[run_id] = (modified, flat, path)
    frame = pd.DataFrame([value[1] for value in records.values()])
    return _normalize_columns(frame), invalid, duplicate_counts


def _load_failures(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in _discover(root, "failures"):
        try:
            record = load_json(path)
            flat = dict(record.get("job") or {})
            flat.update({key: value for key, value in record.items() if key not in {"job", "config"}})
            flat["source_path"] = str(path)
            rows.append(flat)
        except Exception as exception:
            rows.append(
                {
                    "run_id": path.stem,
                    "status": "invalid_failure_record",
                    "exception_type": type(exception).__name__,
                    "message": str(exception),
                    "source_path": str(path),
                }
            )
    return _normalize_columns(pd.DataFrame(rows))


def _load_expected(paths: Iterable[str | Path] | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_path in paths or []:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".json":
            value = load_json(path)
            candidates = value.get("jobs", []) if isinstance(value, dict) else []
            rows.extend(dict(row) for row in candidates)
        else:
            with open(path, newline="", encoding="utf-8") as handle:
                rows.extend(dict(row) for row in csv.DictReader(handle))
    expected = pd.DataFrame(rows)
    if not expected.empty and "run_id" in expected:
        expected = expected.drop_duplicates("run_id", keep="last")
    return _normalize_columns(expected)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    if "target_synset" not in result and "target" in result:
        result["target_synset"] = result["target"]
    if "seed" not in result:
        result["seed"] = 0
    for name in ("data_split_seed", "model_initialization_seed", "training_seed", "sampling_seed", "evaluation_seed"):
        if name not in result:
            result[name] = result["seed"]
        else:
            result[name] = result[name].where(result[name].notna(), result["seed"])
    if "holdout_seed" not in result:
        result["holdout_seed"] = result["data_split_seed"]
    else:
        result["holdout_seed"] = result["holdout_seed"].where(
            result["holdout_seed"].notna(), result["data_split_seed"]
        )
    if "training_subset_seed" not in result:
        result["training_subset_seed"] = result["data_split_seed"]
    else:
        result["training_subset_seed"] = result["training_subset_seed"].where(
            result["training_subset_seed"].notna(), result["data_split_seed"]
        )
    if "training_protocol" not in result:
        result["training_protocol"] = "natural_compute_matched"
    result["training_protocol"] = result["training_protocol"].replace("", "natural_compute_matched").fillna("natural_compute_matched")
    defaults: dict[str, Any] = {
        "experiment": "",
        "target_synset": "",
        "model_type": "",
        "aux_set": "none",
        "n0": 0,
        "m_per_aux": 0,
        "K_aux": 0,
        "run_id": "",
        "sampler": "ddpm",
        "config_hash": "",
        "effective_run_spec_hash": "",
        "sampling_steps": 0,
        "architecture": "legacy_simple_unet",
        "architecture_profile": "legacy",
        "manifest_hash": "",
        "split_manifest_hash": "",
        "subset_manifest_hash": "",
        "target_training_subset_hash": "",
        "paired_target_prefix_hash": "",
        "model_config_hash": "",
        "resolved_run_spec_hash": "",
        "resolved_config_hash": "",
        "study_plan_hash": "",
        "target_set_hash": "",
        "environment_lock_hash": "",
        "environment_runtime_hash": "",
    }
    for key, value in defaults.items():
        if key not in result:
            result[key] = value
        else:
            result[key] = result[key].fillna(value)
    for name in (
        "n0", "m_per_aux", "K_aux", "seed", "data_split_seed", "holdout_seed",
        "training_subset_seed", "model_initialization_seed", "training_seed",
        "sampling_seed", "evaluation_seed", "sampling_steps",
    ):
        result[name] = pd.to_numeric(result[name], errors="coerce").fillna(0).astype(int)
    # Legacy records are kept analyzable, but all v2 identities prefer their
    # explicit hashes.  The fallbacks never make two records with different
    # known identities equal.
    result["split_manifest_hash"] = result["split_manifest_hash"].replace("", np.nan).fillna(
        result["manifest_hash"]
    )
    result["subset_manifest_hash"] = result["subset_manifest_hash"].replace("", np.nan).fillna(
        result["manifest_hash"]
    )
    result["paired_target_prefix_hash"] = result["paired_target_prefix_hash"].replace("", np.nan).fillna(
        result["target_training_subset_hash"]
    )
    result["model_config_hash"] = result["model_config_hash"].replace("", np.nan).fillna(
        result["architecture"].astype(str) + ":" + result["architecture_profile"].astype(str)
    )
    if "effective_run_spec_hash" not in result:
        result["effective_run_spec_hash"] = ""
    result["resolved_run_spec_hash"] = result["resolved_run_spec_hash"].replace("", np.nan).fillna(
        result["effective_run_spec_hash"]
    )
    if "config_hash" not in result:
        result["config_hash"] = ""
    result["resolved_config_hash"] = result["resolved_config_hash"].replace("", np.nan).fillna(
        result["config_hash"]
    )
    result["target_set_hash"] = result["target_set_hash"].replace("", np.nan).fillna(
        result["config_hash"]
    )
    result["study_plan_hash"] = result["study_plan_hash"].replace("", "legacy-untracked")
    result["environment_lock_hash"] = result["environment_lock_hash"].replace("", "legacy-untracked")
    result["environment_runtime_hash"] = result["environment_runtime_hash"].replace("", "legacy-untracked")
    result["training_subset_replicate_id"] = result["training_subset_seed"].astype(str)
    result["optimization_repeat_id"] = (
        result["model_initialization_seed"].astype(str) + ":" + result["training_seed"].astype(str)
    )
    if "total_auxiliary_budget" not in result:
        result["total_auxiliary_budget"] = result["m_per_aux"] * result["K_aux"]
    else:
        result["total_auxiliary_budget"] = pd.to_numeric(result["total_auxiliary_budget"], errors="coerce").fillna(
            result["m_per_aux"] * result["K_aux"]
        )
    inferred_baseline_count = np.where(
        result["experiment"].astype(str) == "B",
        result["n0"] + result["total_auxiliary_budget"],
        result["n0"],
    )
    if "baseline_target_count" not in result:
        result["baseline_target_count"] = inferred_baseline_count
    else:
        result["baseline_target_count"] = pd.to_numeric(
            result["baseline_target_count"], errors="coerce"
        ).fillna(pd.Series(inferred_baseline_count, index=result.index))
    result["baseline_target_count"] = result["baseline_target_count"].astype(int)
    if "auxiliary_ratio" not in result:
        result["auxiliary_ratio"] = np.where(result["n0"] > 0, result["m_per_aux"] / result["n0"], np.nan)
    else:
        result["auxiliary_ratio"] = pd.to_numeric(result["auxiliary_ratio"], errors="coerce")
    return result


def _key(row: pd.Series | dict[str, Any], columns: list[str] = PAIR_KEY_COLUMNS) -> tuple[Any, ...]:
    values = []
    for column in columns:
        value = row.get(column)
        if pd.isna(value):
            value = None
        if isinstance(value, np.generic):
            value = value.item()
        values.append(value)
    return tuple(values)


def _baseline_pair_key(row: pd.Series | dict[str, Any]) -> tuple[Any, ...]:
    """Identify one architecture-matched baseline training/evaluation run.

    Auxiliary composition, draw, and the ``m``/``K`` factorization are not part
    of this key. Experiment B instead keys the target-only control by its actual
    target exposure, ``n0 + m*K``. This lets one predeclared control be reused
    without silently treating duplicate baseline training as replication.
    """

    return _key(row, BASELINE_PAIR_KEY_COLUMNS)


def improvement_positive(model_value: float, baseline_value: float, metric: str) -> float:
    if metric not in METRIC_DIRECTIONS:
        raise KeyError(f"No direction is registered for metric {metric!r}")
    if METRIC_DIRECTIONS[metric] == "lower":
        return float(baseline_value - model_value)
    return float(model_value - baseline_value)


def _sample_count_fields(frame: pd.DataFrame) -> dict[str, Any]:
    """Keep reference/generated counts visible in every metric summary."""

    fields: dict[str, Any] = {}
    for sources, output in (
        (("num_real_eval",), "num_real_eval"),
        (("num_generated", "num_generated_eval"), "num_generated_eval"),
    ):
        source = next((name for name in sources if name in frame), sources[0])
        values = pd.to_numeric(frame.get(source, pd.Series(dtype=float)), errors="coerce").dropna()
        fields[output] = int(values.iloc[0]) if not values.empty and values.nunique() == 1 else np.nan
        fields[f"{output}_min"] = int(values.min()) if not values.empty else np.nan
        fields[f"{output}_max"] = int(values.max()) if not values.empty else np.nan
    return fields


def _paired_baseline_leakage(baseline: pd.Series, candidate: pd.Series) -> float:
    """Re-evaluate one shared baseline histogram for a candidate aux set."""

    raw_auxiliary = candidate.get("aux_synsets", candidate.get("aux_composition", "[]"))
    try:
        auxiliary = json.loads(raw_auxiliary) if isinstance(raw_auxiliary, str) else list(raw_auxiliary)
    except (TypeError, ValueError, json.JSONDecodeError):
        return float("nan")
    indices = [imagenet_synset_to_index(str(synset)) for synset in auxiliary]
    if not indices or any(index is None for index in indices):
        return float("nan")
    raw_histogram = baseline.get("top1_prediction_histogram_json", "{}")
    try:
        histogram = json.loads(raw_histogram) if isinstance(raw_histogram, str) else dict(raw_histogram)
        counts = {int(key): int(value) for key, value in histogram.items()}
    except (TypeError, ValueError, json.JSONDecodeError):
        return float("nan")
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    return float(sum(counts.get(int(index), 0) for index in indices if index is not None) / total)


def compute_paired_gaps(
    results: pd.DataFrame,
    *,
    expected_jobs: pd.DataFrame | None = None,
    failed_run_ids: set[str] | None = None,
) -> pd.DataFrame:
    results = _normalize_columns(results)
    expected_jobs = _normalize_columns(expected_jobs) if expected_jobs is not None else pd.DataFrame()
    failed_run_ids = failed_run_ids or set()
    candidates = expected_jobs if not expected_jobs.empty else results
    if candidates.empty:
        return pd.DataFrame()
    actual_by_id = {str(row["run_id"]): row for _, row in results.iterrows()}
    baseline_lookup: dict[tuple[str, tuple[Any, ...]], list[pd.Series]] = defaultdict(list)
    for _, row in results.iterrows():
        model_type = str(row["model_type"])
        if model_type in BASELINE_MODEL_TYPES:
            baseline_lookup[(model_type, _baseline_pair_key(row))].append(row)
    expected_baseline_lookup: dict[tuple[str, tuple[Any, ...]], list[pd.Series]] = defaultdict(list)
    for _, row in expected_jobs.iterrows():
        model_type = str(row["model_type"])
        if model_type in BASELINE_MODEL_TYPES:
            expected_baseline_lookup[(model_type, _baseline_pair_key(row))].append(row)

    available_metrics = [metric for metric in METRIC_DIRECTIONS if metric in results.columns]
    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        model_type = str(candidate["model_type"])
        if model_type in BASELINE_MODEL_TYPES or not model_type:
            continue
        run_id = str(candidate.get("run_id", ""))
        model = actual_by_id.get(run_id)
        if model is not None:
            model_status = "completed" if str(model.get("status", "completed")) == "completed" else "skipped_model"
        else:
            model_status = "failed_model" if run_id in failed_run_ids else "missing_model"
        # Expected grids do not necessarily contain content hashes that only
        # become available after dataset preflight.  Completed runs do, so use
        # the actual model record for strict pairing whenever it exists.
        pair_identity = model if model is not None else candidate
        experiment = str(candidate["experiment"])
        for baseline_kind, mapping in (("primary", PRIMARY_BASELINES), ("legacy", LEGACY_BASELINES)):
            baseline_type = mapping.get(experiment)
            if baseline_type is None:
                continue
            # A deliberately disabled legacy control is not a missing pair.
            # Only summarize a baseline family that was present in the
            # declared grid or in completed results.
            baseline_declared = any(
                key[0] == baseline_type
                for key in (*baseline_lookup.keys(), *expected_baseline_lookup.keys())
            )
            if not baseline_declared:
                continue
            matching = baseline_lookup.get((baseline_type, _baseline_pair_key(pair_identity)), [])
            ambiguous_baseline = len(matching) > 1
            baseline = matching[0] if len(matching) == 1 else None
            expected_baselines = expected_baseline_lookup.get(
                (baseline_type, _baseline_pair_key(candidate)), []
            )
            expected_baseline_ids = {str(row.get("run_id", "")) for row in expected_baselines}
            if ambiguous_baseline:
                baseline_status = "ambiguous_baseline"
            elif baseline is not None:
                baseline_status = "completed" if str(baseline.get("status", "completed")) == "completed" else "skipped_baseline"
            elif expected_baseline_ids & failed_run_ids:
                baseline_status = "failed_baseline"
            else:
                baseline_status = "missing_baseline"

            for metric in available_metrics:
                pair_status = model_status if model_status != "completed" else baseline_status
                model_value = pd.to_numeric(pd.Series([model.get(metric) if model is not None else np.nan]), errors="coerce").iloc[0]
                baseline_value = pd.to_numeric(pd.Series([baseline.get(metric) if baseline is not None else np.nan]), errors="coerce").iloc[0]
                if metric == "auxiliary_leakage_rate" and baseline is not None and pd.isna(baseline_value):
                    baseline_value = _paired_baseline_leakage(baseline, model if model is not None else candidate)
                gap = float("nan")
                if pair_status == "completed":
                    if pd.isna(model_value) or pd.isna(baseline_value):
                        pair_status = "missing_metric"
                    else:
                        gap = improvement_positive(float(model_value), float(baseline_value), metric)
                row = {column: pair_identity.get(column) for column in PAIR_KEY_COLUMNS}
                for column in (
                    "model_type",
                    "aux_set",
                    "aux_composition",
                    "aux_synsets",
                    "average_auxiliary_similarity",
                    "aux_draw_id",
                    "auxiliary_ratio",
                    "total_auxiliary_budget",
                    "training_subset_replicate_id",
                    "optimization_repeat_id",
                    "target_training_subset_hash",
                    "auxiliary_training_subset_hashes",
                ):
                    row[column] = (model if model is not None else candidate).get(column)
                row.update(
                    {
                        "run_id": run_id,
                        "baseline_run_id": str(baseline.get("run_id", "")) if baseline is not None else "",
                        "baseline_kind": baseline_kind,
                        "baseline_model_type": baseline_type,
                        "metric": metric,
                        "metric_direction": METRIC_DIRECTIONS[metric],
                        "model_metric": model_value,
                        "baseline_metric": baseline_value,
                        "improvement_positive": gap,
                        "pair_status": pair_status,
                        "ambiguous_baseline_count": len(matching) if ambiguous_baseline else 0,
                        "num_real_eval": (
                            model.get("num_real_eval") if model is not None else candidate.get("num_real_eval")
                        ),
                        "num_generated": (
                            model.get("num_generated") if model is not None else candidate.get("num_generated")
                        ),
                        "baseline_num_real_eval": baseline.get("num_real_eval") if baseline is not None else np.nan,
                        "baseline_num_generated": baseline.get("num_generated") if baseline is not None else np.nan,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


_T_975 = [
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
]


def t95_confidence_interval(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray([float(value) for value in values if pd.notna(value)], dtype=float)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return {"mean": np.nan, "standard_deviation": np.nan, "standard_error": np.nan, "ci95_lower": np.nan, "ci95_upper": np.nan, "n": 0}
    mean = float(array.mean())
    if count == 1:
        return {"mean": mean, "standard_deviation": np.nan, "standard_error": np.nan, "ci95_lower": np.nan, "ci95_upper": np.nan, "n": 1}
    standard_deviation = float(array.std(ddof=1))
    standard_error = standard_deviation / math.sqrt(count)
    degrees = count - 1
    if degrees <= len(_T_975):
        critical = _T_975[degrees - 1]
    else:
        # Cornish-Fisher expansion of the Student-t 97.5% quantile.  This keeps
        # the interval a t interval without requiring SciPy on login nodes.
        z = 1.959963984540054
        critical = (
            z
            + (z**3 + z) / (4 * degrees)
            + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * degrees**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * degrees**3)
        )
    half_width = critical * standard_error
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
        "n": count,
    }


def summarize_paired_gaps(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    group_columns = [column for column in GROUP_COLUMNS + ["baseline_kind", "baseline_model_type", "metric"] if column in pairs]
    rows: list[dict[str, Any]] = []
    for keys, group in pairs.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        complete = group[group["pair_status"] == "completed"]
        cluster_columns = [
            column
            for column in (
                "target_synset",
                "split_manifest_hash",
                "subset_manifest_hash",
                "model_initialization_seed",
                "training_seed",
                "sampling_seed",
                "evaluation_seed",
            )
            if column in complete
        ]
        if complete.empty:
            cluster_values: list[float] = []
        elif cluster_columns:
            cluster_values = (
                complete.groupby(cluster_columns, dropna=False)["improvement_positive"]
                .mean()
                .tolist()
            )
        else:
            cluster_values = complete["improvement_positive"].tolist()
        row.update(t95_confidence_interval(cluster_values))
        row.update(
            {
                "summary_type": "paired_transfer_gap",
                "number_paired_runs": int(len(cluster_values)),
                "number_completed_draw_pairs": int(len(complete)),
                "number_missing_failed_pairs": int(len(group) - len(complete)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _condition_columns(frame: pd.DataFrame) -> list[str]:
    """Scientific condition columns that must never be pooled implicitly."""

    return [
        column
        for column in (
            "experiment",
            "model_type",
            "aux_set",
            "n0",
            "m_per_aux",
            "K_aux",
            "auxiliary_ratio",
            "total_auxiliary_budget",
            "average_auxiliary_similarity",
            "training_protocol",
            "architecture",
            "architecture_profile",
            "model_config_hash",
            "sampler",
            "sampling_steps",
            "resolved_config_hash",
            "study_plan_hash",
            "target_set_hash",
            "environment_lock_hash",
            "environment_runtime_hash",
            "baseline_kind",
            "baseline_model_type",
            "metric",
        )
        if column in frame
    ]


def _unique_rows(frame: pd.DataFrame, columns: list[str]) -> int:
    present = [column for column in columns if column in frame]
    return int(len(frame[present].drop_duplicates())) if present and not frame.empty else int(bool(len(frame)))


def summarize_by_training_subset(pairs: pd.DataFrame) -> pd.DataFrame:
    """Average draws within optimization repeats, then repeats within subsets."""

    if pairs.empty:
        return pd.DataFrame()
    condition = _condition_columns(pairs)
    subset_ids = [column for column in HIERARCHY_ID_COLUMNS if column in pairs]
    group_columns = list(dict.fromkeys([*condition, *subset_ids]))
    rows: list[dict[str, Any]] = []
    for keys, group in pairs.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        complete = group[group["pair_status"] == "completed"]
        optimization_ids = [column for column in OPTIMIZATION_ID_COLUMNS if column in complete]
        if complete.empty:
            optimization_values: list[float] = []
        elif optimization_ids:
            # Auxiliary draws and repeated evaluation streams are nested inside
            # one trained-model repeat; they cannot increase the target-data n.
            optimization_values = (
                complete.groupby(optimization_ids, dropna=False)["improvement_positive"].mean().tolist()
            )
        else:
            optimization_values = [float(complete["improvement_positive"].mean())]
        row.update(t95_confidence_interval(optimization_values))
        row.update(_sample_count_fields(complete))
        row.update(
            {
                "summary_type": "training_subset",
                "improvement_positive": float(np.mean(optimization_values)) if optimization_values else np.nan,
                "number_targets": 1,
                "number_independent_training_subsets": 1,
                "number_optimization_repeats": _unique_rows(group, OPTIMIZATION_ID_COLUMNS),
                "number_completed_optimization_repeats": len(optimization_values),
                "number_auxiliary_draws": _unique_rows(group, ["aux_draw_id", "aux_composition"]),
                "number_completed_draw_pairs": int(len(complete)),
                "number_missing_failed_pairs": int(len(group) - len(complete)),
                "number_unique_baseline_runs": int(complete.get("baseline_run_id", pd.Series(dtype=str)).nunique()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_target(subset_summaries: pd.DataFrame) -> pd.DataFrame:
    """Compute target-conditional intervals across independent subsets."""

    if subset_summaries.empty:
        return pd.DataFrame()
    condition = _condition_columns(subset_summaries)
    group_columns = [*condition, "target_synset"]
    rows: list[dict[str, Any]] = []
    for keys, group in subset_summaries.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        values = pd.to_numeric(group["improvement_positive"], errors="coerce")
        stats = t95_confidence_interval(values.tolist())
        row.update(stats)
        row.update(_sample_count_fields(group))
        row.update(
            {
                "summary_type": "target_conditional",
                "improvement_positive": stats["mean"],
                "conditional_on_target_ci95_lower": stats["ci95_lower"],
                "conditional_on_target_ci95_upper": stats["ci95_upper"],
                "across_subset_ci95_lower": stats["ci95_lower"],
                "across_subset_ci95_upper": stats["ci95_upper"],
                "number_targets": 1,
                "number_holdout_splits": int(group.get("split_manifest_hash", pd.Series(dtype=str)).nunique()),
                "number_independent_training_subsets": int(values.notna().sum()),
                "number_optimization_repeats": int(group["number_optimization_repeats"].sum()),
                "number_auxiliary_draws": int(group["number_auxiliary_draws"].sum()),
                "number_completed_draw_pairs": int(group["number_completed_draw_pairs"].sum()),
                "number_missing_failed_pairs": int(group["number_missing_failed_pairs"].sum()),
            }
        )
        row.update(_sample_count_fields(group))
        rows.append(row)
    return pd.DataFrame(rows)


def hierarchical_target_bootstrap(
    subset_summaries: pd.DataFrame,
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Target-cluster bootstrap with subset resampling inside each target."""

    if subset_summaries.empty:
        return pd.DataFrame()
    condition = _condition_columns(subset_summaries)
    rows: list[dict[str, Any]] = []
    for keys, group in subset_summaries.groupby(condition, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(condition, keys))
        target_groups = {
            str(target): part[pd.to_numeric(part["improvement_positive"], errors="coerce").notna()]
            for target, part in group.groupby("target_synset", dropna=False)
        }
        target_groups = {target: part for target, part in target_groups.items() if not part.empty}
        target_means = [float(part["improvement_positive"].mean()) for part in target_groups.values()]
        row.update(
            {
                "summary_type": "hierarchical_across_target",
                "mean": float(np.mean(target_means)) if target_means else np.nan,
                "improvement_positive": float(np.mean(target_means)) if target_means else np.nan,
                "number_targets": len(target_groups),
                "number_independent_training_subsets": int(
                    pd.to_numeric(group["improvement_positive"], errors="coerce").notna().sum()
                ),
                "number_optimization_repeats": int(group["number_optimization_repeats"].sum()),
                "number_auxiliary_draws": int(group["number_auxiliary_draws"].sum()),
                "number_completed_draw_pairs": int(group["number_completed_draw_pairs"].sum()),
                "number_missing_failed_pairs": int(group["number_missing_failed_pairs"].sum()),
            }
        )
        if len(target_groups) < 2:
            row.update(
                {
                    "hierarchical_status": "unavailable_single_target",
                    "across_target_ci95_lower": np.nan,
                    "across_target_ci95_upper": np.nan,
                    "ci95_lower": np.nan,
                    "ci95_upper": np.nan,
                }
            )
            rows.append(row)
            continue
        digest = hashlib.sha256(repr(keys).encode("utf-8")).digest()
        rng = np.random.default_rng(int(bootstrap_seed) ^ int.from_bytes(digest[:8], "big"))
        targets = list(target_groups)
        bootstrap_values: list[float] = []
        for _ in range(int(bootstrap_samples)):
            sampled_target_means: list[float] = []
            for target_index in rng.integers(0, len(targets), len(targets)):
                values = pd.to_numeric(
                    target_groups[targets[int(target_index)]]["improvement_positive"], errors="coerce"
                ).to_numpy(dtype=float)
                sampled = values[rng.integers(0, len(values), len(values))]
                sampled_target_means.append(float(sampled.mean()))
            bootstrap_values.append(float(np.mean(sampled_target_means)))
        lower = float(np.quantile(bootstrap_values, 0.025))
        upper = float(np.quantile(bootstrap_values, 0.975))
        row.update(
            {
                "hierarchical_status": "available",
                "across_target_ci95_lower": lower,
                "across_target_ci95_upper": upper,
                "ci95_lower": lower,
                "ci95_upper": upper,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_raw_metrics(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    group_columns = [column for column in GROUP_COLUMNS if column in results]
    rows: list[dict[str, Any]] = []
    for metric in METRIC_DIRECTIONS:
        if metric not in results:
            continue
        for keys, group in results.groupby(group_columns, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            row = dict(zip(group_columns, keys))
            numeric = pd.to_numeric(group[metric], errors="coerce")
            row.update(t95_confidence_interval(numeric.tolist()))
            row.update(_sample_count_fields(group))
            row.update(
                {
                    "summary_type": "raw_metric",
                    "metric": metric,
                    "number_paired_runs": int(numeric.notna().sum()),
                    "number_missing_failed_pairs": int(numeric.isna().sum()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.rank(method="average").corr(y.rank(method="average")))


def similarity_correlations(pairs: pd.DataFrame, bootstrap_samples: int = 1000) -> pd.DataFrame:
    if pairs.empty or "average_auxiliary_similarity" not in pairs:
        return pd.DataFrame()
    valid = pairs[(pairs["pair_status"] == "completed") & pairs["average_auxiliary_similarity"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    # Correlate across the predeclared close-to-far compositions while holding
    # all quantity axes fixed. Grouping by model_type would condition on the
    # composition itself and often leave constant-similarity/undefined groups.
    group_columns = [
        column
        for column in (
            "experiment",
            "n0",
            "m_per_aux",
            "K_aux",
            "training_protocol",
            "baseline_kind",
            "metric",
        )
        if column in valid
    ]
    rows = []
    rng = np.random.default_rng(0)
    for keys, group in valid.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        x = pd.to_numeric(group["average_auxiliary_similarity"], errors="coerce")
        y = pd.to_numeric(group["improvement_positive"], errors="coerce")
        keep = x.notna() & y.notna()
        group, x, y = group.loc[keep], x.loc[keep], y.loc[keep]
        row = dict(zip(group_columns, keys))
        row["spearman_correlation"] = _spearman(x, y)
        row["n"] = int(len(group))
        cluster_columns = [
            column
            for column in (
                "target_synset",
                "split_manifest_hash",
                "subset_manifest_hash",
                "model_initialization_seed",
                "training_seed",
            )
            if column in group
        ]
        bootstrap: list[float] = []
        if len(group) >= 3 and cluster_columns:
            grouped = [part for _, part in group.groupby(cluster_columns, dropna=False)]
            for _ in range(bootstrap_samples):
                sampled = pd.concat([grouped[index] for index in rng.integers(0, len(grouped), len(grouped))], ignore_index=True)
                value = _spearman(
                    pd.to_numeric(sampled["average_auxiliary_similarity"], errors="coerce"),
                    pd.to_numeric(sampled["improvement_positive"], errors="coerce"),
                )
                if math.isfinite(value):
                    bootstrap.append(value)
        row["ci95_lower"] = float(np.quantile(bootstrap, 0.025)) if bootstrap else np.nan
        row["ci95_upper"] = float(np.quantile(bootstrap, 0.975)) if bootstrap else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _job_completeness(
    results: pd.DataFrame,
    failures: pd.DataFrame,
    expected: pd.DataFrame,
    duplicate_counts: dict[str, int],
) -> pd.DataFrame:
    result_status = {
        str(row.get("run_id", "")): str(row.get("status", "completed")) for _, row in results.iterrows()
    }
    completed_ids = set(result_status)
    failed_ids = set(failures.get("run_id", pd.Series(dtype=str)).astype(str))
    candidates = expected if not expected.empty else pd.concat(
        [
            results[[column for column in results.columns if column in {"run_id", *GROUP_COLUMNS}]],
            failures[[column for column in failures.columns if column in {"run_id", *GROUP_COLUMNS}]],
        ],
        ignore_index=True,
    )
    if "run_id" not in candidates:
        return pd.DataFrame(columns=["run_id", "status", "duplicate_result_count"])
    rows = []
    for _, job in candidates.drop_duplicates("run_id", keep="last").iterrows():
        run_id = str(job.get("run_id", ""))
        status = result_status.get(run_id, "completed") if run_id in completed_ids else ("failed" if run_id in failed_ids else "missing")
        row = {column: job.get(column) for column in ["run_id", *GROUP_COLUMNS] if column in candidates}
        row.update({"status": status, "duplicate_result_count": duplicate_counts.get(run_id, 1) - 1})
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["expected_job_count"] = len(frame)
        frame["completed_job_count"] = int(frame["status"].isin(["completed", "skipped"]).sum())
        frame["failed_job_count"] = int(frame["status"].eq("failed").sum())
        frame["missing_job_count"] = int(frame["status"].eq("missing").sum())
    return frame


def _environment_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize immutable lock identity and the observed runtime fingerprint."""

    columns = [
        "environment_lock_hash",
        "environment_runtime_hash",
        "python_version",
        "torch_version",
        "torch_cuda_build",
        "cudnn_version",
        "driver_version",
        "gpu_names_json",
        "number_runs",
        "number_targets",
        "number_git_revisions",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    normalized = results.copy()
    runtime_rows: list[dict[str, Any]] = []
    for _, result in normalized.iterrows():
        report = result.get("environment_report", {})
        if isinstance(report, str):
            try:
                report = json.loads(report)
            except json.JSONDecodeError:
                report = {}
        report = dict(report) if isinstance(report, dict) else {}
        packages = report.get("packages", {})
        packages = dict(packages) if isinstance(packages, dict) else {}
        runtime_hash = str(
            result.get("environment_runtime_hash", "")
            or report.get("environment_runtime_hash", "")
            or canonical_json_hash(
                {
                    "python_version": report.get("python_version", "untracked"),
                    "packages": packages,
                    "os": report.get("os", "untracked"),
                    "torch_cuda_build": report.get("torch_cuda_build"),
                    "cudnn_version": report.get("cudnn_version"),
                    "gpu_names": report.get("gpu_names", []),
                    "driver_version": report.get("driver_version", "unavailable"),
                }
            )
        )
        runtime_rows.append(
            {
                "environment_lock_hash": str(result.get("environment_lock_hash", "untracked")),
                "environment_runtime_hash": runtime_hash,
                "python_version": report.get("python_version", "untracked"),
                "torch_version": packages.get("torch", "untracked"),
                "torch_cuda_build": report.get("torch_cuda_build"),
                "cudnn_version": report.get("cudnn_version"),
                "driver_version": report.get("driver_version", "unavailable"),
                "gpu_names_json": json.dumps(report.get("gpu_names", []), sort_keys=True),
                "target_synset": result.get("target_synset"),
                "git_sha": result.get("git_sha"),
            }
        )
    frame = pd.DataFrame(runtime_rows)
    group_columns = columns[:-3]
    for keys, group in frame.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "number_runs": int(len(group)),
                "number_targets": int(group["target_synset"].nunique(dropna=True)),
                "number_git_revisions": int(group["git_sha"].nunique(dropna=True)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _readiness_summary(
    results: pd.DataFrame,
    failures: pd.DataFrame,
    completeness: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    invalid_record_count: int,
) -> dict[str, Any]:
    status_counts = (
        completeness.get("status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict()
        if not completeness.empty
        else {}
    )
    pair_counts = (
        pairs.get("pair_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict()
        if not pairs.empty
        else {}
    )
    expected = int(len(completeness))
    missing_or_failed = int(status_counts.get("missing", 0)) + int(status_counts.get("failed", 0))
    noncompleted_pairs = sum(int(value) for key, value in pair_counts.items() if key != "completed")
    complete = bool(
        expected
        and not missing_or_failed
        and not noncompleted_pairs
        and failures.empty
        and invalid_record_count == 0
    )
    return {
        "schema_version": "1.0",
        "status": "complete" if complete else ("incomplete" if expected else "not_evaluable"),
        "meaning": "result completeness only; release readiness is decided by the pilot validator",
        "expected_job_count": expected,
        "valid_result_count": int(len(results)),
        "failure_record_count": int(len(failures)),
        "invalid_result_count": int(invalid_record_count),
        "job_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "pair_status_counts": {str(key): int(value) for key, value in pair_counts.items()},
        "noncompleted_pair_count": int(noncompleted_pairs),
        "number_targets": int(results.get("target_synset", pd.Series(dtype=str)).nunique(dropna=True)),
        "number_independent_training_subsets": int(
            results.get("subset_manifest_hash", pd.Series(dtype=str)).replace("", np.nan).nunique(dropna=True)
        ),
        "number_environment_locks": int(
            results.get("environment_lock_hash", pd.Series(dtype=str)).replace("", np.nan).nunique(dropna=True)
        ),
    }


def aggregate_results(results_root: str | Path, *, expected_job_paths: Iterable[str | Path] | None = None) -> dict[str, pd.DataFrame]:
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    results, invalid_records, duplicate_counts = _load_run_records(root)
    failures = _load_failures(root)
    if invalid_records:
        failures = pd.concat([failures, pd.DataFrame(invalid_records)], ignore_index=True, sort=False)
    supplied_paths = list(expected_job_paths or [])
    if not supplied_paths:
        supplied_paths = sorted(root.glob("**/jobs/*.csv"))
    expected = _load_expected(supplied_paths)
    failed_ids = set(failures.get("run_id", pd.Series(dtype=str)).astype(str))
    pairs = compute_paired_gaps(results, expected_jobs=expected, failed_run_ids=failed_ids)
    subset_summaries = summarize_by_training_subset(pairs)
    target_summaries = summarize_by_target(subset_summaries)
    hierarchical_summaries = hierarchical_target_bootstrap(subset_summaries)
    # ``summary_metrics.csv`` remains the compact compatibility output; its
    # paired portion now uses the correctly nested target-level summaries.
    paired_summary = target_summaries.copy()
    if not paired_summary.empty:
        paired_summary["summary_type"] = "paired_transfer_gap"
    raw_summary = summarize_raw_metrics(results)
    summary = pd.concat([raw_summary, paired_summary], ignore_index=True, sort=False)
    completeness = _job_completeness(results, failures, expected, duplicate_counts)
    correlations = similarity_correlations(pairs)
    environment_summary = _environment_summary(results)
    readiness_summary = _readiness_summary(
        results,
        failures,
        completeness,
        pairs,
        invalid_record_count=len(invalid_records),
    )

    outputs = {
        "all_metrics": results,
        "summary_metrics": summary,
        "paired_transfer_gaps": pairs,
        "subset_level_summaries": subset_summaries,
        "target_level_summaries": target_summaries,
        "hierarchical_summaries": hierarchical_summaries,
        "job_completeness": completeness,
        "failed_jobs": failures,
        "environment_summary": environment_summary,
        "similarity_correlations": correlations,
    }
    for name, frame in outputs.items():
        _atomic_to_csv(frame, root / f"{name}.csv")
    atomic_write_json(readiness_summary, root / "readiness_summary.json")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--expected-jobs", action="append", default=[])
    args = parser.parse_args()
    outputs = aggregate_results(args.results_root, expected_job_paths=args.expected_jobs)
    print(f"aggregated {len(outputs['all_metrics'])} valid runs into {Path(args.results_root) / 'all_metrics.csv'}")


if __name__ == "__main__":
    main()
