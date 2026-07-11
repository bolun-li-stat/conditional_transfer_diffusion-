"""Train, sample, and evaluate one rigorous image-transfer job."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from image_transfer.config import load_resolved_config
from image_transfer.data import (
    ManifestInsufficientDataError,
    build_datasets_for_job,
    canonical_sha256,
    target_training_subset,
)
from image_transfer.diffusion import ImageDDIM, ImageDDPM
from image_transfer.evaluation.classifier_fidelity import (
    evaluate_classifier_fidelity,
    load_synset_index_mapping,
    preflight_classifier_fidelity,
)
from image_transfer.evaluation.corruption_bank import (
    CorruptionBank,
    create_corruption_bank,
    evaluate_corruption_bank,
    load_corruption_bank,
    save_corruption_bank,
)
from image_transfer.evaluation.feature_metrics import (
    compute_feature_metrics,
    preflight_feature_metric_backend,
)
from image_transfer.evaluation.feature_similarity import average_auxiliary_similarity, build_feature_extractor
from image_transfer.evaluation.nearest_neighbors import (
    calibrate_near_duplicate_threshold,
    compute_memorization_diagnostics,
    make_memorization_grid,
)
from image_transfer.models.model_factory import model_config_hash, resolve_model_config
from image_transfer.readiness import enforce_readiness_gate
from image_transfer.scripts.make_job_grid import (
    EXP_DIR,
    compute_manifest_keys,
    compute_resolved_run_spec_hash,
    run_id_for_row,
)
from image_transfer.scripts.inspect_environment import inspect_environment
from image_transfer.scripts.prepare_metric_assets import verify_manifest
from image_transfer.training.checkpointing import load_checkpoint
from image_transfer.training.trainer import train_image_model
from image_transfer.utils.device import get_device
from image_transfer.utils.io import (
    ensure_dir,
    failure_result_path,
    get_git_sha,
    load_valid_result,
    resolve_env_path,
    run_result_path,
    write_run_result,
)
from image_transfer.utils.seed import make_torch_generator


# Retained for old imports. Results are no longer appended to shared CSV files;
# aggregation discovers schema-valid per-run JSON records instead.
FIELDS = [
    "run_id", "status", "dataset", "experiment", "target_synset", "model_type", "aux_set",
    "n0", "m_per_aux", "K_aux", "training_protocol", "data_split_seed",
    "model_initialization_seed", "training_seed", "sampling_seed", "evaluation_seed",
    "manifest_hash", "target_training_subset_hash", "optimizer_steps", "target_examples_seen",
    "auxiliary_examples_seen", "total_examples_seen", "effective_target_fraction",
    "validation_epsilon_mse_target", "fid_target", "kid_target_mean", "kid_target_std",
    "precision_target", "recall_target", "density_target", "coverage_target",
    "classifier_target_top1_acc", "classifier_target_top5_acc", "auxiliary_leakage_rate",
]

EQUAL_TOTAL_MODEL_TYPES = {"unconditional_equal_total", "conditional_target_only_equal_total"}


def read_job(path: str | Path, index: int) -> dict[str, str]:
    import csv

    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if index < 0 or index >= len(rows):
        raise IndexError(f"job-index {index} out of range for {len(rows)} jobs")
    return rows[index]


def _arg_or_job(args, job: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    cli_value = getattr(args, key, None)
    if cli_value is not None:
        return cli_value
    if job is not None and job.get(key) not in {None, ""}:
        return job[key]
    return default


def _is_conditional(model_type: str) -> bool:
    return not str(model_type).startswith("unconditional")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _config_relative_path(value: Any, config_path: str | Path) -> Path:
    expanded = resolve_env_path(None if value is None else str(value))
    if not expanded:
        raise ValueError("required path is empty after environment expansion")
    path = Path(expanded).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config_path).resolve().parent / path).resolve()


def _boolean_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in {None, ""}:
        return False
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def _direct_run_id(fields: Mapping[str, Any]) -> str:
    return run_id_for_row(fields)


def deterministic_run_id(job: Mapping[str, Any] | None = None, **fields: Any) -> str:
    if job and job.get("run_id"):
        if job.get("resolved_run_spec_hash") and str(job["run_id"]) != _direct_run_id(fields):
            raise ValueError("job grid run_id does not match the resolved run settings; regenerate the grid")
        return str(job["run_id"])
    return _direct_run_id(fields)


def _atomic_torch_save(value: Any, path: str | Path) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temporary = handle.name
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return destination


def _collect_dataset(dataset, *, limit: int | None, batch_size: int, num_workers: int = 0) -> torch.Tensor:
    if dataset is None or len(dataset) == 0:
        return torch.empty(0)
    chunks: list[torch.Tensor] = []
    count = 0
    for images, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
        if limit is not None:
            remaining = int(limit) - count
            if remaining <= 0:
                break
            images = images[:remaining]
        chunks.append(images.detach().cpu())
        count += int(images.shape[0])
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0)


def _bank_path(root: Path, bank: CorruptionBank) -> Path:
    return root / f"{bank.split}_{bank.bank_hash}.json"


def _get_or_create_bank(
    root: Path,
    *,
    split: str,
    manifest_hash: str,
    evaluation_seed: int,
    timesteps: int,
    corruptions_per_image: int,
    num_images: int,
    noise_bins: Any,
) -> tuple[CorruptionBank, Path]:
    expected = create_corruption_bank(
        manifest_hash=manifest_hash,
        evaluation_seed=evaluation_seed,
        timesteps=timesteps,
        corruptions_per_image=corruptions_per_image,
        num_images=num_images,
        noise_bins=noise_bins,
        split=split,
    )
    path = _bank_path(root, expected)
    if path.exists():
        loaded = load_corruption_bank(path)
        if loaded.bank_hash != expected.bank_hash:
            raise ValueError(f"corruption bank at {path} does not match its deterministic specification")
        return loaded, path
    save_corruption_bank(expected, path)
    return expected, path


def _bank_metrics(metrics: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    result = dict(metrics)
    bank_hash = result.pop("corruption_bank_hash", None)
    manifest_hash = result.pop("corruption_bank_manifest_hash", None)
    distribution = result.pop("corruption_timestep_distribution", None)
    corruptions = result.get("num_corruptions")
    if corruptions is not None:
        result[f"num_{split}_corruptions"] = corruptions
        if split != "validation":
            result.pop("num_corruptions")
    corruptions_per_image = result.get("corruptions_per_image")
    if corruptions_per_image is not None:
        result[f"{split}_corruptions_per_image"] = corruptions_per_image
        if split != "validation":
            result.pop("corruptions_per_image")
    result[f"{split}_corruption_bank_hash"] = bank_hash
    result[f"{split}_corruption_bank_manifest_hash"] = manifest_hash
    result[f"{split}_corruption_timestep_distribution"] = distribution
    if split == "validation":
        # Compatibility field requested by the evaluation schema.
        result["corruption_bank_hash"] = bank_hash
    return result


def _mse_gap(holdout_value: Any, train_value: Any) -> float:
    """Return holdout minus train MSE, preserving unavailable diagnostics."""

    try:
        holdout = float(holdout_value)
        train = float(train_value)
    except (TypeError, ValueError):
        return float("nan")
    return holdout - train if torch.isfinite(torch.tensor([holdout, train])).all() else float("nan")


def _sampling_process(
    *, sampler: str, timesteps: int, schedule: str, device: torch.device, ddim_eta: float
):
    if sampler == "ddpm":
        return ImageDDPM(timesteps=timesteps, schedule=schedule, device=device)
    if sampler == "ddim":
        return ImageDDIM(timesteps=timesteps, schedule=schedule, device=device, eta=ddim_eta)
    raise ValueError(f"Unknown sampler {sampler!r}; expected 'ddpm' or 'ddim'")


def sample_batched(
    diffusion,
    model,
    *,
    num_samples: int,
    image_size: int,
    conditional: bool,
    label: int,
    sampling_steps: int,
    batch_size: int,
    sampling_seed: int,
    ddim_eta: float = 0.0,
) -> torch.Tensor:
    """Sample with an RNG stream independent from training and evaluation."""

    if int(batch_size) < 1 or int(num_samples) < 1:
        raise ValueError("sampling batch size and num_samples must be positive")
    generator = make_torch_generator(sampling_seed, diffusion.device)
    chunks: list[torch.Tensor] = []
    for start in range(0, int(num_samples), int(batch_size)):
        current = min(int(batch_size), int(num_samples) - start)
        labels = torch.full((current,), int(label), dtype=torch.long, device=diffusion.device) if conditional else None
        kwargs = {"eta": float(ddim_eta)} if isinstance(diffusion, ImageDDIM) else {}
        chunk = diffusion.sample(
            model,
            (current, 3, int(image_size), int(image_size)),
            y=labels,
            steps=int(sampling_steps),
            generator=generator,
            **kwargs,
        )
        chunks.append(chunk.detach().cpu())
    return torch.cat(chunks, dim=0)


def _disabled_classifier_metrics(cfg: Mapping[str, Any]) -> dict[str, Any]:
    reason = str(
        cfg.get(
            "classifier_unavailable_reason",
            "classifier fidelity was disabled by configuration",
        )
    )
    return {
        "classifier_target_top1_acc": float("nan"),
        "classifier_target_top5_acc": float("nan"),
        "auxiliary_leakage_rate": float("nan"),
        "top1_prediction_histogram_json": "{}",
        "classifier_fidelity_status": "disabled",
        "classifier_unavailable_reason": reason,
        "classifier_architecture": "disabled",
        "classifier_weights": "disabled",
        "classifier_preprocessing": "disabled",
    }


def _write_skip_result(
    results_root: Path,
    run_id: str,
    *,
    job_fields: Mapping[str, Any],
    config_path: str,
    git_sha: str,
    reason: str,
    details: Mapping[str, Any],
    metadata_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "status": "skipped",
        "job": dict(job_fields),
        "metadata": {
            **dict(metadata_extra or {}),
            "config_path": str(config_path),
            "git_sha": git_sha,
            "skip_reason": reason,
            "available_count": details.get(
                "available",
                details.get(
                    "available_target_train_after_reservations",
                    details.get("available_target_train", 0),
                ),
            ),
            "skip_details_json": json.dumps(dict(details), sort_keys=True),
        },
        "training": {},
        "metrics": {},
    }
    write_run_result(results_root, run_id, record)
    return {"run_id": run_id, **record}


def run(args, job: dict[str, Any] | None = None) -> dict[str, Any]:
    run_started = time.time()
    if bool(getattr(args, "force", False)) and bool(getattr(args, "resume", False)):
        raise ValueError("--force and --resume are mutually exclusive")
    resolved_info = load_resolved_config(args.config)
    cfg = resolved_info.resolved
    lock_path = _config_relative_path(
        cfg.get("environment_lock_path", "../../environment/requirements-image-lock.txt"),
        args.config,
    )
    environment_report = inspect_environment(lock_path)
    if environment_report["environment_lock_hash"] != resolved_info.environment_lock_hash:
        raise RuntimeError("runtime environment lock does not match the resolved configuration")
    if not bool(environment_report.get("lock_matches_runtime", False)):
        raise RuntimeError(
            "runtime packages do not match the selected environment definition: "
            + "; ".join(environment_report.get("lock_mismatches", []))
        )
    environment_runtime_hash = str(environment_report.get("environment_runtime_hash", ""))
    git_sha = get_git_sha()
    job_requests_override = _boolean_value((job or {}).get("readiness_gate_override"))
    cli_requests_override = bool(getattr(args, "override_readiness_gate", False))
    if job is not None and cli_requests_override and not job_requests_override:
        raise ValueError("regenerate the job grid with --override-readiness-gate before running it")
    readiness_gate = enforce_readiness_gate(
        cfg,
        override=job_requests_override or cli_requests_override,
        current_git_sha=git_sha,
        config_source_path=resolved_info.source_path or args.config,
    )
    readiness_fields = {
        "readiness_gate_required": bool(readiness_gate.get("required", False)),
        "readiness_gate_override": bool(readiness_gate.get("override", False)),
        "readiness_gate_status": str(readiness_gate.get("status", "not_applicable")),
        "readiness_gate_passed": bool(
            readiness_gate.get("passed", not readiness_gate.get("required", False))
        ),
        "readiness_gate_mismatches": json.dumps(readiness_gate.get("mismatches", []), sort_keys=True),
        "readiness_status_path": str(readiness_gate.get("status_path", "")),
        "readiness_status_file_hash": str(readiness_gate.get("status_file_hash", "")),
        "readiness_pilot_config_hash": str(readiness_gate.get("pilot_config_hash", "")),
        "readiness_validated_git_sha": str(readiness_gate.get("validated_git_sha", "")),
        "readiness_current_git_sha": str(readiness_gate.get("current_git_sha", "")),
    }
    for field, expected in readiness_fields.items():
        recorded = (job or {}).get(field)
        if recorded not in {None, ""} and str(recorded) != str(expected):
            raise ValueError(f"job grid {field} no longer matches the readiness gate; regenerate the grid")
    computed_config_hash = resolved_info.resolved_hash
    for field, expected in (
        ("raw_config_hash", resolved_info.raw_hash),
        ("resolved_config_hash", resolved_info.resolved_hash),
        ("config_hash", resolved_info.resolved_hash),
        ("study_plan_hash", resolved_info.study_plan_hash),
        ("target_set_hash", resolved_info.target_set_hash),
        ("environment_lock_hash", resolved_info.environment_lock_hash),
    ):
        recorded = (job or {}).get(field)
        if recorded not in {None, "", expected}:
            raise ValueError(f"job grid {field} does not match the resolved config; regenerate the grid")

    experiment = str(_arg_or_job(args, job, "experiment", "A"))
    if experiment not in EXP_DIR:
        raise ValueError(f"Unknown experiment {experiment!r}")
    legacy_seed = int(_arg_or_job(args, job, "seed", 0))
    split_cfg = cfg.get("data_split", {})
    training_cfg = cfg.get("training", {})
    sampling_cfg = cfg.get("sampling", {})
    evaluation_cfg = cfg.get("evaluation", {})
    seed_design = cfg.get("seed_design", {})
    default_pair = (seed_design.get("optimization_seed_pairs") or [{}])[0]
    holdout_seed = int(
        _arg_or_job(args, job, "holdout_seed", seed_design.get("holdout_seed", split_cfg.get("holdout_seed", legacy_seed)))
    )
    training_subset_seed = int(
        _arg_or_job(
            args,
            job,
            "training_subset_seed",
            (seed_design.get("training_subset_seeds") or [split_cfg.get("training_subset_seed", legacy_seed)])[0],
        )
    )
    # Compatibility for builders and old result readers.  In new jobs this is
    # the fixed holdout seed, never the variable training-subset seed.
    data_split_seed = int(_arg_or_job(args, job, "data_split_seed", holdout_seed))
    model_initialization_seed = int(
        _arg_or_job(args, job, "model_initialization_seed", default_pair.get("model_initialization_seed", legacy_seed))
    )
    training_seed = int(_arg_or_job(args, job, "training_seed", default_pair.get("training_seed", legacy_seed)))
    sampling_seed = int(
        _arg_or_job(args, job, "sampling_seed", seed_design.get("sampling_seed", sampling_cfg.get("seed", legacy_seed)))
    )
    evaluation_seed = int(
        _arg_or_job(args, job, "evaluation_seed", seed_design.get("evaluation_seed", evaluation_cfg.get("seed", legacy_seed)))
    )

    n0 = int(_arg_or_job(args, job, "n0", cfg.get("n0_values", [100])[0]))
    m_per_aux = int(_arg_or_job(args, job, "m_per_aux", n0))
    k_aux = int(_arg_or_job(args, job, "K_aux", cfg.get("K_aux", 5)))
    model_type = str(_arg_or_job(args, job, "model_type", "unconditional_n0"))
    aux_set = str(_arg_or_job(args, job, "aux_set", "none"))
    aux_draw_id = str(_arg_or_job(args, job, "aux_draw_id", "none"))
    target = cfg.get("targets", [{"synset": "dog", "name": "dog"}])[0]
    target_synset = str(_arg_or_job(args, job, "target_synset", target.get("synset", "dog")))
    target_name = str(_arg_or_job(args, job, "target_name", target.get("name", target_synset)))
    training_protocol = str(_arg_or_job(args, job, "training_protocol", training_cfg.get("protocol", "natural_compute_matched")))
    sampler = str(_arg_or_job(args, job, "sampler", sampling_cfg.get("sampler", "ddpm"))).lower()
    image_size = int(_arg_or_job(args, job, "image_size", cfg.get("image_size", 32)))
    sampling_steps = int(sampling_cfg.get("steps", cfg.get("diffusion", {}).get("timesteps", 1000)))
    ddim_eta = float(sampling_cfg.get("ddim_eta", 0.0))
    training_steps = int(
        getattr(args, "max_steps", None) if getattr(args, "max_steps", None) is not None else training_cfg.get("steps", 1000)
    )
    num_generated = int(getattr(args, "num_generated", None) or cfg.get("num_generated", 64))
    architecture_profile = str(
        _arg_or_job(args, job, "architecture_profile", cfg.get("model", {}).get("profile", "legacy"))
    )
    configured_model = dict(cfg.get("model") or {})
    if architecture_profile != str(configured_model.get("profile", "legacy")):
        run_model_cfg = {
            "architecture": str(configured_model.get("architecture", "legacy_simple_unet")),
            "profile": architecture_profile,
        }
    else:
        run_model_cfg = configured_model
    resolved_model_config = resolve_model_config(run_model_cfg, image_size=image_size)
    resolved_model_hash = model_config_hash(resolved_model_config)
    if (job or {}).get("model_config_hash") not in {None, "", resolved_model_hash}:
        raise ValueError("job grid model_config_hash does not match the resolved model; regenerate the grid")
    split_manifest_key, subset_manifest_key = compute_manifest_keys(
        cfg,
        target_synset,
        {
            "holdout_seed": holdout_seed,
            "training_subset_seed": training_subset_seed,
        },
    )

    job_fields: dict[str, Any] = {
        "dataset": cfg.get("dataset", "cifar10"),
        "experiment": experiment,
        "experiment_name": {"A": "equal_target", "B": "equal_total", "C": "similarity_sweep"}[experiment],
        "target_synset": target_synset,
        "target_name": target_name,
        "model_type": model_type,
        "aux_set": aux_set,
        "aux_composition": (job or {}).get("aux_composition", "[]"),
        "aux_draw_id": aux_draw_id,
        "aux_draw_seed": int(_arg_or_job(args, job, "aux_draw_seed", cfg.get("aux_draw_seed", 0))),
        "aux_unique_combinations": int(_arg_or_job(args, job, "aux_unique_combinations", 1)),
        "n0": n0,
        "m_per_aux": m_per_aux,
        "K_aux": k_aux,
        "total_auxiliary_budget": m_per_aux * k_aux,
        "auxiliary_ratio": float(m_per_aux / n0) if n0 else 0.0,
        "baseline_target_count": int(
            _arg_or_job(
                args,
                job,
                "baseline_target_count",
                n0 + m_per_aux * k_aux if experiment == "B" else n0,
            )
        ),
        "architecture": str(resolved_model_config["architecture"]),
        "architecture_profile": architecture_profile,
        "model_config_hash": resolved_model_hash,
        "seed": training_seed,
        "data_split_seed": data_split_seed,
        "holdout_seed": holdout_seed,
        "training_subset_seed": training_subset_seed,
        "model_initialization_seed": model_initialization_seed,
        "training_seed": training_seed,
        "sampling_seed": sampling_seed,
        "evaluation_seed": evaluation_seed,
        "training_protocol": training_protocol,
        "sampler": sampler,
        "sampling_steps": sampling_steps,
        "image_size": image_size,
        "training_steps": training_steps,
        "num_generated": num_generated,
        "ddim_eta": ddim_eta,
        "raw_config_hash": resolved_info.raw_hash,
        "resolved_config_hash": resolved_info.resolved_hash,
        "study_plan_hash": resolved_info.study_plan_hash,
        "target_set_hash": resolved_info.target_set_hash,
        "environment_lock_hash": resolved_info.environment_lock_hash,
        **readiness_fields,
        "split_manifest_key": split_manifest_key,
        "subset_manifest_key": subset_manifest_key,
        "config_hash": computed_config_hash,
    }
    resolved_run_spec_hash = compute_resolved_run_spec_hash(job_fields)
    job_fields["resolved_run_spec_hash"] = resolved_run_spec_hash
    job_fields["effective_run_spec_hash"] = resolved_run_spec_hash
    for field in ("resolved_run_spec_hash", "effective_run_spec_hash", "split_manifest_key", "subset_manifest_key"):
        recorded = (job or {}).get(field)
        if recorded not in {None, "", job_fields[field]}:
            raise ValueError(f"job grid {field} does not match the resolved run settings")
    run_id = deterministic_run_id(job, **job_fields)
    job_fields["run_id"] = run_id
    results_root = Path(resolve_env_path(cfg.get("output_root"), "image_transfer_results"))
    outdir = Path((job or {}).get("output_dir") or results_root / EXP_DIR[experiment])
    result_path = run_result_path(results_root, run_id)

    if result_path.exists() and not getattr(args, "force", False):
        try:
            existing = load_valid_result(result_path, expected_run_id=run_id)
        except (OSError, ValueError):
            pass
        else:
            print(f"skipped completed run: {result_path}")
            return existing

    checkpoint_dir = outdir / "checkpoints"
    last_checkpoint_path = checkpoint_dir / f"{run_id}_last.pt"
    best_checkpoint_path = checkpoint_dir / f"{run_id}_best.pt"
    train_log_path = outdir / "logs" / f"{run_id}_train_log.csv"
    run_config_path = outdir / "configs" / f"{run_id}.yaml"
    sample_path = outdir / "samples" / f"{run_id}_samples.pt"
    ensure_dir(outdir)
    if getattr(args, "dry_run", False):
        preview = {**job_fields, "output_dir": str(outdir), "result_path": str(result_path)}
        print(json.dumps(preview, indent=2, sort_keys=True))
        return preview

    failure_path = failure_result_path(results_root, run_id)
    if getattr(args, "force", False) and not getattr(args, "resume", False):
        for stale_path in (
            result_path,
            failure_path,
            last_checkpoint_path,
            best_checkpoint_path,
            train_log_path,
            sample_path,
            run_config_path,
        ):
            stale_path.unlink(missing_ok=True)
    elif not getattr(args, "resume", False) and (last_checkpoint_path.exists() or best_checkpoint_path.exists()):
        raise FileExistsError(
            f"incomplete checkpoint artifacts exist for {run_id}; pass --resume to continue or --force to restart"
        )

    builder_job = dict(job or {})
    builder_job.update({
        "experiment": experiment,
        "target_synset": target_synset,
        "model_type": model_type,
        "data_split_seed": data_split_seed,
        "holdout_seed": holdout_seed,
        "training_subset_seed": training_subset_seed,
    })
    try:
        bundle = build_datasets_for_job(
            cfg,
            builder_job,
            n0=n0,
            m_per_aux=m_per_aux,
            k_aux=k_aux,
            seed=training_seed,
            model_type=model_type,
        )
    except ManifestInsufficientDataError as exception:
        exp_cfg = cfg.get("experiments", {}).get(experiment, {})
        action = str(split_cfg.get("insufficient_data_action", "fail"))
        may_skip = action == "skip" or (
            model_type in EQUAL_TOTAL_MODEL_TYPES and bool(exp_cfg.get("skip_if_insufficient_target_images", False))
        )
        if not may_skip:
            raise
        shortage_subject = str(exception.details.get("what", exception.details.get("reason", "target"))).lower()
        skip_reason = (
            "insufficient_auxiliary_images_after_holdout"
            if "auxiliary" in shortage_subject
            else "insufficient_target_images_after_holdout"
        )
        record = _write_skip_result(
            results_root,
            run_id,
            job_fields=job_fields,
            config_path=args.config,
            git_sha=git_sha,
            reason=skip_reason,
            details=exception.details,
            metadata_extra={
                "raw_config_hash": resolved_info.raw_hash,
                "resolved_config_hash": resolved_info.resolved_hash,
                "study_plan_hash": resolved_info.study_plan_hash,
                "target_set_hash": resolved_info.target_set_hash,
                "environment_lock_hash": resolved_info.environment_lock_hash,
                "environment_runtime_hash": environment_runtime_hash,
                "environment_report": environment_report,
                "readiness_gate": readiness_gate,
                **readiness_fields,
            },
        )
        failure_path.unlink(missing_ok=True)
        print(f"skipped {run_id}: insufficient target images after fixed holdouts")
        return record

    ensure_dir(run_config_path.parent)
    with open(run_config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "raw_config": resolved_info.raw,
                "resolved_config": cfg,
                "environment_report": environment_report,
                "readiness_gate": readiness_gate,
                **readiness_fields,
                **resolved_info.provenance(),
                "resolved_model_config": resolved_model_config,
                "model_config_hash": resolved_model_hash,
                "job": builder_job,
                "run_id": run_id,
            },
            handle,
            sort_keys=False,
        )

    target_subset_count = n0 + k_aux * m_per_aux if model_type in EQUAL_TOTAL_MODEL_TYPES else n0
    legacy_target_subset_hash = canonical_sha256(target_training_subset(bundle.manifest, target_subset_count))
    split_manifest_hash = str(
        getattr(bundle, "split_manifest_hash", "")
        or bundle.manifest.get("split_manifest_hash", "")
        or bundle.manifest_hash
    )
    subset_manifest_hash = str(
        getattr(bundle, "subset_manifest_hash", "")
        or bundle.manifest.get("subset_manifest_hash", "")
        or bundle.manifest_hash
    )
    target_subset_hash = str(
        getattr(bundle, "target_training_subset_hash", "")
        or bundle.manifest.get("target_training_subset_hash", "")
        or legacy_target_subset_hash
    )

    timesteps = int(cfg.get("diffusion", {}).get("timesteps", 1000))
    schedule = str(cfg.get("diffusion", {}).get("schedule", "linear"))
    corruption_bank_root = Path(
        resolve_env_path(evaluation_cfg.get("corruption_bank_root"), str(results_root / "corruption_banks"))
    )
    legacy_corruptions = evaluation_cfg.get("corruptions_per_image")
    validation_corruptions_per_image = int(
        evaluation_cfg.get(
            "validation_corruptions_per_image",
            legacy_corruptions if legacy_corruptions is not None else 16,
        )
    )
    test_corruptions_per_image = int(
        evaluation_cfg.get(
            "test_corruptions_per_image",
            legacy_corruptions if legacy_corruptions is not None else 16,
        )
    )
    train_corruptions_per_image = int(
        evaluation_cfg.get(
            "train_diagnostic_corruptions_per_image",
            legacy_corruptions if legacy_corruptions is not None else 8,
        )
    )
    train_diagnostic_max_images = max(
        1,
        int(evaluation_cfg.get("train_diagnostic_max_images", min(len(bundle.target_train_eval), 256))),
    )
    num_train_diagnostic_images = min(len(bundle.target_train_eval), train_diagnostic_max_images)
    noise_bins = evaluation_cfg.get("noise_bins")
    validation_bank, validation_bank_path = _get_or_create_bank(
        corruption_bank_root,
        split="validation",
        manifest_hash=split_manifest_hash,
        evaluation_seed=evaluation_seed,
        timesteps=timesteps,
        corruptions_per_image=validation_corruptions_per_image,
        num_images=len(bundle.target_val),
        noise_bins=noise_bins,
    )
    test_bank, test_bank_path = _get_or_create_bank(
        corruption_bank_root,
        split="test",
        manifest_hash=split_manifest_hash,
        evaluation_seed=evaluation_seed,
        timesteps=timesteps,
        corruptions_per_image=test_corruptions_per_image,
        num_images=len(bundle.target_eval),
        noise_bins=noise_bins,
    )
    train_bank_identity = canonical_sha256(
        {
            "split_manifest_hash": split_manifest_hash,
            "subset_manifest_hash": subset_manifest_hash,
            "target_training_subset_hash": target_subset_hash,
        }
    )
    train_bank, train_bank_path = _get_or_create_bank(
        corruption_bank_root,
        split="train_diagnostic",
        manifest_hash=train_bank_identity,
        evaluation_seed=evaluation_seed,
        timesteps=timesteps,
        corruptions_per_image=train_corruptions_per_image,
        num_images=num_train_diagnostic_images,
        noise_bins=noise_bins,
    )

    device = get_device(_arg_or_job(args, job, "device", cfg.get("device", "auto")))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batch_size = int(training_cfg.get("batch_size", 32))
    conditional = _is_conditional(model_type)
    mode = str(evaluation_cfg.get("mode", "debug" if cfg.get("use_fake_data", False) else "strict")).lower()
    classifier_enabled = bool(
        evaluation_cfg.get("compute_classifier_fidelity", evaluation_cfg.get("compute_classifier", False))
    )
    compute_similarity = bool(evaluation_cfg.get("compute_feature_similarity", experiment == "C"))
    compute_neighbors = bool(evaluation_cfg.get("make_nearest_neighbors", False))
    classifier_mapping = None
    classifier_mapping_value = evaluation_cfg.get(
        "classifier_synset_mapping_path", cfg.get("classifier_synset_mapping_path")
    )
    if classifier_mapping_value:
        resolved_mapping = Path(str(resolve_env_path(str(classifier_mapping_value))))
        if not resolved_mapping.is_absolute():
            resolved_mapping = Path(args.config).resolve().parent / resolved_mapping
        classifier_mapping = load_synset_index_mapping(resolved_mapping)

    # Strict runs fail on missing metric/classifier/diagnostic weights before
    # spending compute on training. Debug mode remains fully offline-capable.
    if mode == "strict":
        asset_manifest_path = _config_relative_path(cfg.get("metric_assets_manifest_path"), args.config)
        asset_root = asset_manifest_path.parent
        os.environ["TORCH_HOME"] = str(asset_root)
        torch.hub.set_dir(str(asset_root / "hub"))
        verify_manifest(asset_root, asset_manifest_path, verify_runtime=True)
        preflight_feature_metric_backend(
            feature_dimension=int(evaluation_cfg.get("feature_dimension", 2048)),
            device=device,
        )
        if classifier_enabled:
            preflight_classifier_fidelity(
                target_synset,
                bundle.aux_synsets,
                dataset_name=str(cfg.get("dataset", "imagenet")),
                synset_to_index=classifier_mapping,
            )
        if compute_similarity or compute_neighbors:
            diagnostic_preflight = build_feature_extractor(device, strict=True)
            del diagnostic_preflight
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model, training_diffusion, train_metrics = train_image_model(
        bundle.train,
        bundle.target_val,
        conditional=conditional,
        num_classes=max(len(bundle.class_labels), 1),
        image_size=image_size,
        model_cfg=resolved_model_config,
        timesteps=timesteps,
        schedule=schedule,
        steps=training_steps,
        batch_size=batch_size,
        lr=float(cfg.get("optimizer", {}).get("lr", 2e-4)),
        optimizer_name=str(cfg.get("optimizer", {}).get("name", "adamw")),
        optimizer_betas=tuple(cfg.get("optimizer", {}).get("betas", [0.9, 0.999])),
        optimizer_eps=float(cfg.get("optimizer", {}).get("eps", 1.0e-8)),
        weight_decay=float(cfg.get("optimizer", {}).get("weight_decay", 0.0)),
        device=device,
        precision=str(training_cfg.get("precision", "fp32")),
        ema_decay=float(training_cfg.get("ema_decay", 0.999)),
        max_grad_norm=(
            None if training_cfg.get("max_grad_norm") is None else float(training_cfg["max_grad_norm"])
        ),
        train_log_path=train_log_path,
        resume=bool(getattr(args, "resume", False)),
        validation_interval=int(training_cfg.get("validation_interval", 100)),
        checkpoint_interval=int(
            training_cfg.get("checkpoint_interval", training_cfg.get("validation_interval", 100))
        ),
        rolling_loss_window=int(training_cfg.get("rolling_loss_window", 100)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        target_dataset=bundle.target_train,
        auxiliary_dataset=bundle.auxiliary_train,
        training_protocol=training_protocol,
        target_batch_size=int(training_cfg.get("target_batch_size", batch_size)),
        auxiliary_batch_size=int(training_cfg.get("auxiliary_batch_size", batch_size)),
        auxiliary_loss_weight=float(training_cfg.get("auxiliary_loss_weight", 1.0)),
        model_initialization_seed=model_initialization_seed,
        training_seed=training_seed,
        validation_corruption_bank=validation_bank,
        last_checkpoint_path=last_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        config_hash=computed_config_hash,
        manifest_hash=bundle.manifest_hash,
        git_sha=git_sha,
        checkpoint_provenance={
            "raw_config": resolved_info.raw,
            "resolved_config": cfg,
            "raw_config_hash": resolved_info.raw_hash,
            "resolved_config_hash": resolved_info.resolved_hash,
            "model_config_hash": resolved_model_hash,
            "study_plan_hash": resolved_info.study_plan_hash,
            "target_set_hash": resolved_info.target_set_hash,
            "environment_lock_hash": resolved_info.environment_lock_hash,
            "environment_runtime_hash": environment_runtime_hash,
            "environment_report": environment_report,
            "readiness_gate": readiness_gate,
            **readiness_fields,
            "split_manifest": dict(getattr(bundle, "split_manifest", {}) or {}),
            "subset_manifest": dict(getattr(bundle, "subset_manifest", {}) or {}),
            "split_manifest_hash": split_manifest_hash,
            "subset_manifest_hash": subset_manifest_hash,
            "target_training_subset_hash": target_subset_hash,
            "run_id": run_id,
            "job": job_fields,
        },
        deterministic_cpu=bool(training_cfg.get("deterministic_cpu", device.type == "cpu")),
    )

    selected_checkpoint_path = best_checkpoint_path if best_checkpoint_path.exists() else last_checkpoint_path
    load_checkpoint(selected_checkpoint_path, model, map_location=device, use_ema=True)
    selected_validation = evaluate_corruption_bank(
        model,
        training_diffusion,
        bundle.target_val,
        validation_bank,
        device,
        label=0 if conditional else None,
        batch_size=int(evaluation_cfg.get("denoising_batch_size", batch_size)),
        metric_prefix="validation",
    )
    selected_validation_metrics = _bank_metrics(selected_validation, split="validation")
    train_diagnostic_metrics = _bank_metrics(
        evaluate_corruption_bank(
            model,
            training_diffusion,
            bundle.target_train_eval,
            train_bank,
            device,
            label=0 if conditional else None,
            batch_size=int(evaluation_cfg.get("denoising_batch_size", batch_size)),
            metric_prefix="train",
        ),
        split="train",
    )
    train_metrics.update(selected_validation_metrics)
    train_metrics.update(train_diagnostic_metrics)

    evaluation_start = time.time()
    sampling_process = _sampling_process(
        sampler=sampler,
        timesteps=timesteps,
        schedule=schedule,
        device=device,
        ddim_eta=ddim_eta,
    )
    sampling_batch_size = int(evaluation_cfg.get("sampling_batch_size", sampling_cfg.get("batch_size", batch_size)))
    samples = sample_batched(
        sampling_process,
        model,
        num_samples=num_generated,
        image_size=image_size,
        conditional=conditional,
        label=0,
        sampling_steps=sampling_steps,
        batch_size=sampling_batch_size,
        sampling_seed=sampling_seed,
        ddim_eta=ddim_eta,
    )
    _atomic_torch_save(samples, sample_path)

    final_denoising = _bank_metrics(
        evaluate_corruption_bank(
            model,
            training_diffusion,
            bundle.target_eval,
            test_bank,
            device,
            label=0 if conditional else None,
            batch_size=int(evaluation_cfg.get("denoising_batch_size", batch_size)),
            metric_prefix="test",
        ),
        split="test",
    )

    real_eval_limit = evaluation_cfg.get("real_eval_max")
    real = _collect_dataset(
        bundle.target_eval,
        limit=None if real_eval_limit is None else int(real_eval_limit),
        batch_size=int(evaluation_cfg.get("feature_batch_size", batch_size)),
    )
    feature_metrics = compute_feature_metrics(
        samples,
        real,
        mode=mode,
        real_manifest_hash=split_manifest_hash,
        cache_dir=results_root / "cache" / "real_features",
        feature_dimension=int(evaluation_cfg.get("feature_dimension", 2048)),
        image_input_range=tuple(evaluation_cfg.get("image_input_range", [-1.0, 1.0])),
        compute_fid=bool(evaluation_cfg.get("compute_fid", evaluation_cfg.get("compute_fid_kid", True))),
        compute_kid=bool(evaluation_cfg.get("compute_kid", evaluation_cfg.get("compute_fid_kid", True))),
        compute_prdc_metrics=bool(evaluation_cfg.get("compute_prdc", False)),
        compute_inception_score=bool(evaluation_cfg.get("compute_inception_score", False)),
        fid_reliable_min_real=int(evaluation_cfg.get("fid_reliable_min_real", 1000)),
        kid_subset_size=int(evaluation_cfg.get("kid_subset_size", 100)),
        kid_num_subsets=int(evaluation_cfg.get("kid_num_subsets", 100)),
        prdc_k=int(evaluation_cfg.get("prdc_k", 5)),
        feature_batch_size=int(evaluation_cfg.get("feature_batch_size", evaluation_cfg.get("fid_batch_size", 64))),
        distance_batch_size=int(evaluation_cfg.get("distance_batch_size", 512)),
        evaluation_seed=evaluation_seed,
        device=device,
    )

    if classifier_enabled:
        classifier_metrics = evaluate_classifier_fidelity(
            samples,
            target_synset,
            bundle.aux_synsets,
            dataset_name=str(cfg.get("dataset", "imagenet")),
            device=device,
            batch_size=int(evaluation_cfg.get("classifier_batch_size", batch_size)),
            synset_to_index=classifier_mapping,
            strict=mode == "strict",
        )
    else:
        classifier_metrics = _disabled_classifier_metrics(evaluation_cfg)

    shared_diagnostic_extractor = (
        build_feature_extractor(device, strict=mode == "strict")
        if compute_similarity or compute_neighbors
        else None
    )
    if compute_similarity and bundle.aux_eval_datasets:
        average_similarity = average_auxiliary_similarity(
            bundle.target_eval,
            bundle.aux_eval_datasets,
            batch_size=int(evaluation_cfg.get("feature_batch_size", batch_size)),
            device=device,
            extractor=shared_diagnostic_extractor,
            strict=mode == "strict",
        )
    else:
        average_similarity = float("nan")

    memorization_metrics: dict[str, Any] = {}
    nearest_neighbor_grid_path = ""
    if compute_neighbors:
        reference_limit_value = evaluation_cfg.get("nearest_neighbor_reference_max")
        reference_limit = None if reference_limit_value is None else int(reference_limit_value)
        auxiliary_train_eval = (
            ConcatDataset(list(bundle.auxiliary_train_eval_by_class.values()))
            if bundle.auxiliary_train_eval_by_class
            else None
        )
        auxiliary_eval = (
            ConcatDataset(list(bundle.auxiliary_eval_by_class.values()))
            if bundle.auxiliary_eval_by_class
            else None
        )
        reference_sets = {
            "target_train": _collect_dataset(
                bundle.target_train_eval,
                limit=reference_limit,
                batch_size=int(evaluation_cfg.get("feature_batch_size", batch_size)),
            ),
            "target_eval": real,
            "auxiliary_train": _collect_dataset(
                auxiliary_train_eval,
                limit=reference_limit,
                batch_size=int(evaluation_cfg.get("feature_batch_size", batch_size)),
            ),
            "auxiliary_eval": _collect_dataset(
                auxiliary_eval,
                limit=reference_limit,
                batch_size=int(evaluation_cfg.get("feature_batch_size", batch_size)),
            ),
        }
        calibration_quantile = evaluation_cfg.get("near_duplicate_calibration_quantile")
        if calibration_quantile is not None or mode == "strict":
            calibration_limit_value = evaluation_cfg.get("near_duplicate_calibration_max_images")
            calibration_images = _collect_dataset(
                bundle.target_val,
                limit=None if calibration_limit_value is None else int(calibration_limit_value),
                batch_size=int(evaluation_cfg.get("feature_batch_size", batch_size)),
            )
            calibration_metrics = calibrate_near_duplicate_threshold(
                calibration_images,
                quantile=float(calibration_quantile if calibration_quantile is not None else 0.01),
                device=device,
                feature_batch_size=int(evaluation_cfg.get("feature_batch_size", batch_size)),
                distance_batch_size=int(evaluation_cfg.get("distance_batch_size", 256)),
                reference_batch_size=int(evaluation_cfg.get("reference_batch_size", 1024)),
                extractor=shared_diagnostic_extractor,
                strict_feature_extractor=mode == "strict",
            )
            near_duplicate_threshold = float(calibration_metrics["near_duplicate_threshold"])
        else:
            configured_threshold = evaluation_cfg.get("near_duplicate_threshold")
            near_duplicate_threshold = (
                None if configured_threshold is None else float(configured_threshold)
            )
            calibration_metrics = {
                "near_duplicate_threshold": (
                    "disabled" if near_duplicate_threshold is None else near_duplicate_threshold
                ),
                "near_duplicate_calibration_method": (
                    "disabled" if near_duplicate_threshold is None else "debug_fixed_threshold"
                ),
                "near_duplicate_calibration_split": "none",
                "near_duplicate_calibration_quantile": float("nan"),
                "near_duplicate_calibration_num_images": 0,
            }
        memorization_metrics = compute_memorization_diagnostics(
            samples,
            reference_sets,
            near_duplicate_threshold=near_duplicate_threshold,
            device=device,
            feature_batch_size=int(evaluation_cfg.get("feature_batch_size", batch_size)),
            distance_batch_size=int(evaluation_cfg.get("distance_batch_size", 256)),
            reference_batch_size=int(evaluation_cfg.get("reference_batch_size", 1024)),
            extractor=shared_diagnostic_extractor,
            strict_feature_extractor=mode == "strict",
        )
        memorization_metrics.update(calibration_metrics)
        nearest_neighbor_grid_path = str(outdir / "figures" / f"{run_id}_nearest_neighbors.png")
        fixed_indices = evaluation_cfg.get("nearest_neighbor_generated_indices")
        make_memorization_grid(
            samples,
            reference_sets,
            nearest_neighbor_grid_path,
            generated_indices=fixed_indices,
            max_items=int(evaluation_cfg.get("nearest_neighbor_grid_items", 8)),
            device=device,
            feature_batch_size=int(evaluation_cfg.get("feature_batch_size", batch_size)),
            distance_batch_size=int(evaluation_cfg.get("distance_batch_size", 256)),
            reference_batch_size=int(evaluation_cfg.get("reference_batch_size", 1024)),
            extractor=shared_diagnostic_extractor,
            strict_feature_extractor=mode == "strict",
        )

    train_mse = train_diagnostic_metrics.get("train_epsilon_mse_target")
    validation_mse = selected_validation_metrics.get("validation_epsilon_mse_target")
    test_mse = final_denoising.get("test_epsilon_mse_target")
    metrics: dict[str, Any] = {
        **feature_metrics,
        **classifier_metrics,
        **train_diagnostic_metrics,
        **selected_validation_metrics,
        **final_denoising,
        **memorization_metrics,
        "train_validation_mse_gap": _mse_gap(validation_mse, train_mse),
        "train_test_mse_gap": _mse_gap(test_mse, train_mse),
        "mse_gap_convention": "holdout_minus_train",
        "average_auxiliary_similarity": average_similarity,
        "num_generated": int(samples.shape[0]),
        "num_real_eval": int(real.shape[0]),
        "wallclock_eval_seconds": time.time() - evaluation_start,
    }
    metadata: dict[str, Any] = {
        "manifest_hash": bundle.manifest_hash,
        "split_manifest_hash": split_manifest_hash,
        "subset_manifest_hash": subset_manifest_hash,
        "manifest_path": bundle.manifest_path,
        "split_manifest_path": str(getattr(bundle, "split_manifest_path", "")),
        "subset_manifest_path": str(getattr(bundle, "subset_manifest_path", bundle.manifest_path)),
        "manifest_schema_version": (
            getattr(bundle, "split_manifest", {}).get("schema_version")
            or bundle.manifest.get("schema_version")
        ),
        "dataset_fingerprint": (
            getattr(bundle, "split_manifest", {}).get("dataset_fingerprint")
            or bundle.manifest.get("dataset_fingerprint")
        ),
        "target_training_subset_hash": target_subset_hash,
        "paired_target_prefix_hash": str(getattr(bundle, "paired_target_prefix_hash", target_subset_hash)),
        "target_eval_indices_hash": str(
            getattr(bundle, "target_eval_indices_hash", "")
            or canonical_sha256(bundle.manifest["target"]["eval"])
        ),
        "target_validation_indices_hash": str(
            getattr(bundle, "target_validation_indices_hash", "")
            or canonical_sha256(bundle.manifest["target"]["validation"])
        ),
        "auxiliary_training_subset_hashes": dict(
            getattr(bundle, "auxiliary_training_subset_hashes", {})
        ),
        "auxiliary_training_subset_hashes_json": json.dumps(
            getattr(bundle, "auxiliary_training_subset_hashes", {}), sort_keys=True
        ),
        "total_train_images": bundle.total_train_images,
        "num_target_available_after_holdouts": bundle.num_target_available,
        "equal_total_feasibility_json": json.dumps(bundle.feasibility, sort_keys=True),
        "aux_synsets": json.dumps(bundle.aux_synsets),
        "image_size": image_size,
        "training_steps": training_steps,
        "sampler": sampler,
        "sampling_steps": sampling_steps,
        "ddim_eta": ddim_eta,
        "sampling_seed": sampling_seed,
        "sampling_batch_size": sampling_batch_size,
        "evaluation_seed": evaluation_seed,
        "holdout_seed": holdout_seed,
        "training_subset_seed": training_subset_seed,
        "raw_config_hash": resolved_info.raw_hash,
        "resolved_config_hash": resolved_info.resolved_hash,
        "study_plan_hash": resolved_info.study_plan_hash,
        "target_set_hash": resolved_info.target_set_hash,
        "environment_lock_hash": resolved_info.environment_lock_hash,
        "environment_runtime_hash": environment_runtime_hash,
        "environment_report": environment_report,
        "readiness_gate": readiness_gate,
        **readiness_fields,
        "split_manifest_key": split_manifest_key,
        "subset_manifest_key": subset_manifest_key,
        "resolved_run_spec_hash": resolved_run_spec_hash,
        "config_hash": computed_config_hash,
        "config_path": str(args.config),
        "git_sha": git_sha,
        "selected_checkpoint_path": str(selected_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "validation_corruption_bank_path": str(validation_bank_path),
        "test_corruption_bank_path": str(test_bank_path),
        "train_corruption_bank_path": str(train_bank_path),
        "corruption_bank_config": {
            "validation_corruptions_per_image": validation_corruptions_per_image,
            "test_corruptions_per_image": test_corruptions_per_image,
            "train_diagnostic_corruptions_per_image": train_corruptions_per_image,
            "train_diagnostic_max_images": train_diagnostic_max_images,
            "noise_bins": noise_bins,
        },
        "sample_path": str(sample_path),
        "run_config_path": str(run_config_path),
        "nearest_neighbor_grid_path": nearest_neighbor_grid_path,
        "diagnostic_feature_extractor": (
            getattr(shared_diagnostic_extractor, "extractor_name", type(shared_diagnostic_extractor).__name__)
            if shared_diagnostic_extractor is not None
            else "disabled"
        ),
        "diagnostic_feature_weights": (
            getattr(shared_diagnostic_extractor, "weights_name", "unknown")
            if shared_diagnostic_extractor is not None
            else "disabled"
        ),
        "diagnostic_feature_preprocessing": (
            getattr(shared_diagnostic_extractor, "preprocessing_name", "unknown")
            if shared_diagnostic_extractor is not None
            else "disabled"
        ),
        "diagnostic_torchvision_version": _package_version("torchvision"),
        "architecture": train_metrics.get("architecture"),
        "architecture_profile": train_metrics.get("architecture_profile"),
        "model_parameter_count": train_metrics.get("model_parameter_count"),
        "backbone_parameter_count": train_metrics.get("backbone_parameter_count"),
        "conditioning_parameter_count": train_metrics.get("conditioning_parameter_count"),
        "model_config_hash": train_metrics.get("model_config_hash"),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "wallclock_total_seconds": time.time() - run_started,
    }
    record = {
        "status": "completed",
        "job": job_fields,
        "metadata": metadata,
        "training": train_metrics,
        "metrics": metrics,
    }
    write_run_result(results_root, run_id, record)
    failure_path.unlink(missing_ok=True)
    print(result_path)
    return {"run_id": run_id, **record}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", choices=["A", "B", "C"])
    parser.add_argument("--model-type")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--n0", type=int)
    parser.add_argument("--m-per-aux", type=int, dest="m_per_aux")
    parser.add_argument("--K-aux", type=int, dest="K_aux")
    parser.add_argument("--num-generated", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-split-seed", type=int, dest="data_split_seed")
    parser.add_argument("--holdout-seed", type=int, dest="holdout_seed")
    parser.add_argument("--training-subset-seed", type=int, dest="training_subset_seed")
    parser.add_argument("--model-initialization-seed", type=int, dest="model_initialization_seed")
    parser.add_argument("--training-seed", type=int, dest="training_seed")
    parser.add_argument("--sampling-seed", type=int, dest="sampling_seed")
    parser.add_argument("--evaluation-seed", type=int, dest="evaluation_seed")
    parser.add_argument("--training-protocol", choices=["natural_compute_matched", "target_exposure_matched"])
    parser.add_argument("--image-size", type=int, dest="image_size")
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true")
    restart.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--override-readiness-gate", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
