"""Train, sample, and evaluate one publication-grade image-transfer job."""

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

from image_transfer.data import (
    ManifestInsufficientDataError,
    build_datasets_for_job,
    canonical_sha256,
    config_hash as canonical_config_hash,
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
    compute_memorization_diagnostics,
    make_memorization_grid,
)
from image_transfer.scripts.make_job_grid import EXP_DIR
from image_transfer.training.checkpointing import load_checkpoint
from image_transfer.training.trainer import train_image_model
from image_transfer.utils.device import get_device
from image_transfer.utils.io import (
    canonical_json_hash,
    ensure_dir,
    failure_result_path,
    get_git_sha,
    load_valid_result,
    load_yaml,
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


def _direct_run_id(fields: Mapping[str, Any]) -> str:
    parts = [
        fields["experiment"], fields["target_synset"], fields["model_type"], fields["aux_set"],
        f"draw{fields['aux_draw_id']}", f"n0{fields['n0']}", f"m{fields['m_per_aux']}",
        f"k{fields['K_aux']}", f"ds{fields['data_split_seed']}",
        f"mi{fields['model_initialization_seed']}", f"tr{fields['training_seed']}",
        fields["training_protocol"], fields["sampler"], f"ss{fields['sampling_seed']}",
        f"ev{fields['evaluation_seed']}", f"spec{str(fields['effective_run_spec_hash'])[:12]}",
        str(fields["config_hash"])[:12],
    ]
    safe = ["".join(character if character.isalnum() or character in "._-" else "-" for character in str(part)) for part in parts]
    return "_".join(safe)


def deterministic_run_id(job: Mapping[str, Any] | None = None, **fields: Any) -> str:
    if job and job.get("run_id"):
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
    if split != "validation" and corruptions is not None:
        result[f"num_{split}_corruptions"] = result.pop("num_corruptions")
    result[f"{split}_corruption_bank_hash"] = bank_hash
    result[f"{split}_corruption_bank_manifest_hash"] = manifest_hash
    result[f"{split}_corruption_timestep_distribution"] = distribution
    if split == "validation":
        # Compatibility field requested by the evaluation schema.
        result["corruption_bank_hash"] = bank_hash
    return result


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
) -> dict[str, Any]:
    record = {
        "status": "skipped",
        "job": dict(job_fields),
        "metadata": {
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
    if bool(getattr(args, "force", False)) and bool(getattr(args, "resume", False)):
        raise ValueError("--force and --resume are mutually exclusive")
    cfg = load_yaml(args.config)
    computed_config_hash = canonical_config_hash(cfg)
    recorded_config_hash = str((job or {}).get("config_hash") or computed_config_hash)
    if recorded_config_hash != computed_config_hash:
        raise ValueError("job grid config_hash does not match the current config file; regenerate the grid")

    experiment = str(_arg_or_job(args, job, "experiment", "A"))
    if experiment not in EXP_DIR:
        raise ValueError(f"Unknown experiment {experiment!r}")
    legacy_seed = int(_arg_or_job(args, job, "seed", 0))
    split_cfg = cfg.get("data_split", {})
    training_cfg = cfg.get("training", {})
    sampling_cfg = cfg.get("sampling", {})
    evaluation_cfg = cfg.get("evaluation", {})
    data_split_seed = int(_arg_or_job(args, job, "data_split_seed", split_cfg.get("data_split_seed", legacy_seed)))
    model_initialization_seed = int(_arg_or_job(args, job, "model_initialization_seed", legacy_seed))
    training_seed = int(_arg_or_job(args, job, "training_seed", legacy_seed))
    sampling_seed = int(_arg_or_job(args, job, "sampling_seed", sampling_cfg.get("seed", legacy_seed)))
    evaluation_seed = int(_arg_or_job(args, job, "evaluation_seed", evaluation_cfg.get("seed", legacy_seed)))

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
    sampler = str(_arg_or_job(args, job, "sampler", cfg.get("sampler", sampling_cfg.get("sampler", "ddpm")))).lower()
    image_size = int(_arg_or_job(args, job, "image_size", cfg.get("image_size", 32)))
    sampling_steps = int(sampling_cfg.get("steps", cfg.get("sampling_steps", cfg.get("diffusion", {}).get("timesteps", 1000))))
    ddim_eta = float(sampling_cfg.get("ddim_eta", cfg.get("ddim_eta", 0.0)))
    training_steps = int(
        getattr(args, "max_steps", None) if getattr(args, "max_steps", None) is not None else training_cfg.get("steps", 1000)
    )
    num_generated = int(getattr(args, "num_generated", None) or cfg.get("num_generated", 64))
    effective_run_spec_hash = canonical_json_hash(
        {
            "image_size": image_size,
            "training_steps": training_steps,
            "num_generated": num_generated,
            "sampling_steps": sampling_steps,
            "ddim_eta": ddim_eta,
        }
    )
    if job and job.get("effective_run_spec_hash") not in {None, "", effective_run_spec_hash}:
        raise ValueError("job grid effective_run_spec_hash does not match the resolved run settings")

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
        "seed": training_seed,
        "data_split_seed": data_split_seed,
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
        "effective_run_spec_hash": effective_run_spec_hash,
        "config_hash": computed_config_hash,
    }
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

    git_sha = get_git_sha()
    builder_job = dict(job or {})
    builder_job.update({
        "experiment": experiment,
        "target_synset": target_synset,
        "model_type": model_type,
        "data_split_seed": data_split_seed,
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
        )
        failure_path.unlink(missing_ok=True)
        print(f"skipped {run_id}: insufficient target images after fixed holdouts")
        return record

    ensure_dir(run_config_path.parent)
    with open(run_config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump({"config": cfg, "job": builder_job, "run_id": run_id}, handle, sort_keys=False)

    timesteps = int(cfg.get("diffusion", {}).get("timesteps", 1000))
    schedule = str(cfg.get("diffusion", {}).get("schedule", "linear"))
    corruption_bank_root = Path(
        resolve_env_path(evaluation_cfg.get("corruption_bank_root"), str(results_root / "corruption_banks"))
    )
    corruptions_per_image = int(evaluation_cfg.get("corruptions_per_image", 1))
    noise_bins = evaluation_cfg.get("noise_bins")
    validation_bank, validation_bank_path = _get_or_create_bank(
        corruption_bank_root,
        split="validation",
        manifest_hash=bundle.manifest_hash,
        evaluation_seed=evaluation_seed,
        timesteps=timesteps,
        corruptions_per_image=corruptions_per_image,
        num_images=len(bundle.target_val),
        noise_bins=noise_bins,
    )
    test_bank, test_bank_path = _get_or_create_bank(
        corruption_bank_root,
        split="test",
        manifest_hash=bundle.manifest_hash,
        evaluation_seed=evaluation_seed,
        timesteps=timesteps,
        corruptions_per_image=corruptions_per_image,
        num_images=len(bundle.target_eval),
        noise_bins=noise_bins,
    )

    device = get_device(_arg_or_job(args, job, "device", cfg.get("device", "auto")))
    batch_size = int(training_cfg.get("batch_size", 32))
    conditional = _is_conditional(model_type)
    mode = str(evaluation_cfg.get("mode", "debug" if cfg.get("use_fake_data", False) else "paper")).lower()
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

    # Paper runs fail on missing metric/classifier/diagnostic weights before
    # spending compute on training. Debug mode remains fully offline-capable.
    if mode == "paper":
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
        base_channels=int(cfg.get("model", {}).get("base_channels", 64)),
        channel_mults=cfg.get("model", {}).get("channel_mults", [1, 2, 2, 4]),
        timesteps=timesteps,
        schedule=schedule,
        steps=training_steps,
        batch_size=batch_size,
        lr=float(cfg.get("optimizer", {}).get("lr", 2e-4)),
        device=device,
        precision=str(training_cfg.get("precision", "fp32")),
        ema_decay=float(training_cfg.get("ema_decay", 0.999)),
        train_log_path=train_log_path,
        resume=bool(getattr(args, "resume", False)),
        validation_interval=int(training_cfg.get("validation_interval", 100)),
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
    train_metrics.update(_bank_metrics(selected_validation, split="validation"))

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
        real_manifest_hash=bundle.manifest_hash,
        cache_dir=results_root / "cache" / "real_features",
        feature_dimension=int(evaluation_cfg.get("feature_dimension", 2048)),
        image_input_range=tuple(evaluation_cfg.get("image_input_range", [-1.0, 1.0])),
        compute_fid=bool(evaluation_cfg.get("compute_fid", evaluation_cfg.get("compute_fid_kid", True))),
        compute_kid=bool(evaluation_cfg.get("compute_kid", evaluation_cfg.get("compute_fid_kid", True))),
        compute_prdc_metrics=bool(evaluation_cfg.get("compute_prdc", False)),
        compute_inception_score=bool(evaluation_cfg.get("compute_inception_score", False)),
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
            strict=mode == "paper",
        )
    else:
        classifier_metrics = _disabled_classifier_metrics(evaluation_cfg)

    shared_diagnostic_extractor = (
        build_feature_extractor(device, strict=mode == "paper")
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
            strict=mode == "paper",
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
        memorization_metrics = compute_memorization_diagnostics(
            samples,
            reference_sets,
            near_duplicate_threshold=evaluation_cfg.get("near_duplicate_threshold"),
            device=device,
            feature_batch_size=int(evaluation_cfg.get("feature_batch_size", batch_size)),
            distance_batch_size=int(evaluation_cfg.get("distance_batch_size", 256)),
            reference_batch_size=int(evaluation_cfg.get("reference_batch_size", 1024)),
            extractor=shared_diagnostic_extractor,
            strict_feature_extractor=mode == "paper",
        )
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
            strict_feature_extractor=mode == "paper",
        )

    target_subset_count = n0 + k_aux * m_per_aux if model_type in EQUAL_TOTAL_MODEL_TYPES else n0
    target_subset_hash = canonical_sha256(target_training_subset(bundle.manifest, target_subset_count))
    target_eval_hash = canonical_sha256(bundle.manifest["target"]["eval"])
    metrics: dict[str, Any] = {
        **feature_metrics,
        **classifier_metrics,
        **final_denoising,
        **memorization_metrics,
        "average_auxiliary_similarity": average_similarity,
        "num_generated": int(samples.shape[0]),
        "num_real_eval": int(real.shape[0]),
        "wallclock_eval_seconds": time.time() - evaluation_start,
    }
    metadata: dict[str, Any] = {
        "manifest_hash": bundle.manifest_hash,
        "manifest_path": bundle.manifest_path,
        "manifest_schema_version": bundle.manifest.get("schema_version"),
        "dataset_fingerprint": bundle.manifest.get("dataset_fingerprint"),
        "target_training_subset_hash": target_subset_hash,
        "target_eval_indices_hash": target_eval_hash,
        "target_validation_indices_hash": canonical_sha256(bundle.manifest["target"]["validation"]),
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
        "config_hash": computed_config_hash,
        "config_path": str(args.config),
        "git_sha": git_sha,
        "selected_checkpoint_path": str(selected_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "validation_corruption_bank_path": str(validation_bank_path),
        "test_corruption_bank_path": str(test_bank_path),
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
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
