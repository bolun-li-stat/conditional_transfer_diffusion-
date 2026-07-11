"""Validate a study configuration before submitting training jobs."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from image_transfer.config import load_resolved_config
from image_transfer.evaluation.classifier_fidelity import imagenet_synset_to_index
from image_transfer.models import build_image_model, model_parameter_metadata
from image_transfer.readiness import enforce_readiness_gate
from image_transfer.scripts.inspect_environment import inspect_environment
from image_transfer.scripts.make_job_grid import job_breakdown, rows_for_experiment
from image_transfer.scripts.prepare_metric_assets import initialize_backends_offline, verify_manifest
from image_transfer.utils.io import atomic_write_json, resolve_env_path


def _check(name: str, ok: bool, details: Any, *, required: bool = True) -> dict[str, Any]:
    return {"name": name, "status": "pass" if ok else "fail", "required": required, "details": details}


def _writeable_directory(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path):
            pass
        return True, str(path)
    except OSError as exception:
        return False, f"{path}: {exception}"


def _image_count(path: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions)


def _configured_class_counts(cfg: dict[str, Any], data_root: Path) -> tuple[dict[str, dict[str, int]], list[str]]:
    if cfg.get("use_fake_data", False):
        count = int(cfg.get("fake_data_size", 0))
        labels = {
            str(label)
            for target in cfg.get("targets", [])
            for label in [target["synset"], *sum(target.get("auxiliary_sets", {}).values(), [])]
        }
        return {
            label: {"train": count, "eval": count}
            for label in labels
        }, []
    dataset = str(cfg.get("dataset", "")).lower()
    if dataset.startswith("cifar"):
        try:
            from image_transfer.data.cifar import class_id, load_cifar10

            train = load_cifar10(data_root, int(cfg.get("image_size", 32)), train=True, download=False)
            test = load_cifar10(data_root, int(cfg.get("image_size", 32)), train=False, download=False)
            labels = {
                str(label)
                for target in cfg.get("targets", [])
                for label in [target["synset"], *sum(target.get("auxiliary_sets", {}).values(), [])]
            }
            return {
                label: {
                    "train": sum(int(value) == class_id(label) for value in train.targets),
                    "eval": sum(int(value) == class_id(label) for value in test.targets),
                }
                for label in labels
            }, []
        except Exception as exception:
            return {}, [f"cifar_count:{type(exception).__name__}:{exception}"]
    labels = {
        str(label)
        for target in cfg.get("targets", [])
        for label in [target["synset"], *sum(target.get("auxiliary_sets", {}).values(), [])]
    }
    source = str(cfg.get("data_split", {}).get("eval_source", "train_holdout"))
    counts: dict[str, dict[str, int]] = {}
    failures: list[str] = []
    for label in sorted(labels):
        train_dir = data_root / "train" / label
        eval_dir = train_dir if source == "train_holdout" else data_root / "val" / label
        if not train_dir.is_dir():
            failures.append(f"missing_train_class:{label}")
        if not eval_dir.is_dir():
            failures.append(f"missing_eval_class:{label}")
        counts[label] = {
            "train": _image_count(train_dir) if train_dir.is_dir() else 0,
            "eval": _image_count(eval_dir) if eval_dir.is_dir() else 0,
        }
    return counts, failures


def run_preflight(
    config_path: str | Path,
    *,
    out_dir: str | Path,
    allow_readiness_override: bool = False,
) -> dict[str, Any]:
    resolved = load_resolved_config(config_path)
    cfg = resolved.resolved
    checks: list[dict[str, Any]] = []
    lock_path = cfg.get("environment_lock_path", "environment/requirements-image-lock.txt")
    lock = Path(lock_path)
    if not lock.is_absolute():
        candidate = (Path(config_path).resolve().parent / lock).resolve()
        lock = candidate if candidate.exists() else Path(lock_path).resolve()
    environment = inspect_environment(lock)
    environment_ok = (
        environment["environment_lock_hash"] != "missing"
        and bool(environment.get("lock_matches_runtime", False))
    )
    checks.append(_check("environment_lock", environment_ok, environment))
    checks.append(_check("git_clean", not environment["git_dirty"], {"git_sha": environment["git_sha"], "dirty": environment["git_dirty"]}, required=False))

    requires_cuda = (
        str(cfg.get("training", {}).get("precision", "fp32")).lower() == "amp"
        or "pytorch-cuda" in environment.get("lock_expected_versions", {})
        or str(cfg.get("device", "")).lower().startswith("cuda")
    )
    accelerator_ok = not requires_cuda or bool(environment.get("cuda_available", False))
    checks.append(
        _check(
            "accelerator",
            accelerator_ok,
            {
                "cuda_required": requires_cuda,
                "cuda_available": bool(environment.get("cuda_available", False)),
                "gpu_names": environment.get("gpu_names", []),
                "driver_version": environment.get("driver_version"),
                "cuda_build": environment.get("torch_cuda_build"),
                "cudnn_version": environment.get("cudnn_version"),
            },
        )
    )
    checks.append(
        _check(
            "amp",
            not requires_cuda or (
                torch.cuda.is_available()
                and environment.get("torch_cuda_build") not in {None, ""}
                and environment.get("cudnn_version") is not None
            ),
            {"precision": cfg.get("training", {}).get("precision", "fp32")},
        )
    )

    try:
        device = torch.device("cuda" if requires_cuda and torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = build_image_model(
            cfg["model"], image_size=int(cfg["image_size"]), conditional=True, num_classes=2, model_seed=0
        ).to(device)
        size = int(cfg["image_size"])
        x = torch.randn(1, 3, size, size, device=device)
        output = model(
            x,
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.long, device=device),
        )
        output.mean().backward()
        model_details = model_parameter_metadata(model)
        model_details["device"] = str(device)
        model_details["peak_gpu_memory_bytes"] = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        model_ok = output.shape == x.shape and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()
        )
    except Exception as exception:
        model_ok, model_details = False, f"{type(exception).__name__}: {exception}"
    checks.append(_check("model_forward_backward", model_ok, model_details))

    diffusion_steps = int(cfg.get("diffusion", {}).get("timesteps", 1000))
    sample_steps = int(cfg.get("sampling", {}).get("steps", diffusion_steps))
    sampler = str(cfg.get("sampling", {}).get("sampler", "ddpm"))
    sampler_ok = (sampler == "ddpm" and sample_steps == diffusion_steps) or (
        sampler == "ddim" and 2 <= sample_steps <= diffusion_steps
    )
    checks.append(_check("sampler", sampler_ok, {"sampler": sampler, "steps": sample_steps, "horizon": diffusion_steps}))

    mode = cfg.get("evaluation", {}).get("mode")
    asset_manifest = cfg.get("metric_assets_manifest_path", "")
    asset_ok, asset_details = mode != "strict", "not required in debug mode"
    if mode == "strict":
        try:
            expanded_manifest = resolve_env_path(None if asset_manifest is None else str(asset_manifest))
            if not expanded_manifest or "$" in expanded_manifest:
                raise ValueError(
                    "metric_assets_manifest_path must resolve to an explicit manifest file; "
                    "set METRIC_ASSETS_MANIFEST for the checked-in GPU configurations"
                )
            manifest_path = Path(expanded_manifest).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = (Path(config_path).resolve().parent / manifest_path).resolve()
            manifest = verify_manifest(manifest_path.parent, manifest_path, verify_runtime=True)
            initialized = initialize_backends_offline(manifest_path.parent, device="cpu")
            asset_ok, asset_details = True, {"manifest": manifest, "offline_initialization": initialized}
        except Exception as exception:
            asset_ok, asset_details = False, f"{type(exception).__name__}: {exception}"
    checks.append(_check("offline_metric_assets", asset_ok, asset_details))

    expanded_data_root = str(resolve_env_path(cfg.get("data_root"), "") or "")
    data_root = Path(expanded_data_root).expanduser() if expanded_data_root else Path(".")
    data_ok = bool(cfg.get("use_fake_data")) or (
        bool(expanded_data_root)
        and "$" not in expanded_data_root
        and data_root.is_dir()
        and data_root.resolve() != Path.cwd().resolve()
    )
    checks.append(_check("dataset_root", data_ok, expanded_data_root or "not configured"))
    expanded_output_root = str(resolve_env_path(cfg.get("output_root"), "image_transfer_results") or "")
    output_root = Path(expanded_output_root) if expanded_output_root else Path(".")
    try:
        if not expanded_output_root or "$" in expanded_output_root:
            raise ValueError("output_root must resolve to an explicit path")
        output_ok, output_details = _writeable_directory(output_root)
        if not output_ok:
            raise OSError(output_details)
    except (OSError, ValueError) as exception:
        output_ok, output_root = False, Path(str(exception))
    checks.append(_check("output_writeable", output_ok, str(output_root)))
    disk = shutil.disk_usage(output_root if output_ok else Path.cwd())
    checks.append(_check("free_disk", disk.free > 1024**3, {"free_bytes": disk.free}, required=False))

    try:
        gate = enforce_readiness_gate(
            cfg,
            override=allow_readiness_override,
            config_source_path=resolved.source_path or config_path,
        )
        gate_ok = True
    except Exception as exception:
        gate, gate_ok = f"{type(exception).__name__}: {exception}", False
    checks.append(_check("readiness_gate", gate_ok, gate))

    jobs = []
    try:
        if not gate_ok:
            raise RuntimeError("job grid is blocked by the readiness gate")
        for experiment, experiment_cfg in cfg.get("experiments", {}).items():
            if experiment_cfg.get("enabled", False):
                jobs.extend(
                    rows_for_experiment(
                        str(experiment),
                        resolved.raw,
                        str(config_path),
                        resolved_info=resolved,
                        override_readiness_gate=allow_readiness_override,
                        readiness_gate=gate,
                    )
                )
        jobs_ok = bool(jobs) and len(jobs) == len({row["run_id"] for row in jobs})
        jobs_details = {
            "count": len(jobs),
            "unique_run_ids": len({row["run_id"] for row in jobs}),
            **job_breakdown(jobs, cfg),
        }
    except Exception as exception:
        jobs_ok, jobs_details = False, f"{type(exception).__name__}: {exception}"
    checks.append(_check("job_grid", jobs_ok, jobs_details))

    class_counts, count_failures = _configured_class_counts(cfg, data_root) if data_ok else ({}, ["dataset_root"])
    checks.append(_check("class_counts", not count_failures, {"counts": class_counts, "failures": count_failures}))
    split = cfg.get("data_split", {})
    source = str(split.get("eval_source", "train_holdout"))
    feasibility_failures: list[str] = []
    for row in jobs:
        target = str(row["target_synset"])
        target_counts = class_counts.get(target, {"train": 0, "eval": 0})
        target_reserve = (
            int(split.get("target_eval_size", 0)) + int(split.get("target_val_size", 0))
            if source == "train_holdout"
            else 0
        )
        is_baseline = str(row["model_type"]).startswith(("conditional_target_only", "unconditional"))
        needed_target = int(row.get("baseline_target_count", row["n0"])) if is_baseline else int(row["n0"])
        if int(target_counts.get("train", 0)) - target_reserve < needed_target:
            feasibility_failures.append(f"{row['run_id']}:target")
        if source != "train_holdout" and int(target_counts.get("eval", 0)) < int(
            split.get("target_eval_size", 0)
        ) + int(split.get("target_val_size", 0)):
            feasibility_failures.append(f"{row['run_id']}:target_holdout")
        try:
            auxiliary = json.loads(str(row.get("aux_composition", "[]")))
        except json.JSONDecodeError:
            auxiliary = []
            feasibility_failures.append(f"{row['run_id']}:aux_composition")
        for label in auxiliary:
            counts = class_counts.get(str(label), {"train": 0, "eval": 0})
            auxiliary_reserve = int(split.get("auxiliary_eval_size", 0)) if source == "train_holdout" else 0
            if int(counts.get("train", 0)) - auxiliary_reserve < int(row["m_per_aux"]):
                feasibility_failures.append(f"{row['run_id']}:auxiliary:{label}")
            if source != "train_holdout" and int(counts.get("eval", 0)) < int(
                split.get("auxiliary_eval_size", 0)
            ):
                feasibility_failures.append(f"{row['run_id']}:auxiliary_holdout:{label}")
    checks.append(
        _check(
            "job_feasibility",
            bool(jobs) and not feasibility_failures,
            {"checked_jobs": len(jobs), "failures": feasibility_failures[:100]},
        )
    )

    mapping_failures: list[str] = []
    if bool(cfg.get("evaluation", {}).get("compute_classifier_fidelity", False)):
        labels: set[str] = set()
        for row in jobs:
            labels.add(str(row["target_synset"]))
            try:
                labels.update(map(str, json.loads(str(row.get("aux_composition", "[]")))))
            except json.JSONDecodeError:
                mapping_failures.append(f"invalid_aux_composition:{row['run_id']}")
        mapping_failures.extend(sorted(label for label in labels if imagenet_synset_to_index(label) is None))
    checks.append(_check("classifier_mapping", not mapping_failures, {"missing_synsets": mapping_failures}))

    collisions = [
        str(output_root / "run_results" / f"{row['run_id']}.json")
        for row in jobs
        if (output_root / "run_results" / f"{row['run_id']}.json").exists()
    ]
    checks.append(_check("output_collision", not collisions, {"existing_results": collisions[:100]}))
    manifest_root = Path(str(resolve_env_path(split.get("manifest_root"), str(output_root / "manifests"))))
    manifest_ok, manifest_details = _writeable_directory(manifest_root)
    checks.append(_check("manifest_writeable", manifest_ok, manifest_details))
    checkpoint_probe = output_root / "preflight_checkpoint_write_test"
    checkpoint_ok, checkpoint_details = _writeable_directory(checkpoint_probe)
    checks.append(_check("checkpoint_writeable", checkpoint_ok, checkpoint_details))
    if checkpoint_ok:
        try:
            checkpoint_probe.rmdir()
        except OSError:
            pass
    storage = job_breakdown(jobs, cfg) if jobs else {"estimated_jobs": 0, "estimated_sample_storage_bytes": 0}
    storage_ok = disk.free > int(storage.get("estimated_sample_storage_bytes", 0)) + 1024**3
    checks.append(_check("estimated_storage", storage_ok, {**storage, "free_bytes": disk.free}))
    strict_no_fallback = mode != "strict" or asset_ok
    checks.append(_check("strict_no_fallback", strict_no_fallback, {"mode": mode, "assets_verified": asset_ok}))

    parameter_counts: dict[str, int] = {}
    if isinstance(model_details, dict):
        backbone = int(model_details.get("backbone_parameter_count", 0))
        two_class_conditioning = int(model_details.get("conditioning_parameter_count", 0))
        per_class = two_class_conditioning // 2 if two_class_conditioning else 0
        for row in jobs:
            model_type = str(row["model_type"])
            if model_type.startswith("unconditional"):
                classes = 0
            elif model_type.startswith("conditional_target_only"):
                classes = 1
            else:
                classes = 1 + len(json.loads(str(row.get("aux_composition", "[]"))))
            parameter_counts[f"{model_type}:classes={classes}"] = backbone + classes * per_class
    checks.append(_check("exact_parameter_counts", bool(parameter_counts), parameter_counts))

    required_failures = [item for item in checks if item["required"] and item["status"] == "fail"]
    report = {
        "schema_version": "2.0",
        "status": "ready" if not required_failures else "not_ready",
        "config_path": str(config_path),
        "config_provenance": resolved.provenance(),
        "checks": checks,
        "required_failure_count": len(required_failures),
    }
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, destination / "preflight_report.json")
    lines = [f"# Preflight: {report['status']}", ""]
    lines.extend(f"- {item['status'].upper()}: {item['name']} — {item['details']}" for item in checks)
    (destination / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="readiness")
    parser.add_argument("--override-readiness-gate", action="store_true")
    args = parser.parse_args()
    report = run_preflight(
        args.config,
        out_dir=args.out_dir,
        allow_readiness_override=args.override_readiness_gate,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
