from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader

from image_transfer.data import build_datasets_for_job, count_available_target_images
from image_transfer.evaluation.classifier_fidelity import evaluate_classifier_fidelity
from image_transfer.evaluation.fid_kid import compute_fid_kid
from image_transfer.evaluation.feature_similarity import average_auxiliary_similarity
from image_transfer.evaluation.nearest_neighbors import make_nearest_neighbor_grid
from image_transfer.scripts.make_job_grid import EXP_DIR
from image_transfer.training.trainer import train_image_model
from image_transfer.utils.device import get_device
from image_transfer.utils.io import append_csv_row, ensure_dir, load_yaml, resolve_env_path
from image_transfer.utils.seed import set_seed

FIELDS = [
    "dataset", "experiment", "experiment_name", "target_synset", "target_name", "aux_set", "aux_synsets",
    "aux_composition", "model_type", "n0", "m_per_aux", "K_aux", "total_train_images", "seed", "image_size",
    "training_steps", "checkpoint_path", "final_train_loss", "validation_epsilon_mse_target",
    "validation_epsilon_mse_low_noise", "validation_epsilon_mse_mid_noise", "validation_epsilon_mse_high_noise",
    "fid_target", "kid_target_mean", "kid_target_std", "classifier_target_top1_acc", "classifier_target_top5_acc",
    "auxiliary_leakage_rate", "top1_prediction_histogram_json", "average_auxiliary_similarity", "num_generated",
    "num_real_eval", "sampler", "sampling_steps", "wallclock_train_seconds", "wallclock_eval_seconds",
    "skipped_equal_total_baseline", "skip_reason",
]



def sample_batched(diffusion, model, *, num_samples: int, image_size: int, conditional: bool, label: int, sampling_steps: int, batch_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, num_samples, batch_size):
        current = min(batch_size, num_samples - start)
        labels = torch.full((current,), label, dtype=torch.long, device=diffusion.device) if conditional else None
        chunk = diffusion.sample(model, (current, 3, image_size, image_size), y=labels, steps=sampling_steps)
        chunks.append(chunk.cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, 3, image_size, image_size)

def read_job(path: str, index: int) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if index < 0 or index >= len(rows):
        raise IndexError(f"job-index {index} out of range for {len(rows)} jobs")
    return rows[index]


def _arg_or_job(args, job: dict | None, key: str, default=None):
    cli_value = getattr(args, key, None)
    if cli_value is not None:
        return cli_value
    if job and job.get(key) not in {None, ""}:
        return job[key]
    return default


def deterministic_run_id(job: dict | None, experiment: str, target_synset: str, model_type: str, aux_set: str, n0: int, m_per_aux: int, k_aux: int, seed: int) -> str:
    if job and job.get("run_id"):
        return job["run_id"]
    return f"{experiment}_{target_synset}_{model_type}_{aux_set}_n0{n0}_m{m_per_aux}_k{k_aux}_seed{seed}"


def _write_skip_row(cfg: dict, outdir: Path, row: dict) -> None:
    append_csv_row(outdir / "metrics.csv", row, FIELDS)
    append_csv_row(Path(resolve_env_path(cfg.get("output_root"), "image_transfer_results")) / "all_metrics.csv", row, FIELDS)


def run(args, job: dict | None = None) -> dict:
    cfg = load_yaml(args.config)
    experiment = str(_arg_or_job(args, job, "experiment", "A"))
    seed = int(_arg_or_job(args, job, "seed", 0))
    set_seed(seed)
    n0 = int(_arg_or_job(args, job, "n0", 100))
    m_per_aux = int(getattr(args, "m_per_aux", None) or (job.get("m_per_aux") if job else None) or n0)
    k_aux = int(getattr(args, "K_aux", None) or (job.get("K_aux") if job else None) or cfg.get("K_aux", 5))
    image_size = int(getattr(args, "image_size", None) or cfg.get("image_size", 32))
    model_type = str((job or {}).get("model_type") or getattr(args, "model_type", None) or "unconditional_n0")
    aux_set = str((job or {}).get("aux_set") or "none")
    target_synset = str((job or {}).get("target_synset") or cfg.get("targets", [{"synset": "dog"}])[0].get("synset", "dog"))
    target_name = str((job or {}).get("target_name") or target_synset)
    outdir = Path((job or {}).get("output_dir") or Path(resolve_env_path(cfg.get("output_root"), "image_transfer_results")) / EXP_DIR[experiment])
    run_id = deterministic_run_id(job, experiment, target_synset, model_type, aux_set, n0, m_per_aux, k_aux, seed)
    checkpoint_path = outdir / "checkpoints" / f"{run_id}.pt"
    log_path = outdir / "logs" / f"{run_id}_train_log.csv"
    run_config_path = outdir / "configs" / f"{run_id}.yaml"
    ensure_dir(outdir)

    training_steps = int(getattr(args, "max_steps", None) if getattr(args, "max_steps", None) is not None else cfg.get("training", {}).get("steps", 1000))
    row = {field: "" for field in FIELDS}
    row.update({
        "dataset": cfg.get("dataset", "cifar10"),
        "experiment": experiment,
        "experiment_name": {"A": "equal_target", "B": "equal_total", "C": "similarity_sweep"}[experiment],
        "target_synset": target_synset,
        "target_name": target_name,
        "aux_set": aux_set,
        "aux_synsets": (job or {}).get("aux_composition", "[]"),
        "aux_composition": aux_set,
        "model_type": model_type,
        "n0": n0,
        "m_per_aux": m_per_aux,
        "K_aux": k_aux,
        "seed": seed,
        "image_size": image_size,
        "training_steps": training_steps,
        "checkpoint_path": str(checkpoint_path),
        "num_generated": int(getattr(args, "num_generated", None) or cfg.get("num_generated", 64)),
        "sampler": cfg.get("sampler", "ddpm"),
        "sampling_steps": int(cfg.get("sampling_steps", cfg.get("diffusion", {}).get("timesteps", 1000))),
        "skipped_equal_total_baseline": False,
        "skip_reason": "",
    })

    if model_type == "unconditional_equal_total" and not cfg.get("use_fake_data", False):
        n_total = n0 + k_aux * m_per_aux
        available = count_available_target_images(cfg, target_synset)
        if available < n_total:
            row.update({
                "total_train_images": 0,
                "skipped_equal_total_baseline": True,
                "skip_reason": "insufficient_target_images",
            })
            _write_skip_row(cfg, outdir, row)
            print(f"skipped {run_id}: insufficient_target_images ({available} < {n_total})")
            return row

    if getattr(args, "dry_run", False):
        print(json.dumps({"run_id": run_id, "checkpoint_path": str(checkpoint_path), "output_dir": str(outdir)}, indent=2))
        return row

    if checkpoint_path.exists() and not getattr(args, "force", False) and not getattr(args, "resume", False):
        raise FileExistsError(f"Checkpoint exists for {run_id}; pass --force or --resume")

    bundle = build_datasets_for_job(cfg, job, n0=n0, m_per_aux=m_per_aux, k_aux=k_aux, seed=seed, model_type=model_type)
    row["total_train_images"] = bundle.total_train_images
    ensure_dir(run_config_path.parent)
    with open(run_config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump({"config": cfg, "job": job or {}, "run_id": run_id}, handle, sort_keys=False)

    device = get_device(getattr(args, "device", None) or cfg.get("device", "auto"))
    model, diffusion, train_metrics = train_image_model(
        bundle.train,
        bundle.val,
        conditional=not model_type.startswith("unconditional"),
        num_classes=max(len(bundle.class_labels), 1),
        image_size=image_size,
        base_channels=int(cfg.get("model", {}).get("base_channels", 64)),
        channel_mults=cfg.get("model", {}).get("channel_mults", [1, 2, 2, 4]),
        timesteps=int(cfg.get("diffusion", {}).get("timesteps", 1000)),
        schedule=cfg.get("diffusion", {}).get("schedule", "linear"),
        steps=training_steps,
        batch_size=int(cfg.get("training", {}).get("batch_size", 32)),
        lr=float(cfg.get("optimizer", {}).get("lr", 2e-4)),
        device=device,
        precision=cfg.get("training", {}).get("precision", "fp32"),
        checkpoint_path=checkpoint_path,
        train_log_path=log_path,
        resume=getattr(args, "resume", False),
        validation_interval=int(cfg.get("training", {}).get("validation_interval", 100)),
        num_workers=int(cfg.get("training", {}).get("num_workers", 0)),
    )
    row.update(train_metrics)

    eval_start = time.time()
    num_generated = int(row["num_generated"])
    sampling_batch_size = int(cfg.get("evaluation", {}).get("sampling_batch_size", cfg.get("sampling", {}).get("batch_size", cfg.get("training", {}).get("batch_size", 32))))
    samples = sample_batched(
        diffusion,
        model,
        num_samples=num_generated,
        image_size=image_size,
        conditional=not model_type.startswith("unconditional"),
        label=0,
        sampling_steps=int(row["sampling_steps"]),
        batch_size=sampling_batch_size,
    )
    sample_path = outdir / "samples" / f"{run_id}_samples.pt"
    ensure_dir(sample_path.parent)
    torch.save(samples.cpu(), sample_path)

    real_batches = []
    real_eval_limit = int(cfg.get("evaluation", {}).get("real_eval_max", max(num_generated, 1000)))
    val_loader = DataLoader(bundle.target_eval, batch_size=int(cfg.get("training", {}).get("batch_size", 32)), shuffle=False, num_workers=0)
    for x, _ in val_loader:
        real_batches.append(x)
        if sum(batch.shape[0] for batch in real_batches) >= real_eval_limit:
            break
    real = torch.cat(real_batches, dim=0)[:real_eval_limit]
    row["num_real_eval"] = int(real.shape[0])
    if cfg.get("evaluation", {}).get("compute_fid_kid", True):
        row.update(compute_fid_kid(samples.cpu(), real, outdir / "cache" / f"real_{target_synset}_{image_size}.pt", fid_batch_size=int(cfg.get("evaluation", {}).get("fid_batch_size", 64))))
    else:
        row.update({"fid_target": float("nan"), "kid_target_mean": float("nan"), "kid_target_std": float("nan")})
    if cfg.get("evaluation", {}).get("compute_classifier", False):
        row.update(evaluate_classifier_fidelity(samples.cpu(), target_synset, bundle.aux_synsets, device=device, batch_size=int(cfg.get("evaluation", {}).get("classifier_batch_size", cfg.get("training", {}).get("batch_size", 32)))))
    else:
        row.update({"classifier_target_top1_acc": float("nan"), "classifier_target_top5_acc": float("nan"), "auxiliary_leakage_rate": float("nan"), "top1_prediction_histogram_json": "{}"})
    aux_real = None
    if bundle.aux_eval_datasets:
        aux_batches = []
        for aux_dataset in bundle.aux_eval_datasets:
            for x, _ in DataLoader(aux_dataset, batch_size=int(cfg.get("training", {}).get("batch_size", 32)), shuffle=False, num_workers=0):
                aux_batches.append(x)
                break
        if aux_batches:
            aux_real = torch.cat(aux_batches, dim=0)
    if cfg.get("evaluation", {}).get("make_nearest_neighbors", True):
        make_nearest_neighbor_grid(samples.cpu(), real, aux_real, outdir / "figures" / f"{run_id}_nearest_neighbors.png", device=device, nn_batch_size=int(cfg.get("evaluation", {}).get("nn_batch_size", 64)))
    if cfg.get("evaluation", {}).get("compute_feature_similarity", experiment == "C") and bundle.aux_eval_datasets:
        row["average_auxiliary_similarity"] = average_auxiliary_similarity(bundle.target_eval, bundle.aux_eval_datasets, batch_size=int(cfg.get("training", {}).get("batch_size", 32)), device=device)
    else:
        row["average_auxiliary_similarity"] = float("nan")
    row["wallclock_eval_seconds"] = time.time() - eval_start

    append_csv_row(outdir / "metrics.csv", row, FIELDS)
    append_csv_row(Path(resolve_env_path(cfg.get("output_root"), "image_transfer_results")) / "all_metrics.csv", row, FIELDS)
    print(outdir / "metrics.csv")
    return row


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, dest="image_size")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
