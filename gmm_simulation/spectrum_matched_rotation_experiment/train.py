"""Train and evaluate the architecture-matched spectrum-rotation experiment."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from conditional_model import ConditionalDenoiser
from config import CAPACITIES, ExperimentConfig, smoke_config
from data import build_paired_split, covariance_family
from diffusion import DDPM
from eval import (epsilon_mse, generated_metrics, gradient_alignment,
                  mismatch_diagnostics, score_risk)
from utils import (checkpoint_id, device_from_name, identity_payload,
                   read_seed_metrics, setting_id, training_design_id,
                   upsert_seed_metric)


def _model(cfg: ExperimentConfig) -> ConditionalDenoiser:
    cap = CAPACITIES[cfg.capacity]
    return ConditionalDenoiser(cfg.d, cfg.K, cap.time_embedding_dim,
                               cap.class_embedding_dim, cap.hidden_width,
                               cap.hidden_layers)


def _balanced_batch(x: torch.Tensor, y: torch.Tensor, batch: int,
                    generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.randint(3, (batch,), device=x.device, generator=generator)
    indices = torch.empty(batch, dtype=torch.long, device=x.device)
    for label in range(3):
        positions = torch.where(labels == label)[0]
        pool = torch.where(y == label)[0]
        indices[positions] = pool[torch.randint(
            len(pool), (len(positions),), device=x.device, generator=generator)]
    return x[indices], labels


def _checkpoint_path(cfg: ExperimentConfig, model_type: str) -> tuple[str, Path]:
    cid = checkpoint_id(training_design_id(cfg), cfg.seed, model_type,
                        cfg.rotation_deg)
    return cid, cfg.results_dir / "checkpoints" / f"{cid}.pt"


def train_one(cfg: ExperimentConfig, model_type: str,
              split: dict[str, np.ndarray], diffusion: DDPM,
              initial_state: dict[str, torch.Tensor], resume: bool) \
              -> tuple[ConditionalDenoiser, float, Path]:
    _, checkpoint = _checkpoint_path(cfg, model_type)
    model = _model(cfg).to(diffusion.device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    start, saved_generator_state = 0, None
    final_loss = float("nan")
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=diffusion.device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start = int(saved["step"])
        saved_generator_state = saved.get("generator_state")
        if "final_loss" in saved:
            final_loss = float(saved["final_loss"])
        elif start >= cfg.training_steps:
            raise ValueError(f"Completed checkpoint {checkpoint} lacks final_loss; rerun training explicitly.")
    if model_type == "target_only":
        x = torch.as_tensor(split["target_train"], device=diffusion.device)
        y = torch.zeros(len(x), dtype=torch.long, device=diffusion.device)
    else:
        x = torch.as_tensor(split["joint_x"], device=diffusion.device)
        y = torch.as_tensor(split["joint_y"], device=diffusion.device)
    generator = torch.Generator(device=diffusion.device).manual_seed(cfg.seed + 50_021)
    if saved_generator_state is not None:
        generator.set_state(saved_generator_state)
    for step in range(start, cfg.training_steps):
        if model_type == "joint_conditional":
            xb, labels = _balanced_batch(x, y, cfg.batch_size, generator)
        else:
            index = torch.randint(len(x), (cfg.batch_size,), device=x.device,
                                  generator=generator)
            xb, labels = x[index], y[index]
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.loss(model, xb, labels, generator)
        loss.backward(); optimizer.step()
        final_loss = float(loss.detach())
        if (step + 1) % 1000 == 0:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step + 1, "generator_state": generator.get_state(),
                        "final_loss": final_loss}, checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": cfg.training_steps, "generator_state": generator.get_state(),
                "final_loss": final_loss}, checkpoint)
    return model, final_loss, checkpoint


def load_completed_model(cfg: ExperimentConfig, model_type: str,
                         device: torch.device) -> tuple[ConditionalDenoiser, float, Path]:
    _, checkpoint = _checkpoint_path(cfg, model_type)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Generation-only requires checkpoint: {checkpoint}")
    saved = torch.load(checkpoint, map_location=device)
    step = int(saved.get("step", -1))
    if step != cfg.training_steps:
        raise ValueError(f"Checkpoint {checkpoint} is incomplete: step={step}, expected={cfg.training_steps}")
    if "final_loss" not in saved:
        raise ValueError(f"Completed checkpoint {checkpoint} lacks final_loss")
    model = _model(cfg).to(device)
    model.load_state_dict(saved["model"])
    return model, float(saved["final_loss"]), checkpoint


def _setting_identity(cfg: ExperimentConfig, model_type: str) -> tuple[dict[str, Any], str]:
    identity = identity_payload(cfg)
    sid = setting_id(identity["pair_id"], model_type, cfg.rotation_deg)
    return identity, sid


def _generation_only(cfg: ExperimentConfig, model_type: str,
                     split: dict[str, np.ndarray], family, diffusion: DDPM) -> dict[str, Any]:
    identity, sid = _setting_identity(cfg, model_type)
    existing = read_seed_metrics(cfg.results_dir, cfg.seed)
    matches = existing[existing.setting_id.astype(str) == sid] if not existing.empty else existing
    if len(matches) != 1:
        raise ValueError(f"Generation-only requires exactly one existing score row for setting_id={sid}")
    model, final_loss, checkpoint = load_completed_model(cfg, model_type, diffusion.device)
    evaluation_seed = cfg.seed + 70_027
    generated = diffusion.sample(model, cfg.n_generated, cfg.d, 0,
                                 evaluation_seed + 2).numpy()
    metrics = generated_metrics(generated, family.target)
    row = matches.iloc[0].to_dict()
    row.update(metrics)
    row.update({"training_complete": True, "score_evaluation_complete": True,
                "generation_evaluation_complete": True,
                "final_train_loss": final_loss, "checkpoint_path": str(checkpoint),
                **identity, "setting_id": sid})
    upsert_seed_metric(row, cfg.results_dir)
    return row


def run_setting(cfg: ExperimentConfig, model_type: str,
                skip_generation: bool = False, resume: bool = False,
                force: bool = False, generation_only: bool = False) -> dict[str, Any]:
    cfg.validate(); cfg.results_dir.mkdir(parents=True, exist_ok=True)
    identity, sid = _setting_identity(cfg, model_type)
    existing = read_seed_metrics(cfg.results_dir, cfg.seed)
    if (not generation_only and not force and not existing.empty
            and sid in existing["setting_id"].astype(str).values):
        return {}
    split = build_paired_split(cfg.d, cfg.seed, cfg.rotation_deg,
                               cfg.n_target_train, cfg.n_aux_train,
                               cfg.n_validation, cfg.n_test,
                               cfg.spectrum.lambda_high, cfg.spectrum.lambda_low)
    family = covariance_family(cfg.d, cfg.seed, cfg.rotation_deg,
                               cfg.spectrum.lambda_high, cfg.spectrum.lambda_low)
    reference = covariance_family(cfg.d, cfg.seed, 0, cfg.spectrum.lambda_high,
                                  cfg.spectrum.lambda_low).target
    assert np.array_equal(family.target, reference), "target covariance changed with rotation"
    device = device_from_name(cfg.device)
    diffusion = DDPM(cfg.T, cfg.beta_start, cfg.beta_end, device)
    if generation_only:
        return _generation_only(cfg, model_type, split, family, diffusion)

    torch.manual_seed(cfg.seed + 40_019)
    initial_model = _model(cfg).to(device)
    initial_state = copy.deepcopy(initial_model.state_dict())
    gradient = (np.nan, np.nan, np.nan)
    covariance_distance = noised_distance = np.nan
    if model_type == "joint_conditional":
        class_arrays = [split["target_train"],
                        split["joint_x"][split["joint_y"] == 1],
                        split["joint_x"][split["joint_y"] == 2]]
        gradient = gradient_alignment(initial_model, diffusion, class_arrays,
                                      cfg.seed + 60_023,
                                      min(cfg.batch_size, len(class_arrays[0])))
        grid = diffusion.alpha_bars.detach().cpu().numpy()[
            np.linspace(0, cfg.T - 1, min(20, cfg.T), dtype=int)]
        covariance_distance, noised_distance = mismatch_diagnostics(
            family.target, [family.auxiliary_1, family.auxiliary_2], grid)
    model, final_loss, checkpoint = train_one(
        cfg, model_type, split, diffusion, initial_state, resume)
    evaluation_seed = cfg.seed + 70_027
    risks = {"score_risk": score_risk(model, diffusion, family.target,
                                       cfg.score_risk_mc_samples, evaluation_seed)}
    bins = {"low_noise_score_risk": (0, min(100, cfg.T)),
            "mid_noise_score_risk": (min(100, cfg.T), min(500, cfg.T)),
            "high_noise_score_risk": (min(500, cfg.T), cfg.T)}
    for name, bounds in bins.items():
        risks[name] = (score_risk(model, diffusion, family.target,
                                  cfg.score_risk_mc_samples, evaluation_seed + 1, bounds)
                       if bounds[0] < bounds[1] else np.nan)
    generation_metrics = {"gaussian_w2_squared": np.nan, "mean_error": np.nan,
                          "covariance_error": np.nan}
    if not skip_generation:
        generated = diffusion.sample(model, cfg.n_generated, cfg.d, 0,
                                     evaluation_seed + 2).numpy()
        generation_metrics = generated_metrics(generated, family.target)
    cid, _ = _checkpoint_path(cfg, model_type)
    row: dict[str, Any] = {
        **identity, "checkpoint_id": cid, "setting_id": sid,
        "model_type": model_type,
        "rotation_deg": np.nan if model_type == "target_only" else cfg.rotation_deg,
        "target_covariance_seed": cfg.seed, **risks,
        "validation_epsilon_mse": epsilon_mse(
            model, diffusion, split["target_val"], evaluation_seed + 3, cfg.batch_size),
        **generation_metrics,
        "grad_cos_target_aux1_init": gradient[0],
        "grad_cos_target_aux2_init": gradient[1],
        "grad_cos_target_aux_mean_init": gradient[2],
        "covariance_distance": covariance_distance,
        "noised_score_map_distance": noised_distance,
        "final_train_loss": final_loss, "checkpoint_path": str(checkpoint),
        "training_complete": True, "score_evaluation_complete": True,
        "generation_evaluation_complete": not skip_generation,
    }
    upsert_seed_metric(row, cfg.results_dir)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--capacity", choices=["standard", "limited", "all"], default="all")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--training-steps", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-generated", type=int)
    parser.add_argument("--score-risk-mc-samples", type=int)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--skip-generation", action="store_true")
    stage.add_argument("--generation-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); results = Path(args.results_dir)
    capacities = list(CAPACITIES) if args.capacity == "all" else [args.capacity]
    if args.experiment == "smoke":
        cfg = smoke_config(results)
        cfg.capacity = "limited" if args.capacity == "all" else args.capacity
        configs = [cfg]
    else:
        configs = [ExperimentConfig(seed=seed, capacity=capacity,
                                    rotation_deg=rotation, results_dir=results)
                   for seed in (args.seeds if args.seeds is not None else range(20))
                   for capacity in capacities for rotation in (0, 45, 75)]
    completed_target: set[tuple[str, int]] = set()
    for cfg in configs:
        for name, value in (("training_steps", args.training_steps),
                            ("n_generated", args.n_generated),
                            ("score_risk_mc_samples", args.score_risk_mc_samples)):
            if value is not None:
                setattr(cfg, name, value)
        cfg.device = args.device
        target_key = (training_design_id(cfg), cfg.seed)
        if target_key not in completed_target:
            run_setting(cfg, "target_only", args.skip_generation, args.resume,
                        args.force, args.generation_only)
            completed_target.add(target_key)
        run_setting(cfg, "joint_conditional", args.skip_generation, args.resume,
                    args.force, args.generation_only)


if __name__ == "__main__":
    main()
