"""Train the architecture-matched spectrum-rotation experiment."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import numpy as np
import torch

from conditional_model import ConditionalDenoiser
from config import CAPACITIES, ExperimentConfig, smoke_config
from data import build_paired_split, covariance_family
from diffusion import DDPM
from eval import epsilon_mse, generated_metrics, gradient_alignment, mismatch_diagnostics, score_risk
from utils import (device_from_name, pair_id, pair_payload, read_seed_metrics,
                   setting_id, upsert_seed_metric)


def _model(cfg: ExperimentConfig) -> ConditionalDenoiser:
    cap = CAPACITIES[cfg.capacity]
    return ConditionalDenoiser(cfg.d, cfg.K, cap.time_embedding_dim, cap.class_embedding_dim,
                               cap.hidden_width, cap.hidden_layers)


def _balanced_batch(x: torch.Tensor, y: torch.Tensor, batch: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.randint(3, (batch,), device=x.device, generator=gen)
    indices = torch.empty(batch, dtype=torch.long, device=x.device)
    for label in range(3):
        positions = torch.where(labels == label)[0]
        pool = torch.where(y == label)[0]
        indices[positions] = pool[torch.randint(len(pool), (len(positions),), device=x.device, generator=gen)]
    return x[indices], labels


def train_one(cfg: ExperimentConfig, model_type: str, split: dict[str, np.ndarray],
              diffusion: DDPM, initial_state: dict[str, torch.Tensor], resume: bool,
              generation_enabled: bool = True) -> tuple[ConditionalDenoiser, float, Path]:
    pid = pair_id(cfg, generation_enabled)
    sid = setting_id(pid, model_type, cfg.rotation_deg)
    checkpoint = cfg.results_dir / "checkpoints" / f"{sid}.pt"
    model = _model(cfg).to(diffusion.device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    start = 0
    saved_generator_state = None
    final = float("nan")
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=diffusion.device)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); start = saved["step"]
        saved_generator_state = saved.get("generator_state")
        if "final_loss" in saved:
            final = float(saved["final_loss"])
        elif start >= cfg.training_steps:
            raise ValueError(
                f"Completed legacy checkpoint {checkpoint} lacks final_loss; rerun with --force "
                "instead of silently reporting a fabricated value.")
    if model_type == "target_only":
        x = torch.as_tensor(split["target_train"], device=diffusion.device)
        y = torch.zeros(len(x), dtype=torch.long, device=diffusion.device)
    else:
        x = torch.as_tensor(split["joint_x"], device=diffusion.device)
        y = torch.as_tensor(split["joint_y"], device=diffusion.device)
    gen = torch.Generator(device=diffusion.device).manual_seed(cfg.seed + 50_021)
    if saved_generator_state is not None:
        gen.set_state(saved_generator_state)
    for _ in range(start, cfg.training_steps):
        if model_type == "joint_conditional":
            xb, labels = _balanced_batch(x, y, cfg.batch_size, gen)
        else:
            idx = torch.randint(len(x), (cfg.batch_size,), device=x.device, generator=gen)
            xb, labels = x[idx], y[idx]
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.loss(model, xb, labels, gen); loss.backward(); optimizer.step()
        final = float(loss.detach())
        if (_ + 1) % 1000 == 0:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": _ + 1, "generator_state": gen.get_state(),
                        "final_loss": final}, checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": cfg.training_steps, "generator_state": gen.get_state(),
                "final_loss": final}, checkpoint)
    return model, final, checkpoint


def run_setting(cfg: ExperimentConfig, model_type: str, skip_generation: bool = False,
                resume: bool = False, force: bool = False) -> dict[str, object]:
    cfg.validate(); cfg.results_dir.mkdir(parents=True, exist_ok=True)
    generation_enabled = not skip_generation
    payload = pair_payload(cfg, generation_enabled)
    pid = pair_id(cfg, generation_enabled)
    sid = setting_id(pid, model_type, cfg.rotation_deg)
    existing = read_seed_metrics(cfg.results_dir, cfg.seed)
    if not force and not existing.empty and sid in existing["setting_id"].astype(str).values:
        return {}
    split = build_paired_split(cfg.d, cfg.seed, cfg.rotation_deg, cfg.n_target_train, cfg.n_aux_train,
                               cfg.n_validation, cfg.n_test, cfg.spectrum.lambda_high, cfg.spectrum.lambda_low)
    family = covariance_family(cfg.d, cfg.seed, cfg.rotation_deg, cfg.spectrum.lambda_high, cfg.spectrum.lambda_low)
    reference_target = covariance_family(cfg.d, cfg.seed, 0, cfg.spectrum.lambda_high,
                                         cfg.spectrum.lambda_low).target
    assert np.array_equal(family.target, reference_target), "target covariance changed with rotation"
    device = device_from_name(cfg.device); diffusion = DDPM(cfg.T, cfg.beta_start, cfg.beta_end, device)
    torch.manual_seed(cfg.seed + 40_019)
    initial_model = _model(cfg).to(device); initial_state = copy.deepcopy(initial_model.state_dict())
    grad = (np.nan, np.nan, np.nan)
    cov_dist = score_map_dist = np.nan
    if model_type == "joint_conditional":
        class_arrays = [split["target_train"], split["joint_x"][split["joint_y"]==1], split["joint_x"][split["joint_y"]==2]]
        grad = gradient_alignment(initial_model, diffusion, class_arrays, cfg.seed+60_023, min(cfg.batch_size, len(class_arrays[0])))
        grid = diffusion.alpha_bars.detach().cpu().numpy()[np.linspace(0, cfg.T-1, min(20, cfg.T), dtype=int)]
        cov_dist, score_map_dist = mismatch_diagnostics(family.target, [family.auxiliary_1, family.auxiliary_2], grid)
    model, final_loss, checkpoint = train_one(
        cfg, model_type, split, diffusion, initial_state, resume, generation_enabled)
    evaluation_seed = cfg.seed + 70_027
    risks = {"score_risk": score_risk(model, diffusion, family.target, cfg.score_risk_mc_samples, evaluation_seed)}
    bins = {"low_noise_score_risk": (0, min(100, cfg.T)), "mid_noise_score_risk": (min(100,cfg.T), min(500,cfg.T)), "high_noise_score_risk": (min(500,cfg.T), cfg.T)}
    for name, bounds in bins.items():
        risks[name] = score_risk(model, diffusion, family.target, cfg.score_risk_mc_samples, evaluation_seed+1, bounds) if bounds[0] < bounds[1] else np.nan
    gm = {"gaussian_w2_squared": np.nan, "mean_error": np.nan, "covariance_error": np.nan}
    if not skip_generation:
        gm = generated_metrics(diffusion.sample(model, cfg.n_generated, cfg.d, 0, evaluation_seed+2).numpy(), family.target)
    row = {**payload, "pair_id": pid, "setting_id": sid,
           "model_type": model_type,
           "rotation_deg": np.nan if model_type == "target_only" else cfg.rotation_deg, "K": cfg.K, "d": cfg.d,
           "target_covariance_seed": cfg.seed, "training_steps": cfg.training_steps, **risks,
           "validation_epsilon_mse": epsilon_mse(model, diffusion, split["target_val"], evaluation_seed+3, cfg.batch_size), **gm,
           "grad_cos_target_aux1_init": grad[0], "grad_cos_target_aux2_init": grad[1],
           "grad_cos_target_aux_mean_init": grad[2], "covariance_distance": cov_dist,
           "noised_score_map_distance": score_map_dist, "final_train_loss": final_loss,
           "checkpoint_path": str(checkpoint)}
    upsert_seed_metric(row, cfg.results_dir)
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=["smoke", "full"], default="smoke")
    p.add_argument("--seeds", nargs="*", type=int)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--training-steps", type=int); p.add_argument("--device", default="auto")
    p.add_argument("--n-generated", type=int); p.add_argument("--score-risk-mc-samples", type=int)
    p.add_argument("--skip-generation", action="store_true"); p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args(); results = Path(args.results_dir)
    if args.experiment == "smoke":
        configs = [smoke_config(results)]
    else:
        configs = [ExperimentConfig(seed=s, capacity=c, rotation_deg=r, results_dir=results)
                   for s in (args.seeds if args.seeds is not None else list(range(20)))
                   for c in CAPACITIES for r in (0, 45, 75)]
    completed_target: set[str] = set()
    for cfg in configs:
        for name, value in (("training_steps", args.training_steps), ("n_generated", args.n_generated),
                            ("score_risk_mc_samples", args.score_risk_mc_samples)):
            if value is not None: setattr(cfg, name, value)
        cfg.device = args.device
        key = pair_id(cfg, not args.skip_generation)
        if key not in completed_target:
            run_setting(cfg, "target_only", args.skip_generation, args.resume, args.force); completed_target.add(key)
        run_setting(cfg, "joint_conditional", args.skip_generation, args.resume, args.force)


if __name__ == "__main__": main()
