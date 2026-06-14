"""Training and experiment orchestration for DDPM Gaussian-mixture transfer simulations."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import trange

from conditional_model import ConditionalDenoiser
from config import (
    ExperimentConfig,
    config_to_dict,
    default_experiment1_configs,
    default_experiment2_configs,
    default_smoke_config,
)
from data import build_low_target_data_split, build_same_total_budget_split, make_gaussian_mixture_spec
from diffusion import DDPM
from eval import RESULT_COLUMNS, estimate_score_risk, generated_metrics, validation_epsilon_mse
from unconditional_model import UnconditionalDenoiser
from utils import append_row_to_csv, describe_device, ensure_dir, get_device, save_json, set_seed, stable_hash


class ConditionalBatchSampler:
    def __init__(self, x: np.ndarray, y: np.ndarray, K: int, mode: str, device: torch.device) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32, device=device)
        self.y = torch.as_tensor(y, dtype=torch.long, device=device)
        self.K = K
        self.mode = mode
        self.device = device
        self.by_class = [torch.where(self.y == k)[0] for k in range(K)]

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "natural":
            idx = torch.randint(0, len(self.y), (batch_size,), device=self.device)
            return self.x[idx], self.y[idx]
        labels = torch.randint(0, self.K, (batch_size,), device=self.device)
        idxs = []
        for k in range(self.K):
            mask = labels == k
            n = int(mask.sum().item())
            if n:
                pool = self.by_class[k]
                chosen = pool[torch.randint(0, len(pool), (n,), device=self.device)]
                idxs.append((mask, chosen))
        x_batch = torch.empty(batch_size, self.x.shape[1], device=self.device)
        for mask, chosen in idxs:
            x_batch[mask] = self.x[chosen]
        return x_batch, labels


def common_setting_id(cfg: ExperimentConfig) -> str:
    return stable_hash([
        cfg.experiment_type,
        cfg.data.covariance_scenario,
        cfg.data.rho,
        cfg.data.mismatch_level,
        cfg.data.K,
        cfg.data.d,
        cfg.data.Delta,
        cfg.n,
        cfg.n_target_train,
        cfg.n_aux_train,
        cfg.seed,
        cfg.diffusion.T,
        cfg.diffusion.beta_start,
        cfg.diffusion.beta_end,
        cfg.diffusion.variance_type,
        cfg.training.sampling_mode,
        cfg.training.training_steps,
    ])


def setting_id(cfg: ExperimentConfig, model_type: str) -> str:
    return stable_hash([common_setting_id(cfg), model_type])


def train_model(
    model: torch.nn.Module,
    diffusion: DDPM,
    train_x: np.ndarray,
    val_x: np.ndarray,
    cfg: ExperimentConfig,
    model_type: str,
    train_y: np.ndarray | None = None,
) -> tuple[torch.nn.Module, pd.DataFrame, Path, float, float]:
    device = diffusion.device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    sid = setting_id(cfg, model_type)
    ckpt_path = cfg.checkpoint_dir / sid / f"{model_type}_last.pt"
    log_path = cfg.log_dir / f"{sid}_{model_type}_train_log.csv"
    ensure_dir(ckpt_path.parent)
    ensure_dir(log_path.parent)
    start_step = 0
    rows: list[dict[str, float]] = []
    if cfg.training.resume_from_checkpoint and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt.get("step", 0))
        if log_path.exists():
            rows = pd.read_csv(log_path).to_dict("records")
    uncond_x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    cond_sampler = ConditionalBatchSampler(train_x, train_y, cfg.data.K, cfg.training.sampling_mode, device) if train_y is not None else None
    final_loss = float("nan")
    val_mse = float("nan")
    if start_step < cfg.training.training_steps:
        bar = trange(start_step, cfg.training.training_steps, desc=f"train {model_type}", leave=False)
        for step in bar:
            model.train()
            if cond_sampler is None:
                idx = torch.randint(0, len(uncond_x), (cfg.training.batch_size,), device=device)
                xb, labels = uncond_x[idx], None
            else:
                xb, labels = cond_sampler.sample(cfg.training.batch_size)
            loss = diffusion.epsilon_loss(model, xb, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
            if (step + 1) % cfg.training.validation_interval == 0 or step + 1 == cfg.training.training_steps:
                label = 0 if train_y is not None else None
                val_mse = validation_epsilon_mse(model, diffusion, val_x, conditional_label=label, batch_size=cfg.training.batch_size)
                rows.append({"step": step + 1, "train_loss": final_loss, "validation_epsilon_mse": val_mse})
                pd.DataFrame(rows).to_csv(log_path, index=False)
                bar.set_postfix(loss=final_loss, val=val_mse)
            if cfg.training.save_checkpoints and ((step + 1) % cfg.training.checkpoint_interval == 0 or step + 1 == cfg.training.training_steps):
                torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "step": step + 1}, ckpt_path)
    elif log_path.exists():
        log_df_existing = pd.read_csv(log_path)
        if len(log_df_existing):
            final_loss = float(log_df_existing.iloc[-1]["train_loss"])
            val_mse = float(log_df_existing.iloc[-1]["validation_epsilon_mse"])
    log_df = pd.DataFrame(rows) if rows else (pd.read_csv(log_path) if log_path.exists() else pd.DataFrame())
    if len(log_df):
        final_loss = float(log_df.iloc[-1]["train_loss"])
        val_mse = float(log_df.iloc[-1]["validation_epsilon_mse"])
    return model, log_df, ckpt_path, final_loss, val_mse


def _build_split(cfg: ExperimentConfig, spec):
    if cfg.experiment_type in {"smoke", "low_target_data"}:
        return build_low_target_data_split(spec, cfg.n_target_train or 100, cfg.n_aux_train or 1000, cfg.evaluation.n_test_target, cfg.seed)
    if cfg.n is None:
        raise ValueError("cfg.n is required for same_total_budget.")
    return build_same_total_budget_split(spec, cfg.n, cfg.evaluation.n_test_target, cfg.seed)


def _existing_metric_mask(existing: pd.DataFrame, cfg: ExperimentConfig, model_type: str) -> pd.Series:
    if existing.empty:
        return pd.Series(dtype=bool)
    mask = existing["experiment_type"].astype(str).eq(cfg.experiment_type)
    mask &= existing["seed"].eq(cfg.seed)
    mask &= existing["model_type"].astype(str).eq(model_type)
    mask &= existing["covariance_scenario"].astype(str).eq(cfg.data.covariance_scenario)
    mask &= existing["K"].eq(cfg.data.K)
    mask &= existing["d"].eq(cfg.data.d)
    mask &= existing["Delta"].eq(cfg.data.Delta)
    mask &= existing["sampling_mode"].astype(str).eq(cfg.training.sampling_mode)
    mask &= existing["training_steps"].eq(cfg.training.training_steps)

    def norm(v):
        return "None" if pd.isna(v) else str(v)
    mask &= existing["rho"].map(norm).eq(norm(cfg.data.rho))
    mask &= existing["mismatch_level"].map(norm).eq(norm(cfg.data.mismatch_level))
    mask &= existing["n"].map(norm).eq(norm(cfg.n))
    mask &= existing["n_target_train"].map(norm).eq(norm(cfg.n_target_train))
    mask &= existing["n_aux_train"].map(norm).eq(norm(cfg.n_aux_train))
    return mask


def run_single_setting(cfg: ExperimentConfig, force: bool = False) -> pd.DataFrame:
    set_seed(cfg.seed)
    device = get_device(cfg.training.device)
    for path in [cfg.results_dir, cfg.figure_dir, cfg.checkpoint_dir, cfg.config_dir, cfg.log_dir, cfg.sample_dir]:
        ensure_dir(path)
    metrics_path = cfg.results_dir / "metrics.csv"
    existing = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    spec = make_gaussian_mixture_spec(
        K=cfg.data.K,
        d=cfg.data.d,
        Delta=cfg.data.Delta,
        seed=cfg.seed,
        target_class=cfg.data.target_class,
        min_pairwise_mean_distance=cfg.data.min_pairwise_mean_distance,
        covariance_scenario=cfg.data.covariance_scenario,
        rho=cfg.data.rho,
        mismatch_level=cfg.data.mismatch_level,
        jitter=cfg.data.jitter,
    )
    split = _build_split(cfg, spec)
    diffusion = DDPM(cfg.diffusion, device)
    rows = []
    save_json(config_to_dict(cfg), cfg.config_dir / f"{stable_hash([cfg.experiment_type, cfg.seed, cfg.n, cfg.n_target_train, cfg.data.covariance_scenario, cfg.data.rho, cfg.data.mismatch_level])}.json")
    for model_type in ["unconditional", "conditional"]:
        sid = setting_id(cfg, model_type)
        if not force and len(existing) and _existing_metric_mask(existing, cfg, model_type).any():
            continue
        if model_type == "unconditional":
            model = UnconditionalDenoiser(cfg.data.d, cfg.model.time_embedding_dim, cfg.model.hidden_width, cfg.model.hidden_layers)
            train_x, train_y, label = split["uncond_train_x"], None, None
        else:
            model = ConditionalDenoiser(cfg.data.d, cfg.data.K, cfg.model.time_embedding_dim, cfg.model.class_embedding_dim, cfg.model.hidden_width, cfg.model.hidden_layers)
            train_x, train_y, label = split["cond_train_x"], split["cond_train_y"], 0
        model, _, ckpt_path, final_loss, val_mse = train_model(model, diffusion, train_x, split["target_val_x"], cfg, model_type, train_y)
        score_risk = estimate_score_risk(model, diffusion, spec.means[spec.target_index], spec.covariances[spec.target_index], cfg.evaluation.score_risk_mc_samples, conditional_label=label, batch_size=cfg.training.batch_size, seed=cfg.seed + 50)
        labels = None if label is None else torch.tensor([label], dtype=torch.long, device=device)
        gen = diffusion.sample(model, cfg.evaluation.n_generated, cfg.data.d, labels=labels, batch_size=cfg.training.batch_size).numpy()
        if cfg.evaluation.save_samples:
            sample_prefix = common_setting_id(cfg)
            np.save(cfg.sample_dir / f"{sample_prefix}_{model_type}_samples.npy", gen)
            if model_type == "unconditional":
                np.save(cfg.sample_dir / f"{sample_prefix}_target_test.npy", split["target_test_x"])
        gm = generated_metrics(gen, split["target_test_x"], spec.means[spec.target_index], spec.covariances[spec.target_index], cfg.evaluation.mmd_max_samples, cfg.seed)
        row: dict[str, Any] = {
            "experiment_type": cfg.experiment_type,
            "covariance_scenario": cfg.data.covariance_scenario,
            "rho": cfg.data.rho,
            "mismatch_level": cfg.data.mismatch_level,
            "K": cfg.data.K,
            "d": cfg.data.d,
            "Delta": cfg.data.Delta,
            "n": cfg.n,
            "n_target_train": cfg.n_target_train,
            "n_aux_train": cfg.n_aux_train,
            "seed": cfg.seed,
            "model_type": model_type,
            "sampling_mode": cfg.training.sampling_mode,
            "training_steps": cfg.training.training_steps,
            "score_risk": score_risk,
            "validation_epsilon_mse": val_mse,
            "mean_error": gm["mean_error"],
            "covariance_error": gm["covariance_error"],
            "gaussian_w2_squared": gm["gaussian_w2_squared"],
            "mmd_rbf": gm["mmd_rbf"],
            "final_train_loss": final_loss,
            "checkpoint_path": str(ckpt_path),
            "figure_dir": str(cfg.figure_dir),
        }
        append_row_to_csv(row, metrics_path, RESULT_COLUMNS)
        rows.append(row)
    return pd.DataFrame(rows)


def build_cli_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    if args.experiment == "smoke":
        configs = [default_smoke_config()]
    elif args.experiment == "low_target_data":
        configs = default_experiment1_configs(args.seeds)
    elif args.experiment == "same_total_budget":
        configs = default_experiment2_configs(args.seeds)
    elif args.experiment == "all":
        configs = default_experiment1_configs(args.seeds) + default_experiment2_configs(args.seeds)
    else:
        raise ValueError(args.experiment)
    for cfg in configs:
        cfg.results_dir = Path(args.results_dir)
        cfg.training.device = args.device
        cfg.training.sampling_mode = args.sampling_mode
        cfg.training.resume_from_checkpoint = args.resume
        if args.training_steps is not None:
            cfg.training.training_steps = args.training_steps
        if args.n_generated is not None:
            cfg.evaluation.n_generated = args.n_generated
        if args.score_risk_mc_samples is not None:
            cfg.evaluation.score_risk_mc_samples = args.score_risk_mc_samples
        if args.save_samples:
            cfg.evaluation.save_samples = True
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["smoke", "low_target_data", "same_total_budget", "all"], default="smoke")
    parser.add_argument("--results-dir", default="results_T500")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampling-mode", choices=["balanced", "natural"], default="balanced")
    parser.add_argument("--training-steps", type=int, default=None)
    parser.add_argument("--n-generated", type=int, default=None)
    parser.add_argument("--score-risk-mc-samples", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    describe_device()
    frames = []
    for cfg in build_cli_configs(args):
        frames.append(run_single_setting(cfg, force=args.force))
    if frames:
        print(pd.concat(frames, ignore_index=True) if any(len(f) for f in frames) else "No new settings were run.")


if __name__ == "__main__":
    main()
