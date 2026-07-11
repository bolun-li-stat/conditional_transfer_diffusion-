"""Evaluation metrics for target-class DDPM transfer experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import pdist

from diffusion import DDPM
from utils import covariance_np, matrix_sqrt_psd, symmetrize

RESULT_COLUMNS = [
    "experiment_type",
    "covariance_scenario",
    "rho",
    "mismatch_level",
    "target_rho",
    "auxiliary_rhos",
    "sqrt_alpha_bar_T",
    "K",
    "d",
    "Delta",
    "n",
    "n_target_train",
    "n_aux_train",
    "seed",
    "model_type",
    "sampling_mode",
    "training_steps",
    "score_risk",
    "validation_epsilon_mse",
    "mean_error",
    "covariance_error",
    "gaussian_w2_squared",
    "mmd_rbf",
    "final_train_loss",
    "checkpoint_path",
    "figure_dir",
]


@torch.no_grad()
def validation_epsilon_mse(model: torch.nn.Module, diffusion: DDPM, target_val_x: np.ndarray, conditional_label: int | None = None, batch_size: int = 1024) -> float:
    model.eval()
    losses: list[float] = []
    x = torch.as_tensor(target_val_x, dtype=torch.float32, device=diffusion.device)
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size]
        labels = None
        if conditional_label is not None:
            labels = torch.full((xb.shape[0],), conditional_label, device=diffusion.device, dtype=torch.long)
        loss = diffusion.epsilon_loss(model, xb, labels)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


@torch.no_grad()
def estimate_score_risk(
    model: torch.nn.Module,
    diffusion: DDPM,
    target_mean: np.ndarray,
    target_cov: np.ndarray,
    n_mc: int,
    conditional_label: int | None = None,
    batch_size: int = 1024,
    seed: int = 0,
) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    d = target_mean.shape[0]
    mean = torch.as_tensor(target_mean, dtype=torch.float32, device=diffusion.device)
    cov = torch.as_tensor(target_cov, dtype=torch.float32, device=diffusion.device)
    eye = torch.eye(d, device=diffusion.device)
    total = 0.0
    seen = 0
    for _ in range((n_mc + batch_size - 1) // batch_size):
        b = min(batch_size, n_mc - seen)
        if b <= 0:
            break
        t_np = rng.integers(0, diffusion.T, size=b)
        t = torch.as_tensor(t_np, dtype=torch.long, device=diffusion.device)
        x_t_chunks = []
        score_chunks = []
        for t_idx in np.unique(t_np):
            mask_np = t_np == t_idx
            m = int(mask_np.sum())
            ab = diffusion.alpha_bars[int(t_idx)]
            C = ab * cov + (1.0 - ab) * eye
            chol = torch.linalg.cholesky(C)
            z = torch.randn(m, d, device=diffusion.device)
            mt = torch.sqrt(ab) * mean
            xt = mt + z @ chol.T
            score_true = -torch.linalg.solve(C, (xt - mt).T).T
            x_t_chunks.append((mask_np, xt))
            score_chunks.append((mask_np, score_true))
        x_t = torch.empty(b, d, device=diffusion.device)
        score_true = torch.empty(b, d, device=diffusion.device)
        for mask_np, vals in x_t_chunks:
            x_t[torch.as_tensor(mask_np, device=diffusion.device)] = vals
        for mask_np, vals in score_chunks:
            score_true[torch.as_tensor(mask_np, device=diffusion.device)] = vals
        if conditional_label is None:
            eps = model(x_t, t)
        else:
            labels = torch.full((b,), conditional_label, device=diffusion.device, dtype=torch.long)
            eps = model(x_t, t, labels)
        denom = diffusion.sqrt_one_minus_alpha_bars.gather(0, t).reshape(-1, 1)
        score_hat = -eps / denom
        risk = ((score_hat - score_true) ** 2).sum(dim=1).sum().item()
        total += risk
        seen += b
    return float(total / max(seen, 1))


def mean_error(generated: np.ndarray, target_mean: np.ndarray) -> float:
    return float(np.linalg.norm(np.mean(generated, axis=0) - target_mean))


def relative_covariance_error(generated: np.ndarray, target_cov: np.ndarray) -> float:
    cov = covariance_np(generated)
    return float(np.linalg.norm(cov - target_cov, ord="fro") / np.linalg.norm(target_cov, ord="fro"))


def gaussian_w2_squared_from_samples(generated: np.ndarray, target_mean: np.ndarray, target_cov: np.ndarray) -> float:
    m1 = np.mean(generated, axis=0)
    c1 = covariance_np(generated)
    c2 = target_cov
    sqrt_c2 = matrix_sqrt_psd(c2)
    middle = sqrt_c2 @ c1 @ sqrt_c2
    sqrt_middle = matrix_sqrt_psd(middle)
    val = np.sum((m1 - target_mean) ** 2) + np.trace(c1 + c2 - 2.0 * sqrt_middle)
    return float(max(np.real(val), 0.0))


def _rbf_kernel_sum(x: np.ndarray, y: np.ndarray, gamma: float, same: bool, chunk_size: int = 256) -> float:
    total = 0.0
    y_norm = np.sum(y * y, axis=1)[None, :]
    for start in range(0, len(x), chunk_size):
        xb = x[start : start + chunk_size]
        d2 = np.sum(xb * xb, axis=1)[:, None] + y_norm - 2.0 * xb @ y.T
        k = np.exp(-gamma * np.maximum(d2, 0.0))
        if same:
            rows = np.arange(start, start + len(xb))
            valid = rows < k.shape[1]
            k[np.arange(len(xb))[valid], rows[valid]] = 0.0
        total += float(k.sum())
    return total


def mmd_rbf(generated: np.ndarray, target: np.ndarray, max_samples: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x = generated
    y = target
    if len(x) > max_samples:
        x = x[rng.choice(len(x), size=max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[rng.choice(len(y), size=max_samples, replace=False)]
    pooled = np.concatenate([x, y], axis=0)
    subset = pooled if len(pooled) <= 1000 else pooled[rng.choice(len(pooled), size=1000, replace=False)]
    dists = pdist(subset, metric="sqeuclidean")
    positive = dists[dists > 0]
    bandwidth2 = float(np.median(positive)) if len(positive) else 1.0
    gamma = 1.0 / max(2.0 * bandwidth2, 1e-12)
    n, m = len(x), len(y)
    xx = _rbf_kernel_sum(x, x, gamma, same=True) / max(n * (n - 1), 1)
    yy = _rbf_kernel_sum(y, y, gamma, same=True) / max(m * (m - 1), 1)
    xy = _rbf_kernel_sum(x, y, gamma, same=False) / max(n * m, 1)
    return float(max(xx + yy - 2.0 * xy, 0.0))


def generated_metrics(generated: np.ndarray, target_samples: np.ndarray, target_mean: np.ndarray, target_cov: np.ndarray, mmd_max_samples: int, seed: int) -> dict[str, float]:
    return {
        "mean_error": mean_error(generated, target_mean),
        "covariance_error": relative_covariance_error(generated, target_cov),
        "gaussian_w2_squared": gaussian_w2_squared_from_samples(generated, target_mean, target_cov),
        "mmd_rbf": mmd_rbf(generated, target_samples, max_samples=mmd_max_samples, seed=seed),
    }
