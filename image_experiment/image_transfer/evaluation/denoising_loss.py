"""Denoising evaluation compatibility helpers."""

from __future__ import annotations

import hashlib

import torch
from torch.utils.data import TensorDataset

from .corruption_bank import create_corruption_bank, evaluate_corruption_bank


@torch.no_grad()
def evaluate_denoising_bins(
    model,
    diffusion,
    loader,
    device,
    label=None,
    max_batches=4,
    *,
    evaluation_seed: int = 0,
    corruptions_per_image: int = 3,
    corruption_bank=None,
):
    """Backward-compatible deterministic replacement for the old random loop.

    New code should persist a bank created by :func:`create_corruption_bank` and
    call :func:`evaluate_corruption_bank` directly.  This wrapper materializes
    the same leading loader batches on every call and uses a stable synthetic
    manifest hash, so existing trainer calls are deterministic too.
    """

    if corruption_bank is not None:
        metrics = evaluate_corruption_bank(
            model,
            diffusion,
            loader,
            corruption_bank,
            device,
            label=label,
            batch_size=getattr(loader, "batch_size", None) or 64,
            metric_prefix="validation",
        )
        return {
            "all": metrics["validation_epsilon_mse_target"],
            "low": metrics["validation_epsilon_mse_low_noise"],
            "mid": metrics["validation_epsilon_mse_mid_noise"],
            "high": metrics["validation_epsilon_mse_high_noise"],
            "standard_error": metrics["validation_epsilon_mse_standard_error"],
            "num_validation_images": metrics["num_validation_images"],
            "num_corruptions": metrics["num_corruptions"],
            "corruption_bank_hash": metrics["corruption_bank_hash"],
        }

    images = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= int(max_batches):
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        images.extend(item.detach().cpu() for item in x)
    if not images:
        return {"low": float("nan"), "mid": float("nan"), "high": float("nan"), "all": float("nan")}
    tensor = torch.stack(images)
    digest = hashlib.sha256(
        f"legacy-loader:{len(images)}:{tuple(tensor.shape[1:])}".encode("utf-8")
    ).hexdigest()
    bank = create_corruption_bank(
        manifest_hash=digest,
        evaluation_seed=evaluation_seed,
        timesteps=diffusion.timesteps,
        corruptions_per_image=corruptions_per_image,
        num_images=len(images),
        split="validation",
    )
    metrics = evaluate_corruption_bank(
        model,
        diffusion,
        TensorDataset(tensor),
        bank,
        device,
        label=label,
        batch_size=getattr(loader, "batch_size", None) or 64,
    )
    return {
        "all": metrics["validation_epsilon_mse_target"],
        "low": metrics["validation_epsilon_mse_low_noise"],
        "mid": metrics["validation_epsilon_mse_mid_noise"],
        "high": metrics["validation_epsilon_mse_high_noise"],
    }


__all__ = ["evaluate_denoising_bins"]
