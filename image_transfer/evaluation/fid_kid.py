from __future__ import annotations

from pathlib import Path

import torch


def _simple_features(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.adaptive_avg_pool2d(x.float(), (8, 8)).flatten(1)


def _feature_stats(features: torch.Tensor) -> dict[str, torch.Tensor]:
    features = features.double().cpu()
    mean = features.mean(dim=0)
    centered = features - mean
    cov = centered.T @ centered / max(features.shape[0] - 1, 1)
    return {"features": features, "mean": mean, "cov": cov, "num": torch.tensor(features.shape[0])}


def cache_real_feature_stats(real: torch.Tensor, cache_path: str | Path | None) -> dict[str, torch.Tensor]:
    if cache_path is None:
        return _feature_stats(_simple_features(real))
    path = Path(cache_path)
    if path.exists():
        cached = torch.load(path, map_location="cpu")
        if "features" in cached and "mean" in cached and "cov" in cached:
            return cached
    stats = _feature_stats(_simple_features(real))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)
    return stats


def compute_fid_kid(generated: torch.Tensor, real: torch.Tensor, cache_path: str | Path | None = None) -> dict[str, float]:
    """Compute FID/KID with torchmetrics when available and cache real features.

    The cache stores real feature tensors and summary statistics, not only counts.
    A lightweight pooled-feature fallback keeps smoke tests independent of optional
    torchmetrics/torch-fidelity installs.
    """
    real_stats = cache_real_feature_stats(real, cache_path)
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception:
        gen_stats = _feature_stats(_simple_features(generated))
        diff = real_stats["mean"] - gen_stats["mean"]
        fid = float(diff.dot(diff))
        kid = float(torch.nn.functional.mse_loss(real_stats["features"].mean(0), gen_stats["features"].mean(0)))
        return {"fid_target": fid, "kid_target_mean": kid, "kid_target_std": 0.0}

    def to_uint8(x: torch.Tensor) -> torch.Tensor:
        return ((x.detach().cpu().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)

    gen = to_uint8(generated)
    real_u8 = to_uint8(real)
    fid = FrechetInceptionDistance(feature=2048, normalize=False)
    kid = KernelInceptionDistance(feature=2048, normalize=False)
    fid.update(real_u8, real=True)
    fid.update(gen, real=False)
    kid.update(real_u8, real=True)
    kid.update(gen, real=False)
    kid_mean, kid_std = kid.compute()
    return {"fid_target": float(fid.compute()), "kid_target_mean": float(kid_mean), "kid_target_std": float(kid_std)}
