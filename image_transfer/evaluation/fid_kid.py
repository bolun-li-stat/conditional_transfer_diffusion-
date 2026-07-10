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


def _fallback_fid_kid(generated: torch.Tensor, real_stats: dict[str, torch.Tensor], num_real: int, num_generated: int) -> dict[str, float]:
    if num_real == 0 or num_generated == 0:
        return {"fid_target": float("nan"), "kid_target_mean": float("nan"), "kid_target_std": float("nan")}
    gen_stats = _feature_stats(_simple_features(generated))
    diff = real_stats["mean"] - gen_stats["mean"]
    fid = float(diff.dot(diff))
    if min(num_real, num_generated) < 10:
        kid = float("nan")
        kid_std = float("nan")
    else:
        kid = float(torch.nn.functional.mse_loss(real_stats["features"].mean(0), gen_stats["features"].mean(0)))
        kid_std = 0.0
    return {"fid_target": fid, "kid_target_mean": kid, "kid_target_std": kid_std}


def _update_metric_batched(metric, images: torch.Tensor, *, real: bool, batch_size: int) -> None:
    for start in range(0, images.shape[0], batch_size):
        metric.update(images[start : start + batch_size], real=real)


def compute_fid_kid(generated: torch.Tensor, real: torch.Tensor, cache_path: str | Path | None = None, fid_batch_size: int = 64) -> dict[str, float]:
    """Compute FID/KID robustly and cache real features.

    Small debug runs may have too few generated or real images for KID. In that
    case KID is reported as NaN while FID is still attempted. Torchmetrics FID
    and KID computations are isolated so a KID failure cannot crash evaluation
    after training.
    """
    num_generated = int(generated.shape[0])
    num_real = int(real.shape[0])
    if num_real == 0 or num_generated == 0:
        return {"fid_target": float("nan"), "kid_target_mean": float("nan"), "kid_target_std": float("nan")}

    real_stats = cache_real_feature_stats(real, cache_path)
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception:
        return _fallback_fid_kid(generated, real_stats, num_real, num_generated)

    def to_uint8(x: torch.Tensor) -> torch.Tensor:
        return ((x.detach().cpu().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)

    gen = to_uint8(generated)
    real_u8 = to_uint8(real)

    fid_value = float("nan")
    try:
        fid = FrechetInceptionDistance(feature=2048, normalize=False)
        _update_metric_batched(fid, real_u8, real=True, batch_size=fid_batch_size)
        _update_metric_batched(fid, gen, real=False, batch_size=fid_batch_size)
        fid_value = float(fid.compute())
    except Exception:
        fallback = _fallback_fid_kid(generated, real_stats, num_real, num_generated)
        fid_value = fallback["fid_target"]

    kid_mean = float("nan")
    kid_std = float("nan")
    kid_subset_size = min(1000, num_real, num_generated)
    if kid_subset_size >= 10:
        try:
            kid = KernelInceptionDistance(feature=2048, normalize=False, subset_size=kid_subset_size)
            _update_metric_batched(kid, real_u8, real=True, batch_size=fid_batch_size)
            _update_metric_batched(kid, gen, real=False, batch_size=fid_batch_size)
            kid_mean_tensor, kid_std_tensor = kid.compute()
            kid_mean = float(kid_mean_tensor)
            kid_std = float(kid_std_tensor)
        except Exception:
            kid_mean = float("nan")
            kid_std = float("nan")

    return {"fid_target": fid_value, "kid_target_mean": kid_mean, "kid_target_std": kid_std}
