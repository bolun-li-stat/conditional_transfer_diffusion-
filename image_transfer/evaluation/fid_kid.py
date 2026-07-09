from __future__ import annotations

from pathlib import Path

import torch


def compute_fid_kid(generated: torch.Tensor, real: torch.Tensor, cache_path: str | Path | None = None) -> dict[str, float]:
    """Compute FID/KID when torchmetrics is installed, otherwise return NaNs.

    The fallback keeps CPU smoke tests lightweight while real runs can install
    torchmetrics/torch-fidelity from requirements-image.txt.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception:
        return {"fid_target": float("nan"), "kid_target_mean": float("nan"), "kid_target_std": float("nan")}

    def to_uint8(x: torch.Tensor) -> torch.Tensor:
        x = ((x.detach().cpu().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        return x

    gen = to_uint8(generated)
    real_u8 = to_uint8(real)
    fid = FrechetInceptionDistance(feature=2048, normalize=False)
    kid = KernelInceptionDistance(feature=2048, normalize=False)
    fid.update(real_u8, real=True)
    fid.update(gen, real=False)
    kid.update(real_u8, real=True)
    kid.update(gen, real=False)
    kid_mean, kid_std = kid.compute()
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"num_real": int(real_u8.shape[0])}, cache_path)
    return {"fid_target": float(fid.compute()), "kid_target_mean": float(kid_mean), "kid_target_std": float(kid_std)}
