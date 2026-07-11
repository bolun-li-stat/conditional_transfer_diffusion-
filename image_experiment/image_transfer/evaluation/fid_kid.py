"""Compatibility facade for the unified feature-metric implementation."""

from __future__ import annotations

from pathlib import Path
import hashlib

import torch

from .feature_metrics import (
    MetricBackendError,
    MetricComputationError,
    cache_real_feature_stats,
    compute_feature_metrics,
    real_feature_cache_key,
)


def _legacy_tensor_manifest_hash(real: torch.Tensor) -> str:
    """Content hash used only when legacy callers have not migrated manifests."""

    values = real.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(tuple(values.shape)).encode("ascii"))
    try:
        digest.update(values.numpy().tobytes())
    except Exception:
        # This is slower but avoids silently using a shape-only cache key.
        digest.update(bytes(values.view(torch.uint8).flatten().tolist()))
    return f"legacy-real-tensor-{digest.hexdigest()}"


def compute_fid_kid(
    generated: torch.Tensor,
    real: torch.Tensor,
    cache_path: str | Path | None = None,
    fid_batch_size: int = 64,
    *,
    mode: str = "strict",
    real_manifest_hash: str | None = None,
    kid_subset_size: int = 100,
    kid_num_subsets: int = 100,
    evaluation_seed: int = 0,
    device: str | torch.device = "cpu",
    **kwargs,
) -> dict[str, object]:
    """Backward-compatible FID/KID wrapper.

    New code should call :func:`compute_feature_metrics`.  A legacy caller that
    omits ``real_manifest_hash`` receives a full real-tensor content hash, never
    the old model-specific ``num_real`` cache placeholder.
    """

    manifest_hash = real_manifest_hash or _legacy_tensor_manifest_hash(real)
    result = compute_feature_metrics(
        generated,
        real,
        mode=mode,
        real_manifest_hash=manifest_hash,
        cache_path=cache_path,
        compute_fid=True,
        compute_kid=True,
        compute_prdc_metrics=False,
        kid_subset_size=kid_subset_size,
        kid_num_subsets=kid_num_subsets,
        feature_batch_size=fid_batch_size,
        evaluation_seed=evaluation_seed,
        device=device,
        **kwargs,
    )
    return result


__all__ = [
    "MetricBackendError",
    "MetricComputationError",
    "cache_real_feature_stats",
    "compute_feature_metrics",
    "compute_fid_kid",
    "real_feature_cache_key",
]
