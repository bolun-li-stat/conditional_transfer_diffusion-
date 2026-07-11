"""Strict feature metrics for target-class image evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .prdc import compute_prdc as _compute_prdc


FEATURE_METRIC_SCHEMA_VERSION = "2.0"


class MetricBackendError(RuntimeError):
    """Raised when a required strict-mode metric backend is unavailable."""


class MetricComputationError(RuntimeError):
    """Raised when a requested strict-mode metric fails to compute."""


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _default_extractor_id(feature_dimension: int) -> str:
    """Version the concrete backend in cache identity and result metadata."""

    return (
        f"torchmetrics-inception-v3-compat-{int(feature_dimension)}"
        f":torchmetrics-{_package_version('torchmetrics')}"
        f":torch-fidelity-{_package_version('torch-fidelity')}"
        f":torch-{torch.__version__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def real_feature_cache_key(
    *,
    manifest_hash: str,
    feature_extractor_id: str,
    preprocessing_config: Mapping[str, Any],
    image_size: int | Sequence[int],
    metric_schema_version: str = FEATURE_METRIC_SCHEMA_VERSION,
) -> str:
    """Hash every input that can change cached real features."""

    if not manifest_hash:
        raise ValueError("manifest_hash is required for a real-feature cache key")
    payload = {
        "manifest_hash": str(manifest_hash),
        "feature_extractor_id": str(feature_extractor_id),
        "preprocessing_config": dict(preprocessing_config),
        "image_size": list(image_size) if isinstance(image_size, (tuple, list)) else int(image_size),
        "metric_schema_version": str(metric_schema_version),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


# Descriptive alias used by some integration code.
build_real_feature_cache_key = real_feature_cache_key


def _simple_features(images: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.adaptive_avg_pool2d(images.detach().float(), (8, 8)).flatten(1)


def _feature_stats(features: torch.Tensor) -> dict[str, Any]:
    values = features.detach().double().cpu()
    if values.ndim != 2:
        raise ValueError("features must have shape [samples, dimensions]")
    mean = values.mean(dim=0) if values.shape[0] else torch.full((values.shape[1],), float("nan"))
    if values.shape[0] >= 2:
        centered = values - mean
        covariance = centered.T @ centered / (values.shape[0] - 1)
    else:
        covariance = torch.full((values.shape[1], values.shape[1]), float("nan"), dtype=torch.float64)
    return {"features": values, "mean": mean, "cov": covariance, "num": int(values.shape[0])}


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def cache_real_feature_stats(
    real: torch.Tensor,
    cache_path: str | Path | None,
    *,
    cache_key: str | None = None,
) -> dict[str, Any]:
    """Compatibility helper that caches actual pooled-pixel features.

    This helper is intentionally a debug primitive.  Strict-mode callers should
    use :func:`compute_feature_metrics`, whose cache key includes the manifest,
    extractor and preprocessing configuration.
    """

    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists():
        cached = torch.load(path, map_location="cpu")
        if (
            isinstance(cached, dict)
            and {"features", "mean", "cov", "num"}.issubset(cached)
            and (cache_key is None or cached.get("cache_key") == cache_key)
        ):
            return cached
    result = _feature_stats(_simple_features(real))
    result.update({"cache_key": cache_key, "feature_extractor_id": "debug-pooled-pixels-8x8"})
    if path is not None:
        _atomic_torch_save(result, path)
    return result


def _to_uint8(images: torch.Tensor, image_input_range: tuple[float, float]) -> torch.Tensor:
    if images.dtype == torch.uint8:
        return images.detach().cpu()
    lower, upper = float(image_input_range[0]), float(image_input_range[1])
    if not lower < upper:
        raise ValueError("image_input_range must be increasing")
    normalized = (images.detach().float().cpu().clamp(lower, upper) - lower) / (upper - lower)
    return (normalized * 255.0).round().to(torch.uint8)


def _unwrap_features(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        features = output
    elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        features = output[0]
    elif isinstance(output, Mapping):
        tensors = [value for value in output.values() if isinstance(value, torch.Tensor)]
        if not tensors:
            raise TypeError("feature extractor mapping did not contain a tensor")
        features = tensors[0]
    else:
        raise TypeError("feature extractor returned an unsupported value")
    if features.ndim > 2:
        features = features.flatten(1)
    if features.ndim != 2:
        raise ValueError("feature extractor must return [samples, dimensions]")
    return features


def _build_torchmetrics_extractor(feature_dimension: int, device: torch.device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception as exc:
        raise MetricBackendError(
            "strict evaluation requires torchmetrics with torch-fidelity support; "
            "install the image_experiment requirements and make the Inception weights available"
        ) from exc
    try:
        metric = FrechetInceptionDistance(feature=int(feature_dimension), normalize=False).to(device)
        network = metric.inception.eval()
    except Exception as exc:
        raise MetricBackendError(
            "could not initialize the strict FID/KID Inception feature extractor"
        ) from exc
    return network


def preflight_feature_metric_backend(
    *,
    feature_dimension: int = 2048,
    device: str | torch.device = "cpu",
) -> dict[str, str | int]:
    """Initialize required strict-evaluation weights before an expensive training run."""

    network = _build_torchmetrics_extractor(int(feature_dimension), torch.device(device))
    del network
    return {
        "feature_extractor_id": _default_extractor_id(int(feature_dimension)),
        "feature_dimension": int(feature_dimension),
        "torchmetrics_version": _package_version("torchmetrics"),
        "torch_fidelity_version": _package_version("torch-fidelity"),
    }


@torch.no_grad()
def _extract_features(
    images: torch.Tensor,
    extractor,
    *,
    batch_size: int,
    device: torch.device,
    image_input_range: tuple[float, float],
) -> torch.Tensor:
    if int(batch_size) < 1:
        raise ValueError("feature_batch_size must be positive")
    uint8_images = _to_uint8(images, image_input_range)
    chunks: list[torch.Tensor] = []
    if hasattr(extractor, "eval"):
        extractor.eval()
    for start in range(0, uint8_images.shape[0], int(batch_size)):
        batch = uint8_images[start : start + int(batch_size)].to(device)
        features = _unwrap_features(extractor(batch))
        chunks.append(features.detach().float().cpu())
    if not chunks:
        return torch.empty(0, 0)
    return torch.cat(chunks, dim=0)


def _covariance(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = features.detach().double().cpu()
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("FID requires at least two samples in each feature set")
    mean = values.mean(dim=0)
    centered = values - mean
    return mean, centered.T @ centered / (values.shape[0] - 1)


def _frechet_distance(real_features: torch.Tensor, generated_features: torch.Tensor) -> float:
    real_mean, real_covariance = _covariance(real_features)
    generated_mean, generated_covariance = _covariance(generated_features)
    if real_mean.shape != generated_mean.shape:
        raise ValueError("real and generated feature dimensions differ")
    real_covariance = (real_covariance + real_covariance.T) / 2.0
    generated_covariance = (generated_covariance + generated_covariance.T) / 2.0
    eigenvalues, eigenvectors = torch.linalg.eigh(real_covariance)
    real_sqrt = (eigenvectors * eigenvalues.clamp_min(0.0).sqrt().unsqueeze(0)) @ eigenvectors.T
    middle = real_sqrt @ generated_covariance @ real_sqrt
    middle = (middle + middle.T) / 2.0
    trace_sqrt_product = torch.linalg.eigvalsh(middle).clamp_min(0.0).sqrt().sum()
    mean_difference = real_mean - generated_mean
    value = (
        mean_difference.dot(mean_difference)
        + torch.trace(real_covariance)
        + torch.trace(generated_covariance)
        - 2.0 * trace_sqrt_product
    )
    # Small negative values are possible from eigensolver roundoff.
    return float(value.clamp_min(0.0))


def _polynomial_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x @ y.T / x.shape[1] + 1.0).pow(3)


def _unbiased_polynomial_mmd(real: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    count = real.shape[0]
    real_kernel = _polynomial_kernel(real, real)
    generated_kernel = _polynomial_kernel(generated, generated)
    cross_kernel = _polynomial_kernel(real, generated)
    real_within = (real_kernel.sum() - real_kernel.diagonal().sum()) / (count * (count - 1))
    generated_within = (generated_kernel.sum() - generated_kernel.diagonal().sum()) / (count * (count - 1))
    return real_within + generated_within - 2.0 * cross_kernel.mean()


def _kid(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    *,
    subset_size: int,
    num_subsets: int,
    evaluation_seed: int,
) -> tuple[float, float]:
    if int(subset_size) < 2:
        raise ValueError("KID subset_size must be at least two")
    if int(num_subsets) < 1:
        raise ValueError("KID num_subsets must be positive")
    if min(real_features.shape[0], generated_features.shape[0]) < int(subset_size):
        raise ValueError(
            f"KID requested subset_size={subset_size} but only "
            f"{min(real_features.shape[0], generated_features.shape[0])} paired samples are available"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(evaluation_seed) % (2**63 - 1))
    real = real_features.detach().double().cpu()
    generated = generated_features.detach().double().cpu()
    estimates = []
    for _ in range(int(num_subsets)):
        real_indices = torch.randperm(real.shape[0], generator=generator)[: int(subset_size)]
        generated_indices = torch.randperm(generated.shape[0], generator=generator)[: int(subset_size)]
        estimates.append(_unbiased_polynomial_mmd(real[real_indices], generated[generated_indices]))
    values = torch.stack(estimates)
    standard_deviation = values.std(unbiased=True) if values.numel() > 1 else torch.zeros((), dtype=values.dtype)
    return float(values.mean()), float(standard_deviation)


def _resolve_cache_path(cache_path: str | Path | None, cache_dir: str | Path | None, key: str) -> Path | None:
    if cache_path is not None:
        return Path(cache_path)
    if cache_dir is not None:
        return Path(cache_dir) / f"real_features_{key}.pt"
    return None


def _load_or_extract_real_features(
    real: torch.Tensor,
    extractor,
    *,
    key: str,
    path: Path | None,
    extractor_id: str,
    feature_batch_size: int,
    device: torch.device,
    image_input_range: tuple[float, float],
) -> tuple[torch.Tensor, bool]:
    if path is not None and path.exists():
        cached = torch.load(path, map_location="cpu")
        if (
            isinstance(cached, dict)
            and cached.get("cache_key") == key
            and cached.get("feature_extractor_id") == extractor_id
            and isinstance(cached.get("features"), torch.Tensor)
            and int(cached.get("num", -1)) == int(real.shape[0])
        ):
            return cached["features"].float().cpu(), True
    features = _extract_features(
        real,
        extractor,
        batch_size=feature_batch_size,
        device=device,
        image_input_range=image_input_range,
    )
    stats = _feature_stats(features)
    stats.update(
        {
            "cache_key": key,
            "feature_extractor_id": extractor_id,
            "metric_schema_version": FEATURE_METRIC_SCHEMA_VERSION,
        }
    )
    if path is not None:
        _atomic_torch_save(stats, path)
    return features, False


def _debug_metrics(
    generated: torch.Tensor,
    real: torch.Tensor,
    *,
    manifest_hash: str,
    cache_path: str | Path | None,
    cache_dir: str | Path | None,
) -> dict[str, Any]:
    preprocessing = {"input_range": [-1.0, 1.0], "pool": [8, 8], "resize": None, "antialias": None}
    image_size = list(real.shape[-2:]) if real.ndim == 4 else []
    key = real_feature_cache_key(
        manifest_hash=manifest_hash,
        feature_extractor_id="debug-pooled-pixels-8x8",
        preprocessing_config=preprocessing,
        image_size=image_size,
    )
    path = _resolve_cache_path(cache_path, cache_dir, key)
    stats = cache_real_feature_stats(real, path, cache_key=key)
    generated_features = _simple_features(generated).double().cpu()
    if not len(generated_features) or not int(stats["num"]):
        distance = float("nan")
    else:
        difference = generated_features.mean(dim=0) - stats["features"].mean(dim=0)
        distance = float(difference.square().mean())
    return {
        "debug_pooled_pixel_distance": distance,
        "evaluation_mode": "debug",
        "metric_backend": "debug_pooled_pixels",
        "metric_backend_version": "internal-1",
        "feature_extractor_name": "adaptive_avg_pool_8x8",
        "feature_dimension": int(generated_features.shape[1]) if generated_features.ndim == 2 else 0,
        "image_input_range": "[-1.0,1.0]",
        "resize_size": "none",
        "resize_interpolation": "none",
        "antialias": "not_applicable",
        "real_manifest_hash": manifest_hash,
        "real_feature_cache_key": key,
        "real_feature_cache_path": str(path) if path is not None else "",
        "num_real_eval": int(real.shape[0]),
        "num_generated_eval": int(generated.shape[0]),
        "metric_schema_version": FEATURE_METRIC_SCHEMA_VERSION,
    }


@torch.no_grad()
def compute_feature_metrics(
    generated: torch.Tensor,
    real: torch.Tensor,
    *,
    mode: str = "strict",
    real_manifest_hash: str | None = None,
    cache_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    metric_backend: str = "torchmetrics",
    feature_extractor=None,
    feature_extractor_id: str | None = None,
    feature_dimension: int = 2048,
    image_input_range: tuple[float, float] = (-1.0, 1.0),
    compute_fid: bool = True,
    compute_kid: bool = True,
    compute_prdc: bool | None = None,
    compute_prdc_metrics: bool | None = None,
    compute_inception_score: bool = False,
    fid_reliable_min_real: int = 1000,
    kid_subset_size: int = 100,
    kid_num_subsets: int = 100,
    prdc_k: int = 5,
    feature_batch_size: int = 64,
    distance_batch_size: int = 512,
    inception_score_splits: int = 10,
    evaluation_seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Compute one consistent feature-metric suite.

    ``strict`` mode never substitutes another mathematical quantity when a
    backend fails.  ``debug`` mode performs only the fast pooled-pixel diagnostic
    and deliberately returns no ``fid_target`` or ``kid_target_*`` keys.

    A custom extractor, when supplied, must accept uint8 NCHW tensors in [0,255].
    This injection point is useful for offline tests; main runs should use
    the recorded default Inception backend.
    """

    normalized_mode = str(mode).lower()
    if normalized_mode not in {"strict", "debug"}:
        raise ValueError("evaluation mode must be 'strict' or 'debug'")
    if compute_prdc is not None and compute_prdc_metrics is not None and bool(compute_prdc) != bool(compute_prdc_metrics):
        raise ValueError("compute_prdc and compute_prdc_metrics aliases disagree")
    should_compute_prdc = (
        bool(compute_prdc)
        if compute_prdc is not None
        else (bool(compute_prdc_metrics) if compute_prdc_metrics is not None else True)
    )
    if generated.ndim != 4 or real.ndim != 4:
        raise ValueError("generated and real images must be NCHW tensors")
    if not real_manifest_hash:
        if normalized_mode == "strict":
            raise ValueError("strict mode requires real_manifest_hash")
        real_manifest_hash = "debug-untracked-real-images"
    if normalized_mode == "debug":
        return _debug_metrics(
            generated, real, manifest_hash=real_manifest_hash, cache_path=cache_path, cache_dir=cache_dir
        )
    if int(real.shape[0]) == 0 or int(generated.shape[0]) == 0:
        raise ValueError("strict metrics require non-empty real and generated samples")
    if int(fid_reliable_min_real) < 1:
        raise ValueError("fid_reliable_min_real must be positive")
    if metric_backend != "torchmetrics" and feature_extractor is None:
        raise MetricBackendError(
            f"unsupported strict metric backend {metric_backend!r}; use 'torchmetrics' or inject a tested extractor"
        )

    evaluation_device = torch.device(device)
    extractor_id = feature_extractor_id or (
        _default_extractor_id(int(feature_dimension))
        if feature_extractor is None
        else "provided-feature-extractor"
    )
    preprocessing = {
        "image_input_range": [float(image_input_range[0]), float(image_input_range[1])],
        "backend_input_dtype": "uint8",
        "backend_input_range": [0, 255],
        "resize_size": 299,
        "resize_interpolation": "backend_internal_inception_compat",
        "antialias": "backend_internal_inception_compat",
    }
    key = real_feature_cache_key(
        manifest_hash=real_manifest_hash,
        feature_extractor_id=extractor_id,
        preprocessing_config=preprocessing,
        image_size=list(real.shape[-2:]),
    )
    resolved_cache_path = _resolve_cache_path(cache_path, cache_dir, key)
    extractor = feature_extractor or _build_torchmetrics_extractor(feature_dimension, evaluation_device)
    try:
        real_features, cache_hit = _load_or_extract_real_features(
            real,
            extractor,
            key=key,
            path=resolved_cache_path,
            extractor_id=extractor_id,
            feature_batch_size=feature_batch_size,
            device=evaluation_device,
            image_input_range=image_input_range,
        )
        generated_features = _extract_features(
            generated,
            extractor,
            batch_size=feature_batch_size,
            device=evaluation_device,
            image_input_range=image_input_range,
        )
    except MetricBackendError:
        raise
    except Exception as exc:
        raise MetricComputationError("strict feature extraction failed; no fallback was used") from exc
    if real_features.shape[1:] != generated_features.shape[1:]:
        raise MetricComputationError("real and generated feature dimensions differ")

    result: dict[str, Any] = {
        "evaluation_mode": "strict",
        "metric_backend": "torchmetrics_inception_features" if feature_extractor is None else "provided_tested_extractor",
        "metric_backend_version": _package_version("torchmetrics") if feature_extractor is None else "provided",
        "metric_implementation": "internal_frechet_eigendecomposition+unbiased_polynomial_mmd+batched_prdc",
        "torch_version": str(torch.__version__),
        "torchvision_version": _package_version("torchvision"),
        "feature_extractor_name": extractor_id,
        "feature_dimension": int(real_features.shape[1]),
        "image_input_range": _canonical_json(list(image_input_range)),
        "resize_size": preprocessing["resize_size"],
        "resize_interpolation": preprocessing["resize_interpolation"],
        "antialias": preprocessing["antialias"],
        "preprocessing_config_json": _canonical_json(preprocessing),
        "real_manifest_hash": real_manifest_hash,
        "real_feature_cache_key": key,
        "real_feature_cache_path": str(resolved_cache_path) if resolved_cache_path is not None else "",
        "real_feature_cache_hit": bool(cache_hit),
        "num_real_eval": int(real.shape[0]),
        "num_generated_eval": int(generated.shape[0]),
        "metric_schema_version": FEATURE_METRIC_SCHEMA_VERSION,
        "kid_subset_size": int(kid_subset_size),
        "kid_num_subsets": int(kid_num_subsets),
        "prdc_k": int(prdc_k),
        "fid_reliable_min_real": int(fid_reliable_min_real),
    }

    if compute_fid:
        result["fid_reliability_warning"] = (
            ""
            if int(real.shape[0]) >= int(fid_reliable_min_real)
            else (
                f"FID uses only {int(real.shape[0])} real reference images, below the pre-specified "
                f"reliability threshold of {int(fid_reliable_min_real)}; interpret it as a secondary endpoint"
            )
        )
        if min(real_features.shape[0], generated_features.shape[0]) < 2:
            result.update({"fid_target": float("nan"), "fid_target_status": "unavailable: fewer than two samples"})
        else:
            try:
                result.update({"fid_target": _frechet_distance(real_features, generated_features), "fid_target_status": "ok"})
            except Exception as exc:
                raise MetricComputationError("FID computation failed; no fallback was used") from exc

    if compute_kid:
        try:
            kid_mean, kid_std = _kid(
                real_features,
                generated_features,
                subset_size=kid_subset_size,
                num_subsets=kid_num_subsets,
                evaluation_seed=evaluation_seed,
            )
            result.update(
                {"kid_target_mean": kid_mean, "kid_target_std": kid_std, "kid_target_status": "ok"}
            )
        except ValueError as exc:
            result.update(
                {
                    "kid_target_mean": float("nan"),
                    "kid_target_std": float("nan"),
                    "kid_target_status": f"unavailable: {exc}",
                }
            )
        except Exception as exc:
            raise MetricComputationError("KID computation failed; no fallback was used") from exc

    if should_compute_prdc:
        try:
            result.update(
                _compute_prdc(
                    real_features,
                    generated_features,
                    k=prdc_k,
                    batch_size=distance_batch_size,
                    device=evaluation_device,
                )
            )
            result["prdc_status"] = "ok"
        except ValueError as exc:
            result.update(
                {
                    "precision_target": float("nan"),
                    "recall_target": float("nan"),
                    "density_target": float("nan"),
                    "coverage_target": float("nan"),
                    "prdc_status": f"unavailable: {exc}",
                }
            )
        except Exception as exc:
            raise MetricComputationError("PRDC computation failed; no fallback was used") from exc

    if compute_inception_score:
        try:
            from torchmetrics.image.inception import InceptionScore

            metric = InceptionScore(normalize=False, splits=int(inception_score_splits)).to(evaluation_device)
            uint8_generated = _to_uint8(generated, image_input_range)
            for start in range(0, uint8_generated.shape[0], int(feature_batch_size)):
                metric.update(uint8_generated[start : start + int(feature_batch_size)].to(evaluation_device))
            score_mean, score_std = metric.compute()
            result.update(
                {
                    "inception_score_mean": float(score_mean),
                    "inception_score_std": float(score_std),
                    "inception_score_status": "ok",
                }
            )
        except Exception as exc:
            raise MetricComputationError("optional Inception Score computation failed; no fallback was used") from exc
    return result


__all__ = [
    "FEATURE_METRIC_SCHEMA_VERSION",
    "MetricBackendError",
    "MetricComputationError",
    "build_real_feature_cache_key",
    "cache_real_feature_stats",
    "compute_feature_metrics",
    "preflight_feature_metric_backend",
    "real_feature_cache_key",
]
