"""Memory-bounded precision/recall/density/coverage metrics."""

from __future__ import annotations

import math

import torch


def _validate_features(real_features: torch.Tensor, generated_features: torch.Tensor, k: int) -> None:
    if real_features.ndim != 2 or generated_features.ndim != 2:
        raise ValueError("PRDC features must be two-dimensional [samples, features]")
    if real_features.shape[1] != generated_features.shape[1]:
        raise ValueError("real and generated feature dimensions differ")
    if int(k) < 1:
        raise ValueError("PRDC k must be positive")
    if real_features.shape[0] <= int(k):
        raise ValueError(f"PRDC needs more than k={k} real samples")
    if generated_features.shape[0] <= int(k):
        raise ValueError(f"PRDC needs more than k={k} generated samples")


def _device(value: str | torch.device | None, features: torch.Tensor) -> torch.device:
    return torch.device(value) if value is not None else features.device


@torch.no_grad()
def kth_neighbor_radii(
    features: torch.Tensor,
    k: int,
    *,
    batch_size: int = 512,
    reference_batch_size: int | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return each sample's k-th non-self neighbor radius without an NxN matrix."""

    if features.ndim != 2 or features.shape[0] <= int(k):
        raise ValueError(f"need a [n,d] feature matrix with n > k={k}")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    reference_batch_size = int(reference_batch_size or batch_size)
    if reference_batch_size < 1:
        raise ValueError("reference_batch_size must be positive")
    compute_device = _device(device, features)
    cpu_features = features.detach().float().cpu()
    radii = torch.empty(cpu_features.shape[0], dtype=torch.float32)
    for query_start in range(0, cpu_features.shape[0], int(batch_size)):
        query_end = min(query_start + int(batch_size), cpu_features.shape[0])
        query = cpu_features[query_start:query_end].to(compute_device)
        best = torch.full((query.shape[0], int(k)), float("inf"), device=compute_device)
        for reference_start in range(0, cpu_features.shape[0], reference_batch_size):
            reference_end = min(reference_start + reference_batch_size, cpu_features.shape[0])
            reference = cpu_features[reference_start:reference_end].to(compute_device)
            distances = torch.cdist(query, reference)
            overlap_start = max(query_start, reference_start)
            overlap_end = min(query_end, reference_end)
            if overlap_start < overlap_end:
                global_indices = torch.arange(overlap_start, overlap_end, device=compute_device)
                distances[global_indices - query_start, global_indices - reference_start] = float("inf")
            best = torch.topk(torch.cat([best, distances], dim=1), k=int(k), largest=False).values
        radii[query_start:query_end] = best[:, -1].cpu()
    return radii


@torch.no_grad()
def compute_prdc(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    *,
    k: int = 5,
    batch_size: int = 512,
    reference_batch_size: int | None = None,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Compute standard PRDC metrics with bounded pairwise-distance memory."""

    _validate_features(real_features, generated_features, k)
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    reference_batch_size = int(reference_batch_size or batch_size)
    compute_device = _device(device, real_features)
    real = real_features.detach().float().cpu()
    generated = generated_features.detach().float().cpu()
    real_radii = kth_neighbor_radii(
        real, k, batch_size=batch_size, reference_batch_size=reference_batch_size, device=compute_device
    )
    generated_radii = kth_neighbor_radii(
        generated, k, batch_size=batch_size, reference_batch_size=reference_batch_size, device=compute_device
    )

    precision_hits = 0
    density_sum = 0.0
    # Min generated distance per real point is sufficient for coverage.  Recall
    # additionally checks whether a real point lies in any generated sphere.
    minimum_generated_distance = torch.full((real.shape[0],), float("inf"), dtype=torch.float32)
    recall_hits = torch.zeros(real.shape[0], dtype=torch.bool)

    for generated_start in range(0, generated.shape[0], int(batch_size)):
        generated_end = min(generated_start + int(batch_size), generated.shape[0])
        generated_batch = generated[generated_start:generated_end].to(compute_device)
        generated_batch_radii = generated_radii[generated_start:generated_end].to(compute_device)
        batch_precision = torch.zeros(generated_batch.shape[0], dtype=torch.bool, device=compute_device)
        batch_density = torch.zeros(generated_batch.shape[0], dtype=torch.float64, device=compute_device)
        for real_start in range(0, real.shape[0], reference_batch_size):
            real_end = min(real_start + reference_batch_size, real.shape[0])
            real_batch = real[real_start:real_end].to(compute_device)
            distances = torch.cdist(generated_batch, real_batch)
            real_batch_radii = real_radii[real_start:real_end].to(compute_device)
            inside_real = distances <= real_batch_radii.unsqueeze(0)
            batch_precision |= inside_real.any(dim=1)
            batch_density += inside_real.sum(dim=1).double()

            real_minimum = distances.min(dim=0).values.cpu()
            minimum_generated_distance[real_start:real_end] = torch.minimum(
                minimum_generated_distance[real_start:real_end], real_minimum
            )
            inside_generated = distances <= generated_batch_radii.unsqueeze(1)
            recall_hits[real_start:real_end] |= inside_generated.any(dim=0).cpu()
        precision_hits += int(batch_precision.sum().item())
        density_sum += float(batch_density.sum().item())

    coverage_hits = minimum_generated_distance <= real_radii
    return {
        "precision_target": precision_hits / generated.shape[0],
        "recall_target": float(recall_hits.float().mean()),
        "density_target": density_sum / (int(k) * generated.shape[0]),
        "coverage_target": float(coverage_hits.float().mean()),
    }


__all__ = ["compute_prdc", "kth_neighbor_radii"]
