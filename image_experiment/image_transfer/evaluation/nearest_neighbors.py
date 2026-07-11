"""Batched nearest-neighbor and memorization diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch

from .feature_similarity import build_feature_extractor


@torch.no_grad()
def batched_features(extractor, images: torch.Tensor, *, batch_size: int, device="cpu") -> torch.Tensor:
    if int(batch_size) < 1:
        raise ValueError("feature batch_size must be positive")
    chunks = []
    for start in range(0, images.shape[0], int(batch_size)):
        batch = images[start : start + int(batch_size)].to(device).float()
        chunks.append(extractor(batch).detach().flatten(1).cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, 1)


# Preserve the prior private helper name.
_batched_features = batched_features


@torch.no_grad()
def nearest_neighbor_search_from_features(
    query_features: torch.Tensor,
    reference_features: torch.Tensor,
    *,
    k: int = 1,
    query_batch_size: int = 256,
    reference_batch_size: int = 1024,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest distances/indices without materializing a full matrix."""

    if query_features.ndim != 2 or reference_features.ndim != 2:
        raise ValueError("nearest-neighbor features must be [samples, dimensions]")
    if query_features.shape[1] != reference_features.shape[1]:
        raise ValueError("query and reference feature dimensions differ")
    if reference_features.shape[0] < int(k) or int(k) < 1:
        raise ValueError(f"reference set must contain at least k={k} samples")
    if int(query_batch_size) < 1 or int(reference_batch_size) < 1:
        raise ValueError("nearest-neighbor batch sizes must be positive")
    compute_device = torch.device(device)
    query = query_features.detach().float().cpu()
    reference = reference_features.detach().float().cpu()
    all_distances: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    for query_start in range(0, query.shape[0], int(query_batch_size)):
        query_batch = query[query_start : query_start + int(query_batch_size)].to(compute_device)
        best_distances = torch.full((query_batch.shape[0], int(k)), float("inf"), device=compute_device)
        best_indices = torch.full((query_batch.shape[0], int(k)), -1, dtype=torch.long, device=compute_device)
        for reference_start in range(0, reference.shape[0], int(reference_batch_size)):
            reference_batch = reference[reference_start : reference_start + int(reference_batch_size)].to(compute_device)
            distances = torch.cdist(query_batch, reference_batch)
            local_k = min(int(k), reference_batch.shape[0])
            local_distances, local_indices = distances.topk(local_k, largest=False)
            local_indices = local_indices + reference_start
            combined_distances = torch.cat([best_distances, local_distances], dim=1)
            combined_indices = torch.cat([best_indices, local_indices], dim=1)
            selection = combined_distances.topk(int(k), largest=False).indices
            best_distances = combined_distances.gather(1, selection)
            best_indices = combined_indices.gather(1, selection)
        all_distances.append(best_distances.cpu())
        all_indices.append(best_indices.cpu())
    if not all_distances:
        return torch.empty(0, int(k)), torch.empty(0, int(k), dtype=torch.long)
    return torch.cat(all_distances), torch.cat(all_indices)


@torch.no_grad()
def nearest_neighbor_search(
    generated: torch.Tensor,
    real: torch.Tensor,
    *,
    k: int = 1,
    device: str | torch.device = "cpu",
    feature_batch_size: int = 64,
    distance_batch_size: int = 256,
    reference_batch_size: int = 1024,
    extractor=None,
    strict_feature_extractor: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    extractor = extractor or build_feature_extractor(device, strict=strict_feature_extractor)
    generated_features = batched_features(extractor, generated, batch_size=feature_batch_size, device=device)
    real_features = batched_features(extractor, real, batch_size=feature_batch_size, device=device)
    return nearest_neighbor_search_from_features(
        generated_features,
        real_features,
        k=k,
        query_batch_size=distance_batch_size,
        reference_batch_size=reference_batch_size,
        device=device,
    )


@torch.no_grad()
def nearest_neighbor_indices(
    generated: torch.Tensor,
    real: torch.Tensor,
    k: int = 1,
    device="cpu",
    batch_size: int = 64,
) -> torch.Tensor:
    """Compatibility wrapper returning only indices."""

    _, indices = nearest_neighbor_search(
        generated,
        real,
        k=k,
        device=device,
        feature_batch_size=batch_size,
        distance_batch_size=batch_size,
    )
    return indices


def nearest_neighbor_statistics(
    minimum_distances: torch.Tensor,
    *,
    near_duplicate_threshold: float | None,
) -> dict[str, float | int]:
    values = minimum_distances.detach().double().flatten().cpu()
    if not len(values):
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "q05": float("nan"),
            "minimum": float("nan"),
            "near_duplicate_count": 0,
            "near_duplicate_rate": float("nan"),
        }
    result: dict[str, float | int] = {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "q05": float(torch.quantile(values, 0.05)),
        "minimum": float(values.min()),
    }
    if near_duplicate_threshold is None:
        result.update({"near_duplicate_count": 0, "near_duplicate_rate": float("nan")})
    else:
        count = int((values <= float(near_duplicate_threshold)).sum())
        result.update({"near_duplicate_count": count, "near_duplicate_rate": count / len(values)})
    return result


@torch.no_grad()
def calibrate_near_duplicate_threshold(
    validation_images: torch.Tensor,
    *,
    quantile: float = 0.01,
    device: str | torch.device = "cpu",
    feature_batch_size: int = 64,
    distance_batch_size: int = 256,
    reference_batch_size: int = 1024,
    extractor=None,
    strict_feature_extractor: bool = False,
) -> dict[str, float | int | str]:
    """Calibrate a near-duplicate cutoff from validation real-to-real distances.

    Every validation image is queried against the same validation set and its
    self match is discarded by taking the second nearest neighbor. Generated
    images and the final test split never participate in threshold selection.
    """

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("near-duplicate calibration quantile must lie strictly between zero and one")
    if validation_images.ndim != 4 or int(validation_images.shape[0]) < 2:
        raise ValueError("near-duplicate calibration requires at least two validation images")
    extractor = extractor or build_feature_extractor(device, strict=strict_feature_extractor)
    features = batched_features(
        extractor,
        validation_images,
        batch_size=feature_batch_size,
        device=device,
    )
    distances, _ = nearest_neighbor_search_from_features(
        features,
        features,
        k=2,
        query_batch_size=distance_batch_size,
        reference_batch_size=reference_batch_size,
        device=device,
    )
    leave_one_out = distances[:, 1].detach().double().cpu()
    threshold = float(torch.quantile(leave_one_out, float(quantile)))
    return {
        "near_duplicate_threshold": threshold,
        "near_duplicate_calibration_method": "validation_real_to_real_leave_one_out_quantile",
        "near_duplicate_calibration_split": "target_validation",
        "near_duplicate_calibration_quantile": float(quantile),
        "near_duplicate_calibration_num_images": int(validation_images.shape[0]),
        "near_duplicate_calibration_distance_mean": float(leave_one_out.mean()),
        "near_duplicate_calibration_distance_median": float(leave_one_out.median()),
    }


@torch.no_grad()
def compute_memorization_diagnostics(
    generated: torch.Tensor,
    reference_sets: Mapping[str, torch.Tensor],
    *,
    near_duplicate_threshold: float | None,
    device: str | torch.device = "cpu",
    feature_batch_size: int = 64,
    distance_batch_size: int = 256,
    reference_batch_size: int = 1024,
    extractor=None,
    strict_feature_extractor: bool = False,
) -> dict[str, float | int | str]:
    """Compare generated images to all four train/holdout reference sets.

    Expected keys are ``target_train``, ``target_eval``, ``auxiliary_train`` and
    ``auxiliary_eval``.  Any omitted/empty set is explicitly marked unavailable.
    """

    extractor = extractor or build_feature_extractor(device, strict=strict_feature_extractor)
    generated_features = batched_features(extractor, generated, batch_size=feature_batch_size, device=device)
    output: dict[str, float | int | str] = {
        "nearest_neighbor_feature_extractor": getattr(extractor, "extractor_name", type(extractor).__name__),
        "nearest_neighbor_feature_weights": getattr(extractor, "weights_name", "unknown"),
        "nearest_neighbor_feature_preprocessing": getattr(extractor, "preprocessing_name", "unknown"),
        "near_duplicate_threshold": float(near_duplicate_threshold) if near_duplicate_threshold is not None else "disabled",
    }
    means: dict[str, float] = {}
    for name in ("target_train", "target_eval", "auxiliary_train", "auxiliary_eval"):
        references = reference_sets.get(name)
        prefix = f"nearest_neighbor_{name}"
        if references is None or not int(references.shape[0]):
            output[f"{prefix}_status"] = "unavailable: empty reference set"
            for statistic in ("mean", "median", "q05", "minimum", "near_duplicate_rate"):
                output[f"{prefix}_{statistic}"] = float("nan")
            output[f"{prefix}_near_duplicate_count"] = 0
            continue
        reference_features = batched_features(
            extractor, references, batch_size=feature_batch_size, device=device
        )
        distances, _ = nearest_neighbor_search_from_features(
            generated_features,
            reference_features,
            k=1,
            query_batch_size=distance_batch_size,
            reference_batch_size=reference_batch_size,
            device=device,
        )
        statistics = nearest_neighbor_statistics(
            distances[:, 0], near_duplicate_threshold=near_duplicate_threshold
        )
        for statistic, value in statistics.items():
            output[f"{prefix}_{statistic}"] = value
        output[f"{prefix}_status"] = "ok"
        means[name] = float(statistics["mean"])
    if "target_train" in means and "target_eval" in means:
        # Negative values mean generated samples are closer to training data.
        output["nearest_neighbor_target_train_minus_eval_mean"] = means["target_train"] - means["target_eval"]
    else:
        output["nearest_neighbor_target_train_minus_eval_mean"] = float("nan")
    return output


def make_memorization_grid(
    generated: torch.Tensor,
    reference_sets: Mapping[str, torch.Tensor],
    out_path: str | Path,
    *,
    generated_indices: Sequence[int] | None = None,
    max_items: int = 8,
    device: str | torch.device = "cpu",
    feature_batch_size: int = 64,
    distance_batch_size: int = 256,
    reference_batch_size: int = 1024,
    extractor=None,
    strict_feature_extractor: bool = False,
) -> None:
    """Save a fixed-index grid: generated then each available reference set."""

    try:
        from torchvision.utils import make_grid, save_image
    except Exception:
        return
    if generated_indices is None:
        chosen = list(range(min(int(max_items), int(generated.shape[0]))))
    else:
        chosen = [int(index) for index in generated_indices[: int(max_items)]]
    if not chosen:
        return
    if min(chosen) < 0 or max(chosen) >= generated.shape[0]:
        raise IndexError("generated grid index is out of range")
    selected_generated = generated[chosen]
    extractor = extractor or build_feature_extractor(device, strict=strict_feature_extractor)
    generated_features = batched_features(
        extractor, selected_generated, batch_size=feature_batch_size, device=device
    )
    neighbor_images: dict[str, torch.Tensor] = {}
    for name in ("target_train", "target_eval", "auxiliary_train", "auxiliary_eval"):
        references = reference_sets.get(name)
        if references is None or not len(references):
            continue
        reference_features = batched_features(
            extractor, references, batch_size=feature_batch_size, device=device
        )
        _, indices = nearest_neighbor_search_from_features(
            generated_features,
            reference_features,
            k=1,
            query_batch_size=distance_batch_size,
            reference_batch_size=reference_batch_size,
            device=device,
        )
        neighbor_images[name] = references[indices[:, 0]]
    ordered_names = [
        name for name in ("target_train", "target_eval", "auxiliary_train", "auxiliary_eval") if name in neighbor_images
    ]
    rows = []
    for row_index, generated_image in enumerate(selected_generated):
        rows.append(generated_image.cpu())
        rows.extend(neighbor_images[name][row_index].cpu() for name in ordered_names)
    grid = make_grid(
        torch.stack(rows), nrow=1 + len(ordered_names), normalize=True, value_range=(-1, 1)
    )
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, destination)


def make_nearest_neighbor_grid(
    generated: torch.Tensor,
    target_real: torch.Tensor,
    aux_real: torch.Tensor | None,
    out_path: str | Path,
    max_items: int = 8,
    device="cpu",
    nn_batch_size: int = 64,
) -> None:
    """Compatibility grid for the former two-reference API."""

    references = {"target_eval": target_real}
    if aux_real is not None:
        references["auxiliary_eval"] = aux_real
    make_memorization_grid(
        generated,
        references,
        out_path,
        max_items=max_items,
        device=device,
        feature_batch_size=nn_batch_size,
        distance_batch_size=nn_batch_size,
    )


__all__ = [
    "batched_features",
    "calibrate_near_duplicate_threshold",
    "compute_memorization_diagnostics",
    "make_memorization_grid",
    "make_nearest_neighbor_grid",
    "nearest_neighbor_indices",
    "nearest_neighbor_search",
    "nearest_neighbor_search_from_features",
    "nearest_neighbor_statistics",
]
