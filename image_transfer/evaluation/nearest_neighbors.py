from __future__ import annotations

from pathlib import Path

import torch

from .feature_similarity import build_feature_extractor


@torch.no_grad()
def _batched_features(extractor, images: torch.Tensor, *, batch_size: int, device="cpu") -> torch.Tensor:
    chunks = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device).float()
        chunks.append(extractor(batch).cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, 1)


@torch.no_grad()
def nearest_neighbor_indices(generated: torch.Tensor, real: torch.Tensor, k: int = 1, device="cpu", batch_size: int = 64) -> torch.Tensor:
    extractor = build_feature_extractor(device)
    gen = _batched_features(extractor, generated, batch_size=batch_size, device=device)
    ref = _batched_features(extractor, real, batch_size=batch_size, device=device)
    distances = torch.cdist(gen, ref)
    return distances.topk(k, largest=False).indices


def make_nearest_neighbor_grid(
    generated: torch.Tensor,
    target_real: torch.Tensor,
    aux_real: torch.Tensor | None,
    out_path: str | Path,
    max_items: int = 8,
    device="cpu",
    nn_batch_size: int = 64,
) -> None:
    try:
        from torchvision.utils import make_grid, save_image
    except Exception:
        return
    target_idx = nearest_neighbor_indices(generated[:max_items], target_real, k=1, device=device, batch_size=nn_batch_size).squeeze(1)
    aux_idx = None
    if aux_real is not None and len(aux_real):
        aux_idx = nearest_neighbor_indices(generated[:max_items], aux_real, k=1, device=device, batch_size=nn_batch_size).squeeze(1)
    rows = []
    for i, idx in enumerate(target_idx.tolist()):
        rows.append(generated[i].cpu())
        rows.append(target_real[idx].cpu())
        if aux_idx is not None:
            rows.append(aux_real[aux_idx[i].item()].cpu())
    nrow = 3 if aux_idx is not None else 2
    grid = make_grid(torch.stack(rows), nrow=nrow, normalize=True, value_range=(-1, 1))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)
