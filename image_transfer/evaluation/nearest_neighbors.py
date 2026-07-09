from __future__ import annotations

from pathlib import Path

import torch


def nearest_neighbor_indices(generated: torch.Tensor, real: torch.Tensor, k: int = 1) -> torch.Tensor:
    gen = generated.flatten(1).float().cpu()
    ref = real.flatten(1).float().cpu()
    distances = torch.cdist(gen, ref)
    return distances.topk(k, largest=False).indices


def make_nearest_neighbor_grid(generated: torch.Tensor, real: torch.Tensor, out_path: str | Path, max_items: int = 8) -> None:
    try:
        from torchvision.utils import make_grid, save_image
    except Exception:
        return
    indices = nearest_neighbor_indices(generated[:max_items], real, k=1).squeeze(1)
    pairs = []
    for i, idx in enumerate(indices.tolist()):
        pairs.extend([generated[i].cpu(), real[idx].cpu()])
    grid = make_grid(torch.stack(pairs), nrow=2, normalize=True, value_range=(-1, 1))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)
