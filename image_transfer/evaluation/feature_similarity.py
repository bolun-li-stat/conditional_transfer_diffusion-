from __future__ import annotations

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def average_auxiliary_similarity(target_dataset, aux_datasets: list, batch_size: int = 32, device="cpu") -> float:
    """Compute a lightweight pixel-feature cosine similarity fallback."""
    def mean_feature(dataset):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        total = None
        count = 0
        for x, _ in loader:
            feat = x.flatten(1).to(device).float().mean(dim=0)
            total = feat if total is None else total + feat
            count += 1
        return total / max(count, 1)
    if not aux_datasets:
        return float("nan")
    target = mean_feature(target_dataset)
    sims = []
    for aux in aux_datasets:
        sims.append(torch.nn.functional.cosine_similarity(target, mean_feature(aux), dim=0).item())
    return float(sum(sims) / len(sims))
