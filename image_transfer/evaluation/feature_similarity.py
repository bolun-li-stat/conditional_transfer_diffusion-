from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


class _FallbackFeature(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ((x.float().clamp(-1, 1) + 1.0) / 2.0)
        return torch.nn.functional.adaptive_avg_pool2d(x, (4, 4)).flatten(1)


class _ResNetFeature(nn.Module):
    def __init__(self, backbone: nn.Module, preprocess: nn.Module) -> None:
        super().__init__()
        self.preprocess = preprocess
        self.features = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ((x.float().clamp(-1, 1) + 1.0) / 2.0)
        x = self.preprocess(x)
        return self.features(x)


def build_feature_extractor(device="cpu") -> nn.Module:
    """Prefer ImageNet-pretrained ResNet50 features; fall back to pooled pixels."""
    try:
        from torchvision.models import ResNet50_Weights, resnet50
    except Exception:
        return _FallbackFeature().to(device).eval()
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    return _ResNetFeature(model, weights.transforms()).to(device).eval()


@torch.no_grad()
def dataset_mean_feature(dataset, extractor: nn.Module | None = None, batch_size: int = 32, device="cpu") -> torch.Tensor:
    extractor = extractor or build_feature_extractor(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total = None
    count = 0
    for x, _ in loader:
        x = x.to(device)
        feat = extractor(x).detach().float()
        total = feat.sum(dim=0) if total is None else total + feat.sum(dim=0)
        count += feat.shape[0]
    if total is None:
        return torch.zeros(1, device=device)
    return total / max(count, 1)


@torch.no_grad()
def average_auxiliary_similarity(target_dataset, aux_datasets: list, batch_size: int = 32, device="cpu") -> float:
    if not aux_datasets:
        return float("nan")
    extractor = build_feature_extractor(device)
    target = dataset_mean_feature(target_dataset, extractor=extractor, batch_size=batch_size, device=device)
    sims = []
    for aux in aux_datasets:
        aux_mean = dataset_mean_feature(aux, extractor=extractor, batch_size=batch_size, device=device)
        sims.append(torch.nn.functional.cosine_similarity(target, aux_mean, dim=0).item())
    return float(sum(sims) / len(sims)) if sims else float("nan")
