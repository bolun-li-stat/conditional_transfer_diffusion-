"""Architecture-matched conditional denoiser used by both comparisons."""
from __future__ import annotations
import math
import torch
from torch import nn


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    emb = t.float()[:, None] * freq[None]
    out = torch.cat([emb.sin(), emb.cos()], dim=1)
    return torch.nn.functional.pad(out, (0, dim - out.shape[1]))


class ConditionalDenoiser(nn.Module):
    def __init__(self, d: int, num_classes: int, time_embedding_dim: int,
                 class_embedding_dim: int, hidden_width: int, hidden_layers: int) -> None:
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)
        blocks: list[nn.Module] = []
        width = d + time_embedding_dim + class_embedding_dim
        for _ in range(hidden_layers):
            blocks += [nn.Linear(width, hidden_width), nn.SiLU()]
            width = hidden_width
        blocks.append(nn.Linear(width, d))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, time_embedding(t, self.time_embedding_dim),
                                   self.class_embedding(labels.long())], dim=1))

    def shared_parameters(self) -> list[nn.Parameter]:
        return list(self.net.parameters())
