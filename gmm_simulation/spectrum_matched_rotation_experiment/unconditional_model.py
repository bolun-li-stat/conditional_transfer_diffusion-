"""Label-free denoiser and API adapter for the unconditional target baseline."""
from __future__ import annotations

import torch
from torch import nn

from conditional_model import time_embedding


class UnconditionalDenoiser(nn.Module):
    """A true unconditional epsilon predictor with no class parameters."""

    def __init__(self, d: int, time_embedding_dim: int, hidden_width: int,
                 hidden_layers: int) -> None:
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        blocks: list[nn.Module] = []
        width = d + time_embedding_dim
        for _ in range(hidden_layers):
            blocks += [nn.Linear(width, hidden_width), nn.SiLU()]
            width = hidden_width
        blocks.append(nn.Linear(width, d))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, time_embedding(t, self.time_embedding_dim)], dim=1))


class LabelIgnoringAdapter(nn.Module):
    """Expose the existing conditional DDPM API without label parameters."""

    def __init__(self, core: UnconditionalDenoiser) -> None:
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                labels: torch.Tensor | None = None) -> torch.Tensor:
        del labels
        return self.core(x, t)
