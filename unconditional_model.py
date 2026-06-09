"""Unconditional MLP epsilon-prediction network for 100-D DDPM simulations."""
from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Return sinusoidal embeddings for integer diffusion steps in [0, T-1]."""
    if t.ndim == 0:
        t = t[None]
    half = dim // 2
    device = t.device
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1))
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class UnconditionalDenoiser(nn.Module):
    """epsilon_theta(x_t, t) -> R^d for target-only unconditional DDPM training."""

    def __init__(self, d: int = 100, time_embedding_dim: int = 64, hidden_width: int = 256, hidden_layers: int = 4) -> None:
        super().__init__()
        self.d = d
        self.time_embedding_dim = time_embedding_dim
        layers: list[nn.Module] = []
        in_dim = d + time_embedding_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_width), nn.SiLU()])
            in_dim = hidden_width
        layers.append(nn.Linear(hidden_width, d))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_time_embedding(t, self.time_embedding_dim)
        return self.net(torch.cat([x_t, emb], dim=-1))
