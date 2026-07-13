"""Class-conditional MLP epsilon-prediction network for Gaussian-mixture DDPMs."""
from __future__ import annotations

import torch
from torch import nn

from unconditional_model import sinusoidal_time_embedding


class ConditionalDenoiser(nn.Module):
    """epsilon_theta(x_t, t, c) -> R^d with learned class embeddings."""

    def __init__(
        self,
        d: int = 100,
        num_classes: int = 8,
        time_embedding_dim: int = 64,
        class_embedding_dim: int = 32,
        hidden_width: int = 256,
        hidden_layers: int = 4,
    ) -> None:
        super().__init__()
        self.d = d
        self.time_embedding_dim = time_embedding_dim
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)
        layers: list[nn.Module] = []
        in_dim = d + time_embedding_dim + class_embedding_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_width), nn.SiLU()])
            in_dim = hidden_width
        layers.append(nn.Linear(hidden_width, d))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_embedding_dim)
        c_emb = self.class_embedding(c.long())
        return self.net(torch.cat([x_t, t_emb, c_emb], dim=-1))
