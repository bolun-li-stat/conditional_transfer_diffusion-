"""Minimal DDPM epsilon-prediction process for vector-valued Gaussian data."""
from __future__ import annotations

from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from config import DiffusionConfig


class DDPM:
    """Shared DDPM forward/reverse process used by unconditional and conditional models."""

    def __init__(self, config: DiffusionConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.T = config.T
        self.betas = torch.linspace(config.beta_start, config.beta_end, config.T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        prev = torch.cat([torch.ones(1, device=device), self.alpha_bars[:-1]])
        self.alpha_bars_prev = prev
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.posterior_variance = self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = arr.gather(0, t.long())
        return out.reshape(-1, *([1] * (len(x_shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        return sqrt_ab * x0 + sqrt_om * noise, noise

    def epsilon_loss(self, model: nn.Module, x0: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        batch = x0.shape[0]
        t = torch.randint(0, self.T, (batch,), device=x0.device)
        x_t, eps = self.q_sample(x0, t)
        if labels is None:
            pred = model(x_t, t)
        else:
            pred = model(x_t, t, labels)
        return F.mse_loss(pred, eps)

    @torch.no_grad()
    def p_sample_step(self, model: nn.Module, x_t: torch.Tensor, t_index: int, labels: torch.Tensor | None = None) -> torch.Tensor:
        t = torch.full((x_t.shape[0],), t_index, device=x_t.device, dtype=torch.long)
        eps = model(x_t, t) if labels is None else model(x_t, t, labels)
        beta_t = self._extract(self.betas, t, x_t.shape)
        alpha_t = self._extract(self.alphas, t, x_t.shape)
        alpha_bar_t = self._extract(self.alpha_bars, t, x_t.shape)
        mean = (x_t - beta_t * eps / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)
        if t_index == 0:
            return mean
        if self.config.variance_type == "beta":
            var = beta_t
        else:
            var = self._extract(self.posterior_variance, t, x_t.shape)
        return mean + torch.sqrt(var) * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        n_samples: int,
        d: int,
        labels: torch.Tensor | None = None,
        batch_size: int = 1024,
        progress: Callable | None = None,
    ) -> torch.Tensor:
        model.eval()
        chunks: list[torch.Tensor] = []
        remaining = n_samples
        while remaining > 0:
            b = min(batch_size, remaining)
            x = torch.randn(b, d, device=self.device)
            lab = None
            if labels is not None:
                lab = labels[:b].to(self.device) if labels.numel() != 1 else labels.expand(b).to(self.device)
            iterator = range(self.T - 1, -1, -1)
            if progress is not None:
                iterator = progress(iterator)
            for t_idx in iterator:
                x = self.p_sample_step(model, x, t_idx, lab)
            chunks.append(x.cpu())
            remaining -= b
        return torch.cat(chunks, dim=0)
