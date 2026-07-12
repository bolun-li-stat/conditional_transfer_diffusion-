"""Small vector DDPM implementation."""
from __future__ import annotations
import torch
import torch.nn.functional as F


class DDPM:
    def __init__(self, T: int, beta_start: float, beta_end: float, device: torch.device) -> None:
        self.T, self.device = T, device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = self.alphas.cumprod(0)
        previous = torch.cat([torch.ones(1, device=device), self.alpha_bars[:-1]])
        self.posterior_variance = (self.betas * (1-previous) / (1-self.alpha_bars)).clamp_min(1e-20)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bars[t][:, None]
        return ab.sqrt() * x0 + (1-ab).sqrt() * noise

    def loss(self, model, x0: torch.Tensor, labels: torch.Tensor,
             generator: torch.Generator | None = None) -> torch.Tensor:
        t = torch.randint(self.T, (len(x0),), device=x0.device, generator=generator)
        noise = torch.randn(x0.shape, device=x0.device, generator=generator)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t, labels), noise)

    @torch.no_grad()
    def sample(self, model, n: int, d: int, label: int, seed: int) -> torch.Tensor:
        gen = torch.Generator(device=self.device).manual_seed(seed)
        x = torch.randn((n, d), device=self.device, generator=gen)
        labels = torch.full((n,), label, device=self.device, dtype=torch.long)
        for i in range(self.T-1, -1, -1):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            eps = model(x, t, labels)
            mean = (x - self.betas[i] * eps / torch.sqrt(1-self.alpha_bars[i])) / torch.sqrt(self.alphas[i])
            if i:
                noise = torch.randn(x.shape, device=self.device, generator=gen)
                x = mean + self.posterior_variance[i].sqrt() * noise
            else:
                x = mean
        return x.cpu()
