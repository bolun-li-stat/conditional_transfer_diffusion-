"""Forward diffusion utilities and a mathematically valid DDPM sampler."""

from __future__ import annotations

from typing import Sequence

import torch

from .schedules import make_beta_schedule


def _randn(
    shape: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Generate noise without falling back to the process-global RNG."""

    return torch.randn(tuple(shape), device=device, dtype=dtype, generator=generator)


class ImageDDPM:
    """Discrete DDPM process.

    The ancestral update implemented here is the adjacent-step DDPM update.
    Consequently, a DDPM request may not skip timesteps.  Use :class:`ImageDDIM`
    for a respaced sampler.
    """

    def __init__(self, timesteps: int = 1000, schedule: str = "linear", device: str | torch.device = "cpu") -> None:
        if int(timesteps) < 1:
            raise ValueError(f"timesteps must be positive, got {timesteps}")
        self.timesteps = int(timesteps)
        self.device = torch.device(device)
        betas = make_beta_schedule(schedule, self.timesteps).to(self.device)
        if betas.ndim != 1 or betas.numel() != self.timesteps:
            raise ValueError("beta schedule returned the wrong shape")
        if not bool(torch.all((betas > 0) & (betas < 1))):
            raise ValueError("all beta values must lie strictly between zero and one")
        self.schedule = schedule
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        alpha_bars_prev = torch.cat(
            [torch.ones(1, device=self.device, dtype=betas.dtype), self.alpha_bars[:-1]], dim=0
        )
        # q(x_{t-1} | x_t, x_0) variance from Ho et al. (2020).
        self.posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)
        self.posterior_variance[0] = 0.0

    @staticmethod
    def _extract(values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        if t.dtype != torch.long:
            t = t.long()
        return values.gather(0, t.to(values.device)).view(t.shape[0], *((1,) * (ndim - 1)))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``x_t`` and return both the corruption and the exact noise."""

        if t.ndim != 1 or t.shape[0] != x0.shape[0]:
            raise ValueError("t must be a length-batch vector")
        if bool(torch.any(t < 0)) or bool(torch.any(t >= self.timesteps)):
            raise ValueError("t contains an index outside the diffusion schedule")
        if noise is None:
            noise = _randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        elif noise.shape != x0.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} does not match x0 shape {tuple(x0.shape)}")
        alpha_bar = self._extract(self.alpha_bars, t, x0.ndim).to(dtype=x0.dtype, device=x0.device)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise, noise

    def loss(
        self,
        model,
        x0: torch.Tensor,
        y: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Standard epsilon-prediction objective with optional explicit RNG."""

        if t is None:
            t = torch.randint(
                0,
                self.timesteps,
                (x0.shape[0],),
                device=x0.device,
                generator=generator,
            )
        xt, epsilon = self.q_sample(x0, t, noise=noise, generator=generator)
        return torch.nn.functional.mse_loss(model(xt, t, y), epsilon, reduction=reduction)

    def _validate_sampling_request(self, shape: Sequence[int], steps: int | None) -> int:
        if len(shape) < 2 or int(shape[0]) < 1:
            raise ValueError(f"shape must include a positive batch dimension, got {tuple(shape)}")
        requested = self.timesteps if steps is None else int(steps)
        if requested != self.timesteps:
            raise ValueError(
                "DDPM uses adjacent posterior transitions and therefore requires "
                f"sampling steps == diffusion timesteps ({self.timesteps}); got {requested}. "
                "Use ImageDDIM for respaced sampling."
            )
        return requested

    @torch.no_grad()
    def sample(
        self,
        model,
        shape: Sequence[int],
        y: torch.Tensor | None = None,
        steps: int | None = None,
        *,
        eta: float | None = None,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples with the full ancestral DDPM chain.

        ``generator`` is deliberately explicit so sampling randomness is
        independent of all RNG consumed while training.  ``initial_noise`` is a
        convenience for common-random-number paired comparisons.
        """

        if eta is not None and float(eta) != 0.0:
            raise ValueError("eta is a DDIM parameter; DDPM only accepts eta=None or eta=0")
        self._validate_sampling_request(shape, steps)
        if y is not None and y.shape[0] != int(shape[0]):
            raise ValueError("label batch size does not match sample batch size")
        if initial_noise is None:
            x = _randn(shape, device=self.device, dtype=self.betas.dtype, generator=generator)
        else:
            if tuple(initial_noise.shape) != tuple(shape):
                raise ValueError("initial_noise shape does not match requested sample shape")
            x = initial_noise.to(device=self.device, dtype=self.betas.dtype).clone()

        was_training = bool(getattr(model, "training", False))
        model.eval()
        try:
            for timestep in range(self.timesteps - 1, -1, -1):
                t = torch.full((int(shape[0]),), timestep, device=self.device, dtype=torch.long)
                beta = self._extract(self.betas, t, x.ndim).to(dtype=x.dtype)
                alpha = self._extract(self.alphas, t, x.ndim).to(dtype=x.dtype)
                alpha_bar = self._extract(self.alpha_bars, t, x.ndim).to(dtype=x.dtype)
                epsilon = model(x, t, y)
                mean = (x - beta / (1.0 - alpha_bar).sqrt() * epsilon) / alpha.sqrt()
                if timestep > 0:
                    variance = self._extract(self.posterior_variance, t, x.ndim).to(dtype=x.dtype)
                    noise = _randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
                    x = mean + variance.clamp_min(0.0).sqrt() * noise
                else:
                    x = mean
        finally:
            if was_training:
                model.train()
        return x.clamp(-1.0, 1.0)
