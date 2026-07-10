"""Deterministic and stochastic DDIM sampling."""

from __future__ import annotations

from typing import Sequence

import torch

from .ddpm import ImageDDPM, _randn


class ImageDDIM(ImageDDPM):
    """DDIM sampler supporting mathematically valid timestep respacing."""

    def __init__(self, *args, eta: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if float(eta) < 0:
            raise ValueError("DDIM eta must be non-negative")
        self.eta = float(eta)

    def _sampling_sequence(self, steps: int | None) -> torch.Tensor:
        requested = self.timesteps if steps is None else int(steps)
        if requested < 2 or requested > self.timesteps:
            raise ValueError(
                f"DDIM sampling steps must be in [2, {self.timesteps}], got {requested}; "
                "a one-step request would not define a valid respaced trajectory"
            )
        # Rounding an evenly spaced increasing grid is duplicate-free whenever
        # requested <= timesteps.  Reverse it for the denoising trajectory.
        sequence = torch.linspace(0, self.timesteps - 1, requested, device=self.device).round().long()
        if sequence.unique().numel() != requested:
            raise RuntimeError("internal DDIM respacing produced duplicate timesteps")
        return sequence.flip(0)

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
        """Generate with the DDIM update from Song et al. (2021)."""

        if len(shape) < 2 or int(shape[0]) < 1:
            raise ValueError(f"shape must include a positive batch dimension, got {tuple(shape)}")
        if y is not None and y.shape[0] != int(shape[0]):
            raise ValueError("label batch size does not match sample batch size")
        eta_value = self.eta if eta is None else float(eta)
        if eta_value < 0:
            raise ValueError("DDIM eta must be non-negative")
        sequence = self._sampling_sequence(steps)
        if initial_noise is None:
            x = _randn(shape, device=self.device, dtype=self.betas.dtype, generator=generator)
        else:
            if tuple(initial_noise.shape) != tuple(shape):
                raise ValueError("initial_noise shape does not match requested sample shape")
            x = initial_noise.to(device=self.device, dtype=self.betas.dtype).clone()

        was_training = bool(getattr(model, "training", False))
        model.eval()
        try:
            values = sequence.tolist()
            for index, timestep in enumerate(values):
                previous = values[index + 1] if index + 1 < len(values) else -1
                t = torch.full((int(shape[0]),), int(timestep), device=self.device, dtype=torch.long)
                alpha_bar_t = self.alpha_bars[int(timestep)].to(dtype=x.dtype)
                alpha_bar_previous = (
                    self.alpha_bars[int(previous)].to(dtype=x.dtype)
                    if previous >= 0
                    else torch.ones((), device=self.device, dtype=x.dtype)
                )
                epsilon = model(x, t, y)
                predicted_x0 = (x - (1.0 - alpha_bar_t).sqrt() * epsilon) / alpha_bar_t.sqrt()

                variance_factor = (
                    (1.0 - alpha_bar_previous) / (1.0 - alpha_bar_t)
                    * (1.0 - alpha_bar_t / alpha_bar_previous)
                ).clamp_min(0.0)
                sigma = eta_value * variance_factor.sqrt()
                epsilon_coefficient = (1.0 - alpha_bar_previous - sigma.square()).clamp_min(0.0).sqrt()
                x = alpha_bar_previous.sqrt() * predicted_x0 + epsilon_coefficient * epsilon
                if previous >= 0 and eta_value > 0:
                    x = x + sigma * _randn(
                        x.shape, device=x.device, dtype=x.dtype, generator=generator
                    )
        finally:
            if was_training:
                model.train()
        return x.clamp(-1.0, 1.0)
