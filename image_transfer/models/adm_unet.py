"""A compact pixel-space ADM/Improved-DDPM style U-Net.

The implementation is original to this repository and follows the architectural
ideas described by Dhariwal & Nichol (2021): residual scale/shift conditioning,
multi-head attention, residual resampling, and zero-initialized residual/output
projections. It intentionally implements epsilon prediction only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


ADM_UNET_SCHEMA_VERSION = "1.0"


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


def normalization(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def timestep_embedding(timesteps: torch.Tensor, dimension: int, max_period: int = 10_000) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
    )
    arguments = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimestepBlock(nn.Module):
    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class TimestepSequential(nn.Sequential, TimestepBlock):
    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = layer(x, embedding) if isinstance(layer, TimestepBlock) else layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels: int, *, use_conv: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, 3, padding=1) if use_conv else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError("Upsample channel mismatch")
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Downsample(nn.Module):
    def __init__(self, channels: int, *, use_conv: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.operation = (
            nn.Conv2d(channels, channels, 3, stride=2, padding=1)
            if use_conv
            else nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError("Downsample channel mismatch")
        return self.operation(x)


class ADMResBlock(TimestepBlock):
    def __init__(
        self,
        in_channels: int,
        embedding_channels: int,
        *,
        out_channels: int | None = None,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
        up: bool = False,
        down: bool = False,
        conv_resample: bool = True,
        zero_init_residual: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if up and down:
            raise ValueError("A residual block cannot upsample and downsample simultaneously")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels or in_channels)
        self.use_scale_shift_norm = bool(use_scale_shift_norm)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.in_norm = normalization(self.in_channels)
        self.in_conv = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)
        self.h_resample: nn.Module = nn.Identity()
        self.x_resample: nn.Module = nn.Identity()
        if up:
            self.h_resample = Upsample(self.in_channels, use_conv=conv_resample)
            self.x_resample = Upsample(self.in_channels, use_conv=conv_resample)
        elif down:
            self.h_resample = Downsample(self.in_channels, use_conv=conv_resample)
            self.x_resample = Downsample(self.in_channels, use_conv=conv_resample)
        embedding_outputs = 2 * self.out_channels if self.use_scale_shift_norm else self.out_channels
        self.embedding_projection = nn.Sequential(nn.SiLU(), nn.Linear(embedding_channels, embedding_outputs))
        self.out_norm = normalization(self.out_channels)
        output_conv = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        self.out_conv = zero_module(output_conv) if zero_init_residual else output_conv
        self.dropout = nn.Dropout(float(dropout))
        self.skip = (
            nn.Identity()
            if self.in_channels == self.out_channels
            else nn.Conv2d(self.in_channels, self.out_channels, 1)
        )

    def _forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.in_norm(x))
        h = self.h_resample(h)
        residual = self.x_resample(x)
        h = self.in_conv(h)
        conditioning = self.embedding_projection(embedding).to(h.dtype)[:, :, None, None]
        if self.use_scale_shift_norm:
            scale, shift = conditioning.chunk(2, dim=1)
            h = self.out_norm(h) * (1 + scale) + shift
        else:
            h = self.out_norm(h + conditioning)
        h = self.out_conv(self.dropout(F.silu(h)))
        return self.skip(residual) + h

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._forward, x, embedding, use_reentrant=False)
        return self._forward(x, embedding)


class ADMAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_head_channels: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if num_head_channels <= 0 or channels % num_head_channels:
            raise ValueError(
                f"attention channels={channels} must be divisible by num_head_channels={num_head_channels}"
            )
        self.channels = int(channels)
        self.num_heads = channels // num_head_channels
        self.head_channels = int(num_head_channels)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.projection = zero_module(nn.Conv1d(channels, channels, 1))

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x).reshape(batch, channels, height * width))
        q, k, v = qkv.chunk(3, dim=1)
        length = height * width
        q = q.reshape(batch * self.num_heads, self.head_channels, length).float()
        k = k.reshape(batch * self.num_heads, self.head_channels, length).float()
        v = v.reshape(batch * self.num_heads, self.head_channels, length).float()
        scale = self.head_channels ** -0.5
        weights = torch.softmax(torch.bmm(q.transpose(1, 2), k) * scale, dim=-1)
        attended = torch.bmm(v, weights.transpose(1, 2)).to(x.dtype)
        attended = attended.reshape(batch, channels, length)
        return x + self.projection(attended).reshape(batch, channels, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class ADMUNet(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 64,
        channel_mults: Iterable[int] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Iterable[int] = (16, 8),
        num_head_channels: int = 64,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = True,
        conv_resample: bool = True,
        zero_init_residual: bool = True,
        gradient_checkpointing: bool = False,
        num_classes: int | None = None,
        model_seed: int = 0,
        class_seed_offset: int = 1_000_003,
    ) -> None:
        super().__init__()
        channel_mults = tuple(int(value) for value in channel_mults)
        attention_resolutions = tuple(int(value) for value in attention_resolutions)
        if image_size < 8 or image_size & (image_size - 1):
            raise ValueError("image_size must be a power of two and at least 8")
        available_resolutions = tuple(image_size // (2**level) for level in range(len(channel_mults)))
        missing = sorted(set(attention_resolutions) - set(available_resolutions))
        if missing:
            raise ValueError(f"attention resolutions {missing} are absent from architecture {available_resolutions}")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.model_channels = int(model_channels)
        self.channel_mults = channel_mults
        self.num_res_blocks = int(num_res_blocks)
        self.attention_resolutions = attention_resolutions
        self.num_head_channels = int(num_head_channels)
        self.num_classes = num_classes
        embedding_channels = 4 * model_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        channels = model_channels
        self.input_blocks = nn.ModuleList([TimestepSequential(nn.Conv2d(in_channels, channels, 3, padding=1))])
        input_block_channels = [channels]
        resolution = image_size
        common = dict(
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm,
            conv_resample=conv_resample,
            zero_init_residual=zero_init_residual,
            gradient_checkpointing=gradient_checkpointing,
        )
        for level, multiplier in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                output_channels = multiplier * model_channels
                layers: list[nn.Module] = [
                    ADMResBlock(channels, embedding_channels, out_channels=output_channels, **common)
                ]
                channels = output_channels
                if resolution in attention_resolutions:
                    layers.append(
                        ADMAttentionBlock(
                            channels,
                            num_head_channels=num_head_channels,
                            gradient_checkpointing=gradient_checkpointing,
                        )
                    )
                self.input_blocks.append(TimestepSequential(*layers))
                input_block_channels.append(channels)
            if level != len(channel_mults) - 1:
                down: nn.Module = (
                    ADMResBlock(channels, embedding_channels, out_channels=channels, down=True, **common)
                    if resblock_updown
                    else Downsample(channels, use_conv=conv_resample)
                )
                self.input_blocks.append(TimestepSequential(down))
                input_block_channels.append(channels)
                resolution //= 2

        self.middle_block = TimestepSequential(
            ADMResBlock(channels, embedding_channels, **common),
            ADMAttentionBlock(
                channels,
                num_head_channels=num_head_channels,
                gradient_checkpointing=gradient_checkpointing,
            ),
            ADMResBlock(channels, embedding_channels, **common),
        )
        self.output_blocks = nn.ModuleList()
        for level, multiplier in list(enumerate(channel_mults))[::-1]:
            for block_index in range(num_res_blocks + 1):
                skip_channels = input_block_channels.pop()
                layers = [
                    ADMResBlock(
                        channels + skip_channels,
                        embedding_channels,
                        out_channels=model_channels * multiplier,
                        **common,
                    )
                ]
                channels = model_channels * multiplier
                if resolution in attention_resolutions:
                    layers.append(
                        ADMAttentionBlock(
                            channels,
                            num_head_channels=num_head_channels,
                            gradient_checkpointing=gradient_checkpointing,
                        )
                    )
                if level and block_index == num_res_blocks:
                    layers.append(
                        ADMResBlock(channels, embedding_channels, out_channels=channels, up=True, **common)
                        if resblock_updown
                        else Upsample(channels, use_conv=conv_resample)
                    )
                    resolution *= 2
                self.output_blocks.append(TimestepSequential(*layers))
        if input_block_channels:
            raise RuntimeError("internal ADM U-Net skip bookkeeping did not close")
        self.output_norm = normalization(channels)
        self.output_conv = zero_module(nn.Conv2d(channels, out_channels, 3, padding=1))

        # Construct conditioning last, under a separate RNG stream. Consequently
        # conditionality and class count cannot perturb any shared backbone tensor.
        self.class_embedding: nn.Embedding | None = None
        if num_classes is not None:
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(model_seed) + int(class_seed_offset))
                self.class_embedding = nn.Embedding(int(num_classes), embedding_channels)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected [B,{self.in_channels},H,W] input")
        embedding = self.time_mlp(timestep_embedding(timesteps, self.model_channels))
        if self.class_embedding is not None:
            if y is None:
                raise ValueError("Conditional ADMUNet requires labels")
            embedding = embedding + self.class_embedding(y)
        elif y is not None:
            raise ValueError("Unconditional ADMUNet does not accept labels")
        h = x.to(self.output_conv.weight.dtype)
        skips: list[torch.Tensor] = []
        for module in self.input_blocks:
            h = module(h, embedding)
            skips.append(h)
        h = self.middle_block(h, embedding)
        for module in self.output_blocks:
            skip = skips.pop()
            if skip.shape[-2:] != h.shape[-2:]:
                raise RuntimeError(
                    f"ADM U-Net skip mismatch {skip.shape[-2:]} != {h.shape[-2:]}; interpolation is forbidden"
                )
            h = module(torch.cat([h, skip], dim=1), embedding)
        if skips:
            raise RuntimeError("ADM U-Net left unused skip tensors")
        return self.output_conv(F.silu(self.output_norm(h)))


__all__ = [
    "ADM_UNET_SCHEMA_VERSION",
    "ADMAttentionBlock",
    "ADMResBlock",
    "ADMUNet",
    "Downsample",
    "Upsample",
    "timestep_embedding",
    "zero_module",
]
