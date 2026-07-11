"""Validated image-model profiles, construction, hashing, and audit metadata."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .adm_unet import ADM_UNET_SCHEMA_VERSION, ADMUNet
from .unet import ImageUNet


MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "smoke_tiny": {
        "model_channels": 16, "channel_mults": [1, 2], "num_res_blocks": 1,
        "attention_resolutions": [], "num_head_channels": 16, "dropout": 0.0,
    },
    "pilot_small": {
        "model_channels": 40, "channel_mults": [1, 2, 3, 4], "num_res_blocks": 2,
        "attention_resolutions": [16], "num_head_channels": 40, "dropout": 0.1,
    },
    "main_default": {
        "model_channels": 64, "channel_mults": [1, 2, 3, 4], "num_res_blocks": 2,
        "attention_resolutions": [16, 8], "num_head_channels": 64, "dropout": 0.1,
    },
    "capacity_large": {
        "model_channels": 96, "channel_mults": [1, 2, 3, 4], "num_res_blocks": 2,
        "attention_resolutions": [16, 8], "num_head_channels": 96, "dropout": 0.1,
    },
}

ADM_FIELDS = {
    "architecture", "profile", "in_channels", "out_channels", "model_channels", "channel_mults",
    "num_res_blocks", "attention_resolutions", "num_head_channels", "dropout", "use_scale_shift_norm",
    "resblock_updown", "conv_resample", "zero_init_residual", "gradient_checkpointing",
    "class_conditioning", "class_dropout_probability", "architecture_schema_version", "image_size",
}
LEGACY_FIELDS = {
    "architecture", "profile", "in_channels", "out_channels", "base_channels", "channel_mults",
    "architecture_schema_version", "image_size",
}


def resolve_model_config(model_cfg: Mapping[str, Any] | None, *, image_size: int) -> dict[str, Any]:
    raw = dict(model_cfg or {})
    architecture = str(raw.get("architecture", "legacy_simple_unet"))
    if "architecture" not in raw:
        warnings.warn(
            "model.architecture is absent; mapping this legacy config to legacy_simple_unet",
            DeprecationWarning,
            stacklevel=2,
        )
    if architecture == "legacy_simple_unet":
        unknown = set(raw) - LEGACY_FIELDS
        if unknown:
            raise ValueError(f"unknown legacy model config fields: {sorted(unknown)}")
        return {
            "architecture": architecture,
            "architecture_schema_version": "legacy-1",
            "profile": str(raw.get("profile", "legacy")),
            "in_channels": int(raw.get("in_channels", 3)),
            "out_channels": int(raw.get("out_channels", raw.get("in_channels", 3))),
            "base_channels": int(raw.get("base_channels", 64)),
            "channel_mults": [int(value) for value in raw.get("channel_mults", [1, 2, 2, 4])],
            "image_size": int(image_size),
        }
    if architecture != "adm_unet":
        raise ValueError(f"unknown model architecture {architecture!r}")
    unknown = set(raw) - ADM_FIELDS
    if unknown:
        raise ValueError(f"unknown ADM model config fields: {sorted(unknown)}")
    profile = str(raw.get("profile", "main_default"))
    if profile not in MODEL_PROFILES:
        raise ValueError(f"unknown model profile {profile!r}; expected one of {sorted(MODEL_PROFILES)}")
    resolved = {
        "architecture": "adm_unet",
        "architecture_schema_version": ADM_UNET_SCHEMA_VERSION,
        "profile": profile,
        "in_channels": 3,
        "out_channels": 3,
        **MODEL_PROFILES[profile],
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "conv_resample": True,
        "zero_init_residual": True,
        "gradient_checkpointing": False,
        "class_conditioning": "embedding",
        "class_dropout_probability": 0.0,
        "image_size": int(image_size),
    }
    for key in ADM_FIELDS - {"architecture", "profile", "architecture_schema_version", "image_size"}:
        if key in raw:
            resolved[key] = raw[key]
    resolved["channel_mults"] = [int(value) for value in resolved["channel_mults"]]
    resolved["attention_resolutions"] = [int(value) for value in resolved["attention_resolutions"]]
    for key in ("in_channels", "out_channels", "model_channels", "num_res_blocks", "num_head_channels"):
        resolved[key] = int(resolved[key])
    resolved["dropout"] = float(resolved["dropout"])
    if resolved["class_conditioning"] != "embedding":
        raise ValueError("ADM U-Net supports class_conditioning='embedding' only")
    if float(resolved["class_dropout_probability"]) != 0.0:
        raise ValueError("class_dropout_probability must remain 0.0; classifier-free guidance is not implemented")
    available = {int(image_size) // (2**level) for level in range(len(resolved["channel_mults"]))}
    missing = set(resolved["attention_resolutions"]) - available
    if missing:
        raise ValueError(f"attention resolutions {sorted(missing)} do not occur in {sorted(available, reverse=True)}")
    for resolution, multiplier in zip(
        [int(image_size) // (2**level) for level in range(len(resolved["channel_mults"]))],
        resolved["channel_mults"],
    ):
        channels = resolved["model_channels"] * multiplier
        if resolution in resolved["attention_resolutions"] and channels % resolved["num_head_channels"]:
            raise ValueError(
                f"resolution {resolution} has {channels} channels, not divisible by "
                f"num_head_channels={resolved['num_head_channels']}"
            )
    return resolved


def model_config_hash(resolved: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(resolved), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_image_model(
    model_cfg: Mapping[str, Any] | None,
    *,
    image_size: int,
    conditional: bool,
    num_classes: int,
    model_seed: int,
) -> nn.Module:
    resolved = resolve_model_config(model_cfg, image_size=image_size)
    torch.manual_seed(int(model_seed))
    class_count = int(num_classes) if conditional else None
    if resolved["architecture"] == "legacy_simple_unet":
        model = ImageUNet(
            image_size=image_size,
            in_channels=resolved["in_channels"],
            base_channels=resolved["base_channels"],
            channel_mults=tuple(resolved["channel_mults"]),
            num_classes=class_count,
        )
    else:
        arguments = {
            key: value
            for key, value in resolved.items()
            if key not in {
                "architecture", "architecture_schema_version", "profile", "class_conditioning",
                "class_dropout_probability", "image_size",
            }
        }
        model = ADMUNet(
            image_size=image_size,
            num_classes=class_count,
            model_seed=model_seed,
            **arguments,
        )
    model.resolved_model_config = resolved
    model.model_config_hash = model_config_hash(resolved)
    return model


def model_parameter_metadata(model: nn.Module) -> dict[str, Any]:
    resolved = dict(getattr(model, "resolved_model_config", {}))
    conditioning_prefixes = ("class_embedding.", "class_emb.")
    conditioning = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(conditioning_prefixes)
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "architecture": resolved.get("architecture", "unknown"),
        "architecture_schema_version": resolved.get("architecture_schema_version", "unknown"),
        "architecture_profile": resolved.get("profile", "unknown"),
        "resolved_model_config": resolved,
        "model_config_hash": getattr(model, "model_config_hash", model_config_hash(resolved)),
        "model_parameter_count": int(total),
        "model_trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "backbone_parameter_count": int(total - conditioning),
        "conditioning_parameter_count": int(conditioning),
    }


__all__ = [
    "ADM_FIELDS", "MODEL_PROFILES", "build_image_model", "model_config_hash",
    "model_parameter_metadata", "resolve_model_config",
]
