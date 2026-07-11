"""Print and persist a reproducible image-model capacity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_transfer.models.adm_unet import ADMAttentionBlock, ADMResBlock
from image_transfer.models.model_factory import build_image_model, model_parameter_metadata
from image_transfer.utils.io import atomic_write_json, load_yaml


def inspect_config(config_path: str | Path, *, conditional: bool = True, num_classes: int = 2) -> dict:
    cfg = load_yaml(config_path)
    image_size = int(cfg.get("image_size", 32))
    model = build_image_model(
        cfg.get("model"),
        image_size=image_size,
        conditional=conditional,
        num_classes=num_classes,
        model_seed=0,
    )
    metadata = model_parameter_metadata(model)
    resolved = metadata["resolved_model_config"]
    multipliers = resolved.get("channel_mults", [])
    resolutions = [image_size // (2**level) for level in range(len(multipliers))]
    channels = [resolved.get("model_channels", resolved.get("base_channels", 0)) * value for value in multipliers]
    attention = [module for module in model.modules() if isinstance(module, ADMAttentionBlock)]
    result = {
        **metadata,
        "resolution_channels": dict(zip(map(str, resolutions), channels)),
        "resblock_count": sum(isinstance(module, ADMResBlock) for module in model.modules()),
        "attention_block_count": len(attention),
        "attention_resolutions": resolved.get("attention_resolutions", []),
        "attention_heads_by_channels": [
            {"channels": module.channels, "num_heads": module.num_heads, "head_channels": module.head_channels}
            for module in attention
        ],
        "estimated_activation_shapes": [
            [1, int(channel), int(resolution), int(resolution)]
            for channel, resolution in zip(channels, resolutions)
        ],
        "flops_status": "not_reported: a reliable operator-level FLOP counter is not bundled",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--num-classes", type=int, default=2)
    args = parser.parse_args()
    result = inspect_config(
        args.config,
        conditional=not args.unconditional,
        num_classes=args.num_classes,
    )
    if args.out:
        atomic_write_json(result, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
