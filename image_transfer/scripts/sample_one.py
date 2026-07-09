from __future__ import annotations

import argparse
from pathlib import Path

import torch

from image_transfer.diffusion.ddpm import ImageDDPM
from image_transfer.models.unet import ImageUNet
from image_transfer.training.checkpointing import load_checkpoint
from image_transfer.utils.device import get_device
from image_transfer.utils.io import ensure_dir, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    device = get_device(args.device)
    image_size = int(cfg.get("image_size", 32))
    model = ImageUNet(
        image_size=image_size,
        base_channels=int(cfg.get("model", {}).get("base_channels", 64)),
        channel_mults=cfg.get("model", {}).get("channel_mults", [1, 2, 2, 4]),
        num_classes=args.num_classes if args.conditional else None,
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    diffusion = ImageDDPM(
        timesteps=int(cfg.get("diffusion", {}).get("timesteps", 1000)),
        schedule=cfg.get("diffusion", {}).get("schedule", "linear"),
        device=device,
    )
    labels = torch.full((args.num_samples,), args.label, dtype=torch.long, device=device) if args.conditional else None
    samples = diffusion.sample(model, (args.num_samples, 3, image_size, image_size), y=labels, steps=int(cfg.get("sampling_steps", diffusion.timesteps)))
    out = Path(args.out)
    ensure_dir(out.parent)
    torch.save(samples.cpu(), out)
    print(out)


if __name__ == "__main__":
    main()
