"""Sample an image checkpoint with explicit, training-independent randomness."""

from __future__ import annotations

import argparse
from pathlib import Path

from image_transfer.diffusion import ImageDDIM, ImageDDPM
from image_transfer.models.model_factory import build_image_model
from image_transfer.scripts.train_one import _atomic_torch_save, sample_batched
from image_transfer.training.checkpointing import _torch_load, load_checkpoint
from image_transfer.utils.device import get_device
from image_transfer.utils.io import load_yaml


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
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--sampling-batch-size", type=int)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim-eta", type=float)
    parser.add_argument("--raw", action="store_true", help="sample raw rather than EMA weights")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = get_device(args.device)
    checkpoint_payload = _torch_load(args.checkpoint, map_location="cpu")
    checkpoint_metadata = checkpoint_payload.get("model_metadata") or {}
    model_cfg = checkpoint_metadata.get("resolved_model_config") or cfg.get("model")
    image_size = int(checkpoint_metadata.get("image_size", cfg.get("image_size", 32)))
    conditional = bool(checkpoint_metadata.get("conditional", args.conditional))
    num_classes = int(checkpoint_metadata.get("num_classes", args.num_classes))
    model = build_image_model(
        model_cfg,
        image_size=image_size,
        conditional=conditional,
        num_classes=max(num_classes, 1),
        model_seed=0,
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, use_ema=not args.raw)

    diffusion_cfg = cfg.get("diffusion", {})
    sampling_cfg = cfg.get("sampling", {})
    timesteps = int(diffusion_cfg.get("timesteps", 1000))
    schedule = str(diffusion_cfg.get("schedule", "linear"))
    sampler = str(args.sampler or cfg.get("sampler", sampling_cfg.get("sampler", "ddpm"))).lower()
    eta = float(args.ddim_eta if args.ddim_eta is not None else sampling_cfg.get("ddim_eta", 0.0))
    if sampler == "ddpm":
        diffusion = ImageDDPM(timesteps=timesteps, schedule=schedule, device=device)
    else:
        diffusion = ImageDDIM(timesteps=timesteps, schedule=schedule, device=device, eta=eta)
    steps = int(
        args.sampling_steps
        if args.sampling_steps is not None
        else sampling_cfg.get("steps", cfg.get("sampling_steps", timesteps))
    )
    batch_size = int(
        args.sampling_batch_size
        if args.sampling_batch_size is not None
        else sampling_cfg.get("batch_size", cfg.get("training", {}).get("batch_size", 32))
    )
    seed = int(args.sampling_seed if args.sampling_seed is not None else sampling_cfg.get("seed", 0))
    samples = sample_batched(
        diffusion,
        model,
        num_samples=args.num_samples,
        image_size=image_size,
        conditional=conditional,
        label=args.label,
        sampling_steps=steps,
        batch_size=batch_size,
        sampling_seed=seed,
        ddim_eta=eta,
    )
    destination = _atomic_torch_save(samples, Path(args.out))
    print(destination)


if __name__ == "__main__":
    main()
