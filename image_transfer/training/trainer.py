from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from image_transfer.diffusion.ddpm import ImageDDPM
from image_transfer.models.unet import ImageUNet
from image_transfer.training.ema import EMA
from image_transfer.training.checkpointing import save_checkpoint
from image_transfer.evaluation.denoising_loss import evaluate_denoising_bins
from image_transfer.utils.io import ensure_dir


def _write_log_row(path: Path, row: dict[str, float | int]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "train_loss", "validation_epsilon_mse"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_image_model(
    dataset,
    val_dataset,
    *,
    conditional: bool,
    num_classes: int,
    image_size: int,
    base_channels: int,
    channel_mults: list[int] | tuple[int, ...],
    timesteps: int,
    schedule: str,
    steps: int,
    batch_size: int,
    lr: float,
    device,
    precision: str = "fp32",
    ema_decay: float = 0.999,
    checkpoint_path: str | Path | None = None,
    train_log_path: str | Path | None = None,
    resume: bool = False,
    validation_interval: int = 100,
    num_workers: int = 0,
):
    device = torch.device(device)
    model = ImageUNet(
        image_size=image_size,
        base_channels=base_channels,
        channel_mults=tuple(channel_mults),
        num_classes=num_classes if conditional else None,
    ).to(device)
    diffusion = ImageDDPM(timesteps=timesteps, schedule=schedule, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ema = EMA(model, ema_decay)
    start_step = 0
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    if resume and checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        ema = EMA(model, ema_decay)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    iterator = iter(loader)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "amp" and device.type == "cuda"))
    final_loss = float("nan")
    val_mse = float("nan")
    start_time = time.time()
    train_log = Path(train_log_path) if train_log_path else None

    for step in range(start_step, max(start_step, steps)):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True) if conditional else None
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(precision == "amp" and device.type == "cuda")):
            loss = diffusion.loss(model, x, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        final_loss = float(loss.detach().cpu().item())
        current_step = step + 1
        if current_step % validation_interval == 0 or current_step == steps:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            denoise = evaluate_denoising_bins(ema.shadow, diffusion, val_loader, device, label=0 if conditional else None)
            val_mse = denoise["all"]
            if train_log:
                _write_log_row(train_log, {"step": current_step, "train_loss": final_loss, "validation_epsilon_mse": val_mse})
        if checkpoint_path and (current_step == steps or current_step % max(validation_interval, 1) == 0):
            save_checkpoint(checkpoint_path, ema.shadow, optimizer, current_step, {"conditional": conditional})

    train_seconds = time.time() - start_time
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    denoise = evaluate_denoising_bins(ema.shadow, diffusion, val_loader, device, label=0 if conditional else None)
    if checkpoint_path:
        save_checkpoint(checkpoint_path, ema.shadow, optimizer, steps, {"conditional": conditional})
    return ema.shadow, diffusion, {
        "final_train_loss": final_loss,
        "wallclock_train_seconds": train_seconds,
        "validation_epsilon_mse_target": denoise["all"],
        "validation_epsilon_mse_low_noise": denoise["low"],
        "validation_epsilon_mse_mid_noise": denoise["mid"],
        "validation_epsilon_mse_high_noise": denoise["high"],
    }
