from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset, TensorDataset

from image_transfer.diffusion.ddpm import ImageDDPM
from image_transfer.evaluation.denoising_loss import evaluate_denoising_bins
from image_transfer.models.model_factory import build_image_model, model_parameter_metadata
from image_transfer.models.unet import ImageUNet
from image_transfer.training.checkpointing import checkpoint_paths, load_training_checkpoint, save_checkpoint
from image_transfer.training.ema import EMA
from image_transfer.utils.io import ensure_dir, get_git_sha
from image_transfer.utils.seed import isolated_seed, preserve_rng_state, set_seed

TRAINING_PROTOCOLS = {"natural_compute_matched", "target_exposure_matched"}


class _RollingLoss:
    """Example-weighted rolling mean over a fixed number of optimizer steps."""

    def __init__(self, max_steps: int, state: Sequence[Sequence[float | int]] | None = None) -> None:
        if int(max_steps) < 1:
            raise ValueError("rolling_loss_window must be positive")
        self._values: deque[tuple[float, int]] = deque(maxlen=int(max_steps))
        for item in state or ():
            if len(item) != 2:
                raise ValueError("invalid rolling loss state")
            self._values.append((float(item[0]), int(item[1])))

    def add(self, total: float, count: int) -> None:
        self._values.append((float(total), int(count)))

    def mean(self) -> float:
        count = sum(item[1] for item in self._values)
        if count == 0:
            return float("nan")
        return float(sum(item[0] for item in self._values) / count)

    def state(self) -> list[list[float | int]]:
        return [[total, count] for total, count in self._values]


def _per_example_loss(loss: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Reduce a ``reduction='none'`` loss only over non-batch dimensions."""

    if loss.ndim == 0:
        raise ValueError("diffusion.loss(reduction='none') returned a scalar")
    if int(loss.shape[0]) != int(batch_size):
        raise ValueError("per-example loss has the wrong batch dimension")
    return loss if loss.ndim == 1 else loss.flatten(1).mean(dim=1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    selected = values[mask]
    return selected.mean() if selected.numel() else None


def _write_log_row(path: Path, row: dict[str, float | int]) -> None:
    ensure_dir(path.parent)
    fieldnames = ["step", "train_loss", "validation_epsilon_mse"]
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in fieldnames})


def _clean_resume_log(path: Path, start_step: int, validation_interval: int) -> None:
    """Drop a segment-only final validation row before continuing a run."""

    if not path.exists() or start_step % max(validation_interval, 1) == 0:
        return
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    retained = [row for row in rows if int(row.get("step", -1)) != start_step]
    if len(retained) == len(rows):
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "train_loss", "validation_epsilon_mse"])
        writer.writeheader()
        writer.writerows(retained)


def _stable_seed(seed: int, *parts: Any) -> int:
    digest = hashlib.sha256("|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _permuted(values: Sequence[int], *, seed: int, stream: Any, epoch: int) -> list[int]:
    if not values:
        raise ValueError("Cannot sample from an empty index pool")
    generator = torch.Generator().manual_seed(_stable_seed(seed, stream, epoch))
    order = torch.randperm(len(values), generator=generator).tolist()
    return [int(values[index]) for index in order]


def _stream_take(
    values: Sequence[int],
    *,
    position: int,
    count: int,
    seed: int,
    stream: Any,
    cache: dict[tuple[Any, int], list[int]],
) -> list[int]:
    selected: list[int] = []
    while len(selected) < count:
        epoch, offset = divmod(position, len(values))
        key = (stream, epoch)
        if key not in cache:
            cache[key] = _permuted(values, seed=seed, stream=stream, epoch=epoch)
        permutation = cache[key]
        take = min(count - len(selected), len(values) - offset)
        selected.extend(permutation[offset : offset + take])
        position += take
    return selected


def _balanced_examples_before(step: int, batch_size: int, class_position: int, num_classes: int) -> int:
    base, remainder = divmod(batch_size, num_classes)
    full_cycles, partial = divmod(step, num_classes)
    extras = full_cycles * remainder
    extras += sum(1 for previous in range(partial) if (class_position - previous) % num_classes < remainder)
    return step * base + extras


class DeterministicStepBatchSampler(Sampler[list[int]]):
    """A resume-stable shuffled stream indexed by optimizer step.

    Batch membership is a pure function of seed and global step.  DataLoader
    prefetch therefore cannot change the order saved at a checkpoint.  Batches
    wrap across shuffled epochs and always have the requested nominal size.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        start_step: int,
        end_step: int,
        seed: int,
        class_indices: dict[int, list[int]] | None = None,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("Training dataset is empty")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if not 0 <= start_step <= end_step:
            raise ValueError("Expected 0 <= start_step <= end_step")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.seed = int(seed)
        self.class_indices = {int(key): list(value) for key, value in (class_indices or {}).items() if value}

    def __len__(self) -> int:
        return self.end_step - self.start_step

    def __iter__(self) -> Iterator[list[int]]:
        cache: dict[tuple[Any, int], list[int]] = {}
        if not self.class_indices:
            values = list(range(self.dataset_size))
            for step in range(self.start_step, self.end_step):
                yield _stream_take(
                    values,
                    position=step * self.batch_size,
                    count=self.batch_size,
                    seed=self.seed,
                    stream="pooled",
                    cache=cache,
                )
            return

        classes = sorted(self.class_indices)
        num_classes = len(classes)
        base, remainder = divmod(self.batch_size, num_classes)
        for step in range(self.start_step, self.end_step):
            batch: list[int] = []
            for class_position, class_id in enumerate(classes):
                extra = int((class_position - step) % num_classes < remainder)
                count = base + extra
                if count == 0:
                    continue
                position = _balanced_examples_before(step, self.batch_size, class_position, num_classes)
                batch.extend(
                    _stream_take(
                        self.class_indices[class_id],
                        position=position,
                        count=count,
                        seed=self.seed,
                        stream=("class", class_id),
                        cache=cache,
                    )
                )
            generator = torch.Generator().manual_seed(_stable_seed(self.seed, "batch", step))
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[index] for index in order]


def _dataset_labels(dataset: Dataset) -> list[int]:
    """Extract labels without loading images when common dataset metadata exists."""

    if isinstance(dataset, Subset):
        parent = _dataset_labels(dataset.dataset)
        return [int(parent[index]) for index in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        labels: list[int] = []
        for child in dataset.datasets:
            labels.extend(_dataset_labels(child))
        return labels
    if isinstance(dataset, TensorDataset) and len(dataset.tensors) >= 2:
        return [int(value) for value in dataset.tensors[1].detach().cpu().tolist()]
    targets = getattr(dataset, "targets", None)
    if targets is not None and len(targets) == len(dataset):
        return [int(value) for value in targets]

    wrapped = getattr(dataset, "dataset", None)
    if isinstance(wrapped, Dataset):
        parent = _dataset_labels(wrapped)
        mapping = getattr(dataset, "label_to_new", None) or getattr(dataset, "old_to_new", None)
        if mapping is not None:
            return [int(mapping[int(value)]) for value in parent]
        if len(parent) == len(dataset):
            return parent

    # Last-resort fallback for custom/FakeData datasets.  Preserve RNG state so
    # label discovery cannot alter model initialization or diffusion noise.
    with preserve_rng_state():
        return [int(dataset[index][1]) for index in range(len(dataset))]


def _split_target_auxiliary(dataset: Dataset, target_label: int = 0) -> tuple[Dataset, Dataset | None]:
    labels = _dataset_labels(dataset)
    target_indices = [index for index, label in enumerate(labels) if label == target_label]
    auxiliary_indices = [index for index, label in enumerate(labels) if label != target_label]
    if not target_indices:
        raise ValueError(f"No examples with target label {target_label} were found")
    return Subset(dataset, target_indices), Subset(dataset, auxiliary_indices) if auxiliary_indices else None


def _balanced_class_indices(dataset: Dataset) -> dict[int, list[int]]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(_dataset_labels(dataset)):
        by_class[int(label)].append(index)
    return dict(by_class)


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    start_step: int,
    end_step: int,
    seed: int,
    num_workers: int,
    balanced: bool = False,
) -> DataLoader:
    class_indices = _balanced_class_indices(dataset) if balanced else None
    sampler = DeterministicStepBatchSampler(
        len(dataset),
        batch_size,
        start_step=start_step,
        end_step=end_step,
        seed=seed,
        class_indices=class_indices,
    )
    # A dedicated generator prevents DataLoader iterator construction from
    # advancing the diffusion/training RNG stream.
    loader_generator = torch.Generator().manual_seed(_stable_seed(seed, "loader"))
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        generator=loader_generator,
        persistent_workers=bool(num_workers > 0),
    )


def _new_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - old PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _normalized_denoising_metrics(values: dict[str, Any]) -> dict[str, float]:
    def value(*names: str) -> float:
        for name in names:
            if name in values:
                return float(values[name])
        return float("nan")

    return {
        "all": value("all", "overall", "validation_epsilon_mse_target"),
        "low": value("low", "validation_epsilon_mse_low_noise"),
        "mid": value("mid", "validation_epsilon_mse_mid_noise"),
        "high": value("high", "validation_epsilon_mse_high_noise"),
        "standard_error": value("standard_error", "validation_epsilon_mse_standard_error"),
        "num_validation_images": value("num_validation_images"),
        "num_corruptions": value("num_corruptions"),
    }


def _evaluate_validation(
    model,
    diffusion,
    val_dataset,
    *,
    device,
    conditional: bool,
    batch_size: int,
    num_workers: int,
    validation_evaluator: Callable[..., dict[str, Any]] | None,
    validation_corruption_bank: Any | None,
) -> dict[str, float]:
    # Fixed banks should already be independent; preserving state also protects
    # exact resumes when a legacy/random evaluator is supplied.
    with preserve_rng_state():
        loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if validation_evaluator is None and validation_corruption_bank is not None:
            from image_transfer.evaluation.corruption_bank import evaluate_corruption_bank

            values = evaluate_corruption_bank(
                model,
                diffusion,
                val_dataset,
                validation_corruption_bank,
                device,
                label=0 if conditional else None,
                batch_size=batch_size,
                metric_prefix="validation",
            )
        elif validation_evaluator is not None:
            try:
                values = validation_evaluator(
                    model,
                    diffusion,
                    loader,
                    device,
                    label=0 if conditional else None,
                    corruption_bank=validation_corruption_bank,
                )
            except TypeError:
                values = validation_evaluator(model, diffusion, loader, device, label=0 if conditional else None)
        else:
            try:
                values = evaluate_denoising_bins(
                    model,
                    diffusion,
                    loader,
                    device,
                    label=0 if conditional else None,
                    corruption_bank=validation_corruption_bank,
                )
            except TypeError:
                values = evaluate_denoising_bins(model, diffusion, loader, device, label=0 if conditional else None)
    return _normalized_denoising_metrics(values)


def _protocol_counters(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    by_class = metadata.get("auxiliary_examples_seen_by_class", {})
    if isinstance(by_class, str):
        try:
            by_class = json.loads(by_class)
        except json.JSONDecodeError:
            by_class = {}
    return {
        "optimizer_steps": int(metadata.get("optimizer_steps", 0)),
        "target_examples_seen": int(metadata.get("target_examples_seen", 0)),
        "auxiliary_examples_seen": int(metadata.get("auxiliary_examples_seen", 0)),
        "auxiliary_examples_seen_by_class": {str(key): int(value) for key, value in dict(by_class).items()},
    }


def train_image_model(
    dataset,
    val_dataset,
    *,
    conditional: bool,
    num_classes: int,
    image_size: int,
    base_channels: int | None = None,
    channel_mults: list[int] | tuple[int, ...] | None = None,
    model_cfg: dict[str, Any] | None = None,
    timesteps: int,
    schedule: str,
    steps: int,
    batch_size: int,
    lr: float,
    optimizer_name: str = "adamw",
    optimizer_betas: tuple[float, float] = (0.9, 0.999),
    optimizer_eps: float = 1.0e-8,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = None,
    device,
    precision: str = "fp32",
    ema_decay: float = 0.999,
    checkpoint_path: str | Path | None = None,
    train_log_path: str | Path | None = None,
    resume: bool = False,
    validation_interval: int = 100,
    checkpoint_interval: int | None = None,
    rolling_loss_window: int = 100,
    num_workers: int = 0,
    # Rigorous extensions; all are optional for old callers.
    target_dataset=None,
    auxiliary_dataset=None,
    training_protocol: str = "natural_compute_matched",
    target_batch_size: int | None = None,
    auxiliary_batch_size: int | None = None,
    auxiliary_loss_weight: float = 1.0,
    model_initialization_seed: int | None = None,
    training_seed: int | None = None,
    validation_corruption_bank: Any | None = None,
    validation_evaluator: Callable[..., dict[str, Any]] | None = None,
    last_checkpoint_path: str | Path | None = None,
    best_checkpoint_path: str | Path | None = None,
    best_validation_metric: float | None = None,
    config_hash: str | None = None,
    manifest_hash: str | None = None,
    git_sha: str | None = None,
    checkpoint_provenance: Mapping[str, Any] | None = None,
    deterministic_cpu: bool = False,
):
    """Train an image diffusion model under one of two explicit protocols.

    ``natural_compute_matched`` uses one pooled batch of ``batch_size`` each
    optimizer step. ``target_exposure_matched`` uses an independent target
    batch plus a class-balanced auxiliary batch and optimizes
    ``target_loss + auxiliary_loss_weight * auxiliary_loss``.  In both modes,
    counters report examples actually consumed rather than nominal dataset
    sizes.
    """

    if training_protocol not in TRAINING_PROTOCOLS:
        raise ValueError(f"Unknown training protocol {training_protocol!r}; expected one of {sorted(TRAINING_PROTOCOLS)}")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if auxiliary_loss_weight < 0:
        raise ValueError("auxiliary_loss_weight must be non-negative")
    if int(validation_interval) < 1:
        raise ValueError("validation_interval must be positive")
    resolved_checkpoint_interval = int(checkpoint_interval or validation_interval)
    if resolved_checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if int(rolling_loss_window) < 1:
        raise ValueError("rolling_loss_window must be positive")
    if int(num_workers) > 0:
        warnings.warn(
            "num_workers > 0: checkpoints restore the deterministic sampler position, but augmentation "
            "worker RNG/prefetch state is not captured, so resume is not guaranteed to be bitwise identical",
            RuntimeWarning,
            stacklevel=2,
        )
    device = torch.device(device)

    def build_model():
        if model_cfg is None:
            # Historical Python callers retain the old constructor contract;
            # all YAML-driven runs go through the validated model factory.
            return ImageUNet(
                image_size=image_size,
                base_channels=int(base_channels or 64),
                channel_mults=tuple(channel_mults or [1, 2, 2, 4]),
                num_classes=num_classes if conditional else None,
            ).to(device)
        return build_image_model(
            model_cfg,
            image_size=image_size,
            conditional=conditional,
            num_classes=num_classes,
            model_seed=int(model_initialization_seed or 0),
        ).to(device)

    if model_initialization_seed is None:
        model = build_model()
    else:
        with isolated_seed(model_initialization_seed, deterministic=deterministic_cpu and device.type == "cpu"):
            model = build_model()
    diffusion = ImageDDPM(timesteps=timesteps, schedule=schedule, device=device)
    if str(optimizer_name).lower() != "adamw":
        raise ValueError("Only optimizer.name='adamw' is currently supported")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        betas=tuple(float(value) for value in optimizer_betas),
        eps=float(optimizer_eps),
        weight_decay=float(weight_decay),
    )
    ema = EMA(model, ema_decay)
    architecture_metadata = model_parameter_metadata(model)
    architecture_metadata.update(
        {
            "conditional": bool(conditional),
            "num_classes": int(num_classes if conditional else 0),
            "image_size": int(image_size),
            "in_channels": int(architecture_metadata["resolved_model_config"].get("in_channels", 3)),
            "out_channels": int(architecture_metadata["resolved_model_config"].get("out_channels", 3)),
        }
    )
    scaler = _new_grad_scaler(precision == "amp" and device.type == "cuda")

    last_path, best_path, legacy_alias = checkpoint_paths(
        checkpoint_path, last_checkpoint_path=last_checkpoint_path, best_checkpoint_path=best_checkpoint_path
    )
    resume_path = None
    if resume:
        for candidate in (last_path, Path(checkpoint_path) if checkpoint_path is not None else None):
            if candidate is not None and candidate.exists():
                resume_path = candidate
                break
        if resume_path is None:
            raise FileNotFoundError(f"Resume requested but no checkpoint exists at {last_path or checkpoint_path}")

    start_step = 0
    counters = _protocol_counters()
    cumulative_train_seconds = 0.0
    cumulative_optimizer_seconds = 0.0
    saved_rolling_loss_state: dict[str, Any] = {}
    current_best = float("inf") if best_validation_metric is None else float(best_validation_metric)
    if resume_path is not None:
        checkpoint = load_training_checkpoint(
            resume_path,
            model,
            ema,
            optimizer,
            scaler,
            map_location=device,
            expected_config_hash=config_hash,
            expected_manifest_hash=manifest_hash,
            expected_model_config_hash=architecture_metadata["model_config_hash"],
            expected_architecture=architecture_metadata["architecture"],
            restore_rng=True,
        )
        saved_protocol = checkpoint.get("training_protocol")
        if saved_protocol not in {None, training_protocol}:
            raise ValueError(f"Checkpoint protocol {saved_protocol!r} does not match {training_protocol!r}")
        start_step = int(checkpoint.get("step", 0))
        saved_protocol_metadata = checkpoint.get("training_protocol_metadata") or {}
        counters = _protocol_counters(saved_protocol_metadata)
        cumulative_train_seconds = float(saved_protocol_metadata.get("wallclock_train_seconds", 0.0))
        cumulative_optimizer_seconds = float(saved_protocol_metadata.get("optimizer_compute_seconds", 0.0))
        saved_rolling_loss_state = dict(saved_protocol_metadata.get("rolling_loss_state") or {})
        saved_best = checkpoint.get("best_validation_metric")
        if saved_best is not None:
            current_best = float(saved_best)
    elif training_seed is not None:
        set_seed(training_seed, deterministic=deterministic_cpu and device.type == "cpu")

    if deterministic_cpu and device.type == "cpu":
        torch.use_deterministic_algorithms(True)

    if steps < start_step:
        raise ValueError(f"Requested {steps} total steps but checkpoint is already at step {start_step}")

    train_log = Path(train_log_path) if train_log_path else None
    if resume and train_log is not None:
        _clean_resume_log(train_log, start_step, validation_interval)

    rolling_losses = {
        name: _RollingLoss(int(rolling_loss_window), saved_rolling_loss_state.get(name))
        for name in ("pooled", "target", "auxiliary")
    }

    sampler_seed = int(training_seed if training_seed is not None else torch.initial_seed())
    if training_protocol == "target_exposure_matched":
        if target_dataset is None:
            target_dataset, inferred_auxiliary = _split_target_auxiliary(dataset)
            if auxiliary_dataset is None:
                auxiliary_dataset = inferred_auxiliary
        if target_dataset is None or len(target_dataset) == 0:
            raise ValueError("target_exposure_matched requires a non-empty target dataset")
        target_bs = int(target_batch_size or batch_size)
        aux_bs = int(auxiliary_batch_size or batch_size) if auxiliary_dataset is not None and len(auxiliary_dataset) else 0
        if auxiliary_dataset is not None and len(auxiliary_dataset) and not conditional:
            raise ValueError("An unconditional model cannot train on labeled auxiliary classes")
        target_loader = _make_loader(
            target_dataset,
            batch_size=target_bs,
            start_step=start_step,
            end_step=steps,
            seed=_stable_seed(sampler_seed, "target"),
            num_workers=num_workers,
        )
        auxiliary_loader = (
            _make_loader(
                auxiliary_dataset,
                batch_size=aux_bs,
                start_step=start_step,
                end_step=steps,
                seed=_stable_seed(sampler_seed, "auxiliary"),
                num_workers=num_workers,
                balanced=True,
            )
            if aux_bs
            else None
        )
        target_iterator = iter(target_loader)
        auxiliary_iterator = iter(auxiliary_loader) if auxiliary_loader is not None else None
        pooled_iterator = None
    else:
        target_bs = int(target_batch_size or batch_size)
        aux_bs = 0
        pooled_loader = _make_loader(
            dataset,
            batch_size=batch_size,
            start_step=start_step,
            end_step=steps,
            seed=_stable_seed(sampler_seed, "pooled"),
            num_workers=num_workers,
        )
        pooled_iterator = iter(pooled_loader)
        target_iterator = None
        auxiliary_iterator = None

    final_loss = float("nan")
    final_pooled_loss = float("nan")
    final_target_loss = float("nan")
    final_auxiliary_loss = float("nan")
    final_pooled_batch_size = 0
    final_target_batch_size = 0
    final_auxiliary_batch_size = 0
    final_grad_norm = float("nan")
    gradient_clipping_count = 0
    denoise = {key: float("nan") for key in ("all", "low", "mid", "high", "standard_error", "num_validation_images", "num_corruptions")}
    last_validation_step = -1
    start_time = time.time()
    resolved_git_sha = git_sha or get_git_sha()

    def protocol_metadata(current_step: int) -> dict[str, Any]:
        total = counters["target_examples_seen"] + counters["auxiliary_examples_seen"]
        elapsed = float(cumulative_train_seconds + (time.time() - start_time))
        optimizer_seconds = float(cumulative_optimizer_seconds)
        return {
            **counters,
            "optimizer_steps": int(current_step),
            "total_examples_seen": int(total),
            "effective_target_fraction": float(counters["target_examples_seen"] / total) if total else float("nan"),
            "target_batch_size": target_bs if training_protocol == "target_exposure_matched" else batch_size,
            "auxiliary_batch_size": aux_bs,
            "auxiliary_loss_weight": float(auxiliary_loss_weight),
            "wallclock_train_seconds": elapsed,
            "optimizer_compute_seconds": optimizer_seconds,
            "optimizer_steps_per_second": float(current_step / elapsed) if elapsed > 0 else float("nan"),
            "samples_per_second": float(total / optimizer_seconds) if optimizer_seconds > 0 else float("nan"),
            "images_processed_per_second": float(total / optimizer_seconds) if optimizer_seconds > 0 else float("nan"),
            "wallclock_images_per_second": float(total / elapsed) if elapsed > 0 else float("nan"),
            "actual_pooled_batch_size": int(final_pooled_batch_size),
            "actual_target_batch_size": int(final_target_batch_size),
            "actual_auxiliary_batch_size": int(final_auxiliary_batch_size),
            "rolling_loss_window_steps": int(rolling_loss_window),
            "rolling_loss_state": {name: values.state() for name, values in rolling_losses.items()},
            "resume_bitwise_identity_guaranteed": bool(int(num_workers) == 0 and device.type == "cpu"),
            "checkpoint_interval": int(resolved_checkpoint_interval),
            "validation_interval": int(validation_interval),
        }

    def save_training_state(
        path: Path | None,
        current_step: int,
        *,
        best_metric_override: float | None = None,
    ) -> None:
        if path is None:
            return
        save_checkpoint(
            path,
            model,
            optimizer,
            current_step,
            {"conditional": conditional},
            ema_model=ema,
            scaler=scaler,
            best_validation_metric=(
                best_metric_override
                if best_metric_override is not None
                else (None if math.isinf(current_best) else current_best)
            ),
            config_hash=config_hash,
            manifest_hash=manifest_hash,
            git_sha=resolved_git_sha,
            training_protocol=training_protocol,
            protocol_metadata=protocol_metadata(current_step),
            data_state={"sampler_seed": sampler_seed, "global_step": current_step, "num_workers": num_workers},
            model_metadata=architecture_metadata,
            provenance=checkpoint_provenance,
        )

    for step in range(start_step, steps):
        optimizer_step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if training_protocol == "natural_compute_matched":
            assert pooled_iterator is not None
            x, labels = next(pooled_iterator)
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            y = labels if conditional else None
            with torch.amp.autocast(device_type="cuda", enabled=(precision == "amp" and device.type == "cuda")):
                losses = _per_example_loss(
                    diffusion.loss(model, x, y, reduction="none"),
                    int(x.shape[0]),
                )
                loss = losses.mean()
            target_mask = torch.ones_like(labels, dtype=torch.bool) if not conditional else labels == 0
            auxiliary_mask = ~target_mask
            target_loss = _masked_mean(losses, target_mask)
            auxiliary_loss = _masked_mean(losses, auxiliary_mask)
            pooled_loss = loss
            target_count = int(target_mask.sum().item())
            auxiliary_count = int(auxiliary_mask.sum().item())
            counters["target_examples_seen"] += target_count
            counters["auxiliary_examples_seen"] += auxiliary_count
            if conditional and auxiliary_count:
                unique, counts = labels[labels != 0].detach().cpu().unique(return_counts=True)
                for class_id, count in zip(unique.tolist(), counts.tolist()):
                    key = str(int(class_id))
                    counters["auxiliary_examples_seen_by_class"][key] = counters["auxiliary_examples_seen_by_class"].get(key, 0) + int(count)
            rolling_losses["pooled"].add(float(losses.detach().sum().cpu()), int(losses.numel()))
            rolling_losses["target"].add(
                float(losses[target_mask].detach().sum().cpu()),
                target_count,
            )
            rolling_losses["auxiliary"].add(
                float(losses[auxiliary_mask].detach().sum().cpu()),
                auxiliary_count,
            )
            final_pooled_batch_size = int(losses.numel())
            final_target_batch_size = target_count
            final_auxiliary_batch_size = auxiliary_count
        else:
            assert target_iterator is not None
            target_x, target_labels = next(target_iterator)
            target_x = target_x.to(device, non_blocking=True)
            target_labels = target_labels.to(device, non_blocking=True)
            target_y = target_labels if conditional else None
            with torch.amp.autocast(device_type="cuda", enabled=(precision == "amp" and device.type == "cuda")):
                target_losses = _per_example_loss(
                    diffusion.loss(model, target_x, target_y, reduction="none"),
                    int(target_x.shape[0]),
                )
                target_loss = target_losses.mean()
                if auxiliary_iterator is not None:
                    auxiliary_x, auxiliary_labels = next(auxiliary_iterator)
                    auxiliary_x = auxiliary_x.to(device, non_blocking=True)
                    auxiliary_labels = auxiliary_labels.to(device, non_blocking=True)
                    auxiliary_losses = _per_example_loss(
                        diffusion.loss(model, auxiliary_x, auxiliary_labels, reduction="none"),
                        int(auxiliary_x.shape[0]),
                    )
                    auxiliary_loss = auxiliary_losses.mean()
                    loss = target_loss + auxiliary_loss_weight * auxiliary_loss
                    pooled_loss = torch.cat((target_losses, auxiliary_losses)).mean()
                else:
                    auxiliary_labels = None
                    auxiliary_losses = target_losses.new_empty((0,))
                    auxiliary_loss = None
                    loss = target_loss
                    pooled_loss = target_loss
            counters["target_examples_seen"] += int(target_x.shape[0])
            if auxiliary_labels is not None:
                counters["auxiliary_examples_seen"] += int(auxiliary_labels.shape[0])
                unique, counts = auxiliary_labels.detach().cpu().unique(return_counts=True)
                for class_id, count in zip(unique.tolist(), counts.tolist()):
                    key = str(int(class_id))
                    counters["auxiliary_examples_seen_by_class"][key] = counters["auxiliary_examples_seen_by_class"].get(key, 0) + int(count)
            rolling_losses["pooled"].add(
                float(target_losses.detach().sum().cpu()) + float(auxiliary_losses.detach().sum().cpu()),
                int(target_losses.numel() + auxiliary_losses.numel()),
            )
            rolling_losses["target"].add(float(target_losses.detach().sum().cpu()), int(target_losses.numel()))
            rolling_losses["auxiliary"].add(
                float(auxiliary_losses.detach().sum().cpu()),
                int(auxiliary_losses.numel()),
            )
            final_pooled_batch_size = int(target_losses.numel() + auxiliary_losses.numel())
            final_target_batch_size = int(target_losses.numel())
            final_auxiliary_batch_size = int(auxiliary_losses.numel())

        scaler.scale(loss).backward()
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            final_grad_norm = float(grad_norm_tensor.detach().cpu())
            if math.isfinite(final_grad_norm) and final_grad_norm > float(max_grad_norm):
                gradient_clipping_count += 1
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        cumulative_optimizer_seconds += time.perf_counter() - optimizer_step_start
        final_loss = float(loss.detach().cpu().item())
        final_pooled_loss = float(pooled_loss.detach().cpu().item())
        final_target_loss = float(target_loss.detach().cpu().item()) if target_loss is not None else float("nan")
        final_auxiliary_loss = float(auxiliary_loss.detach().cpu().item()) if auxiliary_loss is not None else float("nan")
        current_step = step + 1
        counters["optimizer_steps"] = current_step

        should_validate = current_step % max(validation_interval, 1) == 0 or current_step == steps
        if should_validate:
            denoise = _evaluate_validation(
                ema.shadow,
                diffusion,
                val_dataset,
                device=device,
                conditional=conditional,
                batch_size=batch_size,
                num_workers=num_workers,
                validation_evaluator=validation_evaluator,
                validation_corruption_bank=validation_corruption_bank,
            )
            last_validation_step = current_step
            validation_metric = denoise["all"]
            if train_log is not None:
                _write_log_row(
                    train_log,
                    {"step": current_step, "train_loss": final_loss, "validation_epsilon_mse": validation_metric},
                )
            # A segment may later be resumed with a larger total-step target.
            # Only scheduled validation points participate in best-checkpoint
            # selection, so an incidental segment endpoint cannot change the
            # best model relative to uninterrupted training.
            eligible_for_best = current_step % max(validation_interval, 1) == 0
            if eligible_for_best and math.isfinite(validation_metric) and validation_metric < current_best:
                current_best = validation_metric
                save_training_state(best_path, current_step)
        if last_path is not None and (
            current_step == steps or current_step % resolved_checkpoint_interval == 0
        ):
            save_training_state(last_path, current_step)
            if legacy_alias is not None:
                save_training_state(legacy_alias, current_step)

    if last_validation_step != steps:
        denoise = _evaluate_validation(
            ema.shadow,
            diffusion,
            val_dataset,
            device=device,
            conditional=conditional,
            batch_size=batch_size,
            num_workers=num_workers,
            validation_evaluator=validation_evaluator,
            validation_corruption_bank=validation_corruption_bank,
        )
        validation_metric = denoise["all"]
        eligible_for_best = steps % max(validation_interval, 1) == 0
        if eligible_for_best and math.isfinite(validation_metric) and validation_metric < current_best:
            current_best = validation_metric
            save_training_state(best_path, steps)

    # A terminal run must always expose both canonical checkpoint names. Until
    # the first scheduled validation there is no eligible historical best, so
    # refresh this provisional artifact at every segment endpoint. Keeping
    # ``current_best`` infinite preserves equivalence with uninterrupted runs.
    if best_path is not None and math.isinf(current_best) and math.isfinite(denoise["all"]):
        save_training_state(best_path, steps, best_metric_override=float(denoise["all"]))

    if last_path is not None:
        save_training_state(last_path, steps)
        if legacy_alias is not None:
            save_training_state(legacy_alias, steps)

    metadata = protocol_metadata(steps)
    train_seconds = float(metadata["wallclock_train_seconds"])
    reported_best = (
        current_best
        if not math.isinf(current_best)
        else (float(denoise["all"]) if best_path is not None and best_path.exists() and math.isfinite(denoise["all"]) else None)
    )
    return ema.shadow, diffusion, {
        "final_objective_train_loss": final_loss,
        "final_pooled_train_loss": final_pooled_loss,
        "final_target_batch_train_loss": final_target_loss,
        "final_auxiliary_batch_train_loss": final_auxiliary_loss,
        "rolling_pooled_train_loss": rolling_losses["pooled"].mean(),
        "rolling_target_train_loss": rolling_losses["target"].mean(),
        "rolling_auxiliary_train_loss": rolling_losses["auxiliary"].mean(),
        # Compatibility aliases for result readers created before loss
        # accounting distinguished the pooled and target-only components.
        "final_train_loss": final_loss,
        "final_target_train_loss": final_target_loss,
        "final_auxiliary_train_loss": final_auxiliary_loss,
        "final_gradient_norm": final_grad_norm,
        "gradient_clipping_count": int(gradient_clipping_count),
        "max_grad_norm": max_grad_norm,
        "optimizer_name": "adamw",
        "optimizer_betas": list(optimizer_betas),
        "optimizer_eps": float(optimizer_eps),
        "optimizer_weight_decay": float(weight_decay),
        "ema_decay": float(ema_decay),
        "wallclock_train_seconds": train_seconds,
        "validation_epsilon_mse_target": denoise["all"],
        "validation_epsilon_mse_low_noise": denoise["low"],
        "validation_epsilon_mse_mid_noise": denoise["mid"],
        "validation_epsilon_mse_high_noise": denoise["high"],
        "validation_epsilon_mse_standard_error": denoise["standard_error"],
        "num_validation_images": denoise["num_validation_images"],
        "num_corruptions": denoise["num_corruptions"],
        "training_protocol": training_protocol,
        **metadata,
        "auxiliary_examples_seen_by_class_json": json.dumps(
            metadata["auxiliary_examples_seen_by_class"], sort_keys=True
        ),
        "best_validation_epsilon_mse_target": reported_best,
        "last_checkpoint_path": str(last_path) if last_path is not None else "",
        "best_checkpoint_path": str(best_path) if best_path is not None else "",
        **{key: value for key, value in architecture_metadata.items() if key != "resolved_model_config"},
    }
