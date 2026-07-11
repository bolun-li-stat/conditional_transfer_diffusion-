"""Runtime probes that exercise the configured model and optimizer path."""

from __future__ import annotations

import gc
import copy
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from image_transfer.diffusion.ddpm import ImageDDPM
from image_transfer.models import build_image_model, model_parameter_metadata
from image_transfer.training.checkpointing import load_training_checkpoint, save_checkpoint
from image_transfer.training.ema import EMA
from image_transfer.utils.seed import set_seed


RUNTIME_PROBE_SCHEMA_VERSION = "3.0"
PROTOCOLS = ("natural_compute_matched", "target_exposure_matched")


def _new_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def maximum_configured_num_classes(config: dict[str, Any]) -> int:
    configured_counts: list[int] = []
    configured_counts.extend(1 + int(value) for value in config.get("K_aux_values", []) or [])
    for experiment in (config.get("experiments") or {}).values():
        configured_counts.extend(
            1 + int(value) for value in experiment.get("K_aux_values", []) or []
        )
        for composition in experiment.get("compositions", []) or []:
            if isinstance(composition, dict):
                configured_counts.append(1 + sum(int(value) for value in composition.values()))
    for setting in config.get("auxiliary_size_settings", []) or []:
        if isinstance(setting, dict):
            configured_counts.append(1 + int(setting.get("K_aux", 0)))

    # A target's auxiliary candidate pool can be larger than the configured K.
    # Probe the largest class count that a job can actually instantiate, using
    # candidate-pool size only as a fallback for legacy configs without K.
    if configured_counts:
        return max(1, *configured_counts)

    candidate_counts = [1]
    targets = config.get("targets") or []
    for target in targets:
        for values in (target.get("auxiliary_sets") or {}).values():
            candidate_counts.append(1 + len(values))
    return max(candidate_counts)


def _optimizer(config: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    cfg = config.get("optimizer") or {}
    if str(cfg.get("name", "adamw")).lower() != "adamw":
        raise ValueError("runtime probe supports the configured AdamW optimizer only")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 2e-4)),
        betas=tuple(float(value) for value in cfg.get("betas", (0.9, 0.999))),
        eps=float(cfg.get("eps", 1e-8)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )


def _batch_sizes(config: dict[str, Any], protocol: str) -> tuple[int, int]:
    training = config.get("training") or {}
    if protocol == "natural_compute_matched":
        return int(training.get("batch_size", 1)), 0
    return (
        int(training.get("target_batch_size", training.get("batch_size", 1))),
        int(training.get("auxiliary_batch_size", training.get("batch_size", 1))),
    )


def _step(
    *,
    model: torch.nn.Module,
    diffusion: ImageDDPM,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    scaler: Any,
    config: dict[str, Any],
    protocol: str,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    training = config.get("training") or {}
    precision = str(training.get("precision", "fp32"))
    amp_enabled = precision == "amp" and device.type == "cuda"
    target_size, auxiliary_size = _batch_sizes(config, protocol)
    size = int(config["image_size"])
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
        if protocol == "natural_compute_matched":
            images = torch.randn(target_size, 3, size, size, device=device)
            labels = torch.arange(target_size, device=device) % max(num_classes, 1)
            loss = diffusion.loss(model, images, labels)
            total_size = target_size
        else:
            target = torch.randn(target_size, 3, size, size, device=device)
            target_labels = torch.zeros(target_size, dtype=torch.long, device=device)
            target_loss = diffusion.loss(model, target, target_labels)
            total_size = target_size + auxiliary_size
            if auxiliary_size:
                auxiliary = torch.randn(auxiliary_size, 3, size, size, device=device)
                auxiliary_labels = 1 + torch.arange(auxiliary_size, device=device) % max(num_classes - 1, 1)
                auxiliary_loss = diffusion.loss(model, auxiliary, auxiliary_labels)
                loss = target_loss + float(training.get("auxiliary_loss_weight", 1.0)) * auxiliary_loss
            else:
                loss = target_loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    max_grad_norm = training.get("max_grad_norm")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(max_grad_norm) if max_grad_norm is not None else math.inf,
    )
    scaler.step(optimizer)
    scaler.update()
    ema.update(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    loss_value = float(loss.detach().cpu())
    gradient_value = float(grad_norm.detach().cpu())
    return {
        "loss": loss_value,
        "gradient_norm": gradient_value,
        "loss_finite": math.isfinite(loss_value),
        "gradient_finite": math.isfinite(gradient_value),
        "step_finite": math.isfinite(loss_value) and math.isfinite(gradient_value),
        "target_batch_size": target_size,
        "auxiliary_batch_size": auxiliary_size,
        "total_batch_size": total_size,
        "elapsed_seconds": time.perf_counter() - started,
        "amp_enabled": amp_enabled,
        "grad_scaler_enabled": bool(scaler.is_enabled()),
        "optimizer": "adamw",
        "gradient_clipping_configured": max_grad_norm is not None,
        "ema_updated": True,
    }


def run_load_probe(
    config: dict[str, Any],
    *,
    device: str | torch.device,
    protocol: str,
) -> dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown runtime-probe protocol {protocol!r}")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(resolved_device)
    num_classes = maximum_configured_num_classes(config)
    model = build_image_model(
        config.get("model"),
        image_size=int(config["image_size"]),
        conditional=True,
        num_classes=num_classes,
        model_seed=0,
    ).to(resolved_device)
    optimizer = _optimizer(config, model)
    ema = EMA(model, float((config.get("training") or {}).get("ema_decay", 0.999)))
    scaler = _new_scaler(
        str((config.get("training") or {}).get("precision", "fp32")) == "amp"
        and resolved_device.type == "cuda"
    )
    diffusion_cfg = config.get("diffusion") or {}
    diffusion = ImageDDPM(
        timesteps=int(diffusion_cfg.get("timesteps", 1000)),
        schedule=str(diffusion_cfg.get("schedule", "linear")),
        device=resolved_device,
    )
    details = _step(
        model=model,
        diffusion=diffusion,
        optimizer=optimizer,
        ema=ema,
        scaler=scaler,
        config=config,
        protocol=protocol,
        device=resolved_device,
        num_classes=num_classes,
    )
    details.update(model_parameter_metadata(model))
    details["protocol"] = protocol
    details["device"] = str(resolved_device)
    details["num_classes"] = num_classes
    if resolved_device.type == "cuda":
        properties = torch.cuda.get_device_properties(resolved_device)
        peak_allocated = int(torch.cuda.max_memory_allocated(resolved_device))
        peak_reserved = int(torch.cuda.max_memory_reserved(resolved_device))
        total = int(properties.total_memory)
        details.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": total,
                "peak_gpu_memory_allocated_bytes": peak_allocated,
                "peak_gpu_memory_reserved_bytes": peak_reserved,
                "gpu_headroom_bytes": max(0, total - peak_reserved),
            }
        )
    else:
        details.update(
            {
                "gpu_name": "not_applicable",
                "gpu_total_memory_bytes": 0,
                "peak_gpu_memory_allocated_bytes": 0,
                "peak_gpu_memory_reserved_bytes": 0,
                "gpu_headroom_bytes": 0,
            }
        )
    return details


def _states_close(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], *, exact: bool) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        if exact:
            if not torch.equal(left[key].detach().cpu(), right[key].detach().cpu()):
                return False
        elif not torch.allclose(
            left[key].detach().cpu(), right[key].detach().cpu(), rtol=1e-5, atol=1e-6
        ):
            return False
    return True


def _clone_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return copy.deepcopy(value)


def _nested_state_matches(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_state_matches(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_state_matches(a, b) for a, b in zip(left, right)
        )
    return left == right


def run_resume_roundtrip(
    config: dict[str, Any],
    *,
    device: str | torch.device,
    protocol: str,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compare continuous training with save/destroy/rebuild/load/continue."""

    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown runtime-probe protocol {protocol!r}")
    resolved_device = torch.device(device)
    exact = resolved_device.type == "cpu"
    num_classes = maximum_configured_num_classes(config)
    root_context = tempfile.TemporaryDirectory() if work_dir is None else None
    root = Path(root_context.name if root_context is not None else work_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / f"resume-probe-{protocol}.pt"

    def build_state():
        model = build_image_model(
            config.get("model"), image_size=int(config["image_size"]), conditional=True,
            num_classes=num_classes, model_seed=123,
        ).to(resolved_device)
        optimizer = _optimizer(config, model)
        ema = EMA(model, float((config.get("training") or {}).get("ema_decay", 0.999)))
        scaler = _new_scaler(
            str((config.get("training") or {}).get("precision", "fp32")) == "amp"
            and resolved_device.type == "cuda"
        )
        diffusion_cfg = config.get("diffusion") or {}
        diffusion = ImageDDPM(
            timesteps=int(diffusion_cfg.get("timesteps", 1000)),
            schedule=str(diffusion_cfg.get("schedule", "linear")), device=resolved_device,
        )
        return model, optimizer, ema, scaler, diffusion

    def execute_steps(state, count: int):
        model, optimizer, ema, scaler, diffusion = state
        records = []
        for _ in range(count):
            records.append(_step(
                model=model, diffusion=diffusion, optimizer=optimizer, ema=ema, scaler=scaler,
                config=config, protocol=protocol, device=resolved_device, num_classes=num_classes,
            ))
        return records

    set_seed(987, deterministic=exact)
    continuous = build_state()
    continuous_records = execute_steps(continuous, 2)
    continuous_raw = {key: value.detach().cpu().clone() for key, value in continuous[0].state_dict().items()}
    continuous_ema = {key: value.detach().cpu().clone() for key, value in continuous[2].shadow.state_dict().items()}

    set_seed(987, deterministic=exact)
    interrupted = build_state()
    first_records = execute_steps(interrupted, 1)
    target_batch_size, auxiliary_batch_size = _batch_sizes(config, protocol)
    first_counters = {
        "optimizer_steps": 1,
        "target_examples_seen": target_batch_size,
        "auxiliary_examples_seen": auxiliary_batch_size,
        "total_examples_seen": target_batch_size + auxiliary_batch_size,
    }
    saved_raw = _clone_state(interrupted[0].state_dict())
    saved_ema = _clone_state(interrupted[2].shadow.state_dict())
    saved_optimizer = _clone_state(interrupted[1].state_dict())
    saved_scaler = _clone_state(interrupted[3].state_dict())
    save_checkpoint(
        checkpoint_path, interrupted[0], interrupted[1], 1,
        ema_model=interrupted[2], scaler=interrupted[3],
        config_hash="runtime-probe", manifest_hash="synthetic-runtime-probe",
        training_protocol=protocol,
        protocol_metadata=first_counters,
        data_state={
            "global_step": 1,
            "sampler_seed": 987,
            "sampler_position": 1,
            "num_workers": 0,
        },
        model_metadata=model_parameter_metadata(interrupted[0]),
        provenance={"runtime_probe": True},
    )
    del interrupted
    gc.collect()
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    resumed = build_state()
    restored_checkpoint = load_training_checkpoint(
        checkpoint_path, resumed[0], resumed[2], resumed[1], resumed[3],
        map_location=resolved_device,
        expected_config_hash="runtime-probe",
        expected_manifest_hash="synthetic-runtime-probe",
        expected_model_config_hash=model_parameter_metadata(resumed[0])["model_config_hash"],
        expected_architecture=model_parameter_metadata(resumed[0])["architecture"],
        restore_rng=True,
    )
    restored_raw = _clone_state(resumed[0].state_dict())
    restored_ema = _clone_state(resumed[2].shadow.state_dict())
    raw_state_restored = _nested_state_matches(saved_raw, restored_raw)
    ema_state_restored = _nested_state_matches(saved_ema, restored_ema)
    optimizer_state_restored = _nested_state_matches(saved_optimizer, resumed[1].state_dict())
    grad_scaler_state_restored = _nested_state_matches(saved_scaler, resumed[3].state_dict())
    raw_and_ema_distinct = not _nested_state_matches(saved_raw, saved_ema)
    raw_not_loaded_from_ema = raw_state_restored and (
        not raw_and_ema_distinct or not _nested_state_matches(restored_raw, saved_ema)
    )
    restored_step = int(restored_checkpoint.get("step", -1))
    global_step_continuous = (
        restored_step == 1
        and int((restored_checkpoint.get("data_state") or {}).get("global_step", -1)) == 1
    )
    sampler_position_continuous = (
        int((restored_checkpoint.get("data_state") or {}).get("sampler_seed", -1)) == 987
        and int((restored_checkpoint.get("data_state") or {}).get("sampler_position", -1)) == 1
    )
    exposure_counters_restored = all(
        int((restored_checkpoint.get("training_protocol_metadata") or {}).get(key, -1)) == value
        for key, value in first_counters.items()
    )
    resumed_records = execute_steps(resumed, 1)
    resumed_raw = {key: value.detach().cpu() for key, value in resumed[0].state_dict().items()}
    resumed_ema = {key: value.detach().cpu() for key, value in resumed[2].shadow.state_dict().items()}
    raw_matches = _states_close(continuous_raw, resumed_raw, exact=exact)
    ema_matches = _states_close(continuous_ema, resumed_ema, exact=exact)
    if exact:
        loss_sequence_matches = [item["loss"] for item in continuous_records] == [
            first_records[0]["loss"], resumed_records[0]["loss"]
        ]
    else:
        loss_sequence_matches = all(
            math.isclose(left["loss"], right["loss"], rel_tol=1e-5, abs_tol=1e-6)
            for left, right in zip(continuous_records, [first_records[0], resumed_records[0]])
        )
    final_counters = {
        "optimizer_steps": restored_step + 1,
        "target_examples_seen": first_counters["target_examples_seen"] + target_batch_size,
        "auxiliary_examples_seen": first_counters["auxiliary_examples_seen"] + auxiliary_batch_size,
        "total_examples_seen": first_counters["total_examples_seen"] + target_batch_size + auxiliary_batch_size,
    }
    exposure_counters_continuous = exposure_counters_restored and final_counters == {
        "optimizer_steps": 2,
        "target_examples_seen": 2 * target_batch_size,
        "auxiliary_examples_seen": 2 * auxiliary_batch_size,
        "total_examples_seen": 2 * (target_batch_size + auxiliary_batch_size),
    }
    passed = all(
        (
            raw_matches,
            ema_matches,
            raw_state_restored,
            ema_state_restored,
            optimizer_state_restored,
            grad_scaler_state_restored,
            raw_not_loaded_from_ema,
            global_step_continuous,
            sampler_position_continuous,
            exposure_counters_continuous,
            loss_sequence_matches,
        )
    )
    if root_context is not None:
        root_context.cleanup()
    return {
        "protocol": protocol,
        "device": str(resolved_device),
        "comparison_mode": "bitwise" if exact else "finite_tolerance",
        "continuous_steps": 2,
        "interrupted_steps": 1,
        "resumed_steps": 1,
        "checkpoint_saved": True,
        "objects_destroyed_before_restore": True,
        "raw_model_matches": raw_matches,
        "ema_model_matches": ema_matches,
        "raw_model_state_restored": raw_state_restored,
        "ema_model_state_restored": ema_state_restored,
        "optimizer_state_restored": optimizer_state_restored,
        "grad_scaler_state_restored": grad_scaler_state_restored,
        "raw_not_loaded_from_ema": raw_not_loaded_from_ema,
        "global_step_restored": restored_step,
        "global_step_after_continue": restored_step + 1,
        "global_step_continuous": global_step_continuous,
        "sampler_position_restored": 1 if sampler_position_continuous else -1,
        "sampler_position_after_continue": 2 if sampler_position_continuous else -1,
        "sampler_position_continuous": sampler_position_continuous,
        "exposure_counters_restored": exposure_counters_restored,
        "exposure_counters_after_continue": final_counters,
        "exposure_counters_continuous": exposure_counters_continuous,
        "loss_sequence_matches": loss_sequence_matches,
        "continuous_losses": [item["loss"] for item in continuous_records],
        "resumed_losses": [first_records[0]["loss"], resumed_records[0]["loss"]],
        "finite": all(math.isfinite(item["loss"]) for item in continuous_records + first_records + resumed_records),
        "passed": bool(passed),
    }


__all__ = [
    "PROTOCOLS",
    "RUNTIME_PROBE_SCHEMA_VERSION",
    "maximum_configured_num_classes",
    "run_load_probe",
    "run_resume_roundtrip",
]
