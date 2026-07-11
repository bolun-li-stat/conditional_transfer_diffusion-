from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from image_transfer.training.ema import EMA
from image_transfer.utils.seed import capture_rng_state, restore_rng_state

CHECKPOINT_SCHEMA_VERSION = 3


def _model_state(value: nn.Module | EMA | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, EMA):
        return value.shadow.state_dict()
    if isinstance(value, nn.Module):
        return value.state_dict()
    return dict(value)


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # PyTorch 2.6 defaults to weights_only=True, which cannot deserialize NumPy
    # RNG state.  These checkpoints are local artifacts created by this code.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        return torch.load(path, map_location=map_location)


def _atomic_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temp_name = handle.name
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
        temp_name = None
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:  # pragma: no cover - not supported by every filesystem
            pass
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
    return destination


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
    *,
    ema_model: nn.Module | EMA | Mapping[str, Any] | None = None,
    scaler: Any | None = None,
    best_validation_metric: float | None = None,
    rng_state: dict[str, Any] | None = None,
    config_hash: str | None = None,
    manifest_hash: str | None = None,
    git_sha: str | None = None,
    training_protocol: str | None = None,
    protocol_metadata: dict[str, Any] | None = None,
    data_state: dict[str, Any] | None = None,
    model_metadata: dict[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a rigorous training checkpoint.

    ``model``, ``optimizer``, ``step`` and ``extra`` retain the original public
    calling convention.  The legacy ``model`` key points to EMA weights when
    available so old sampling scripts continue to evaluate EMA, while resume
    code always reads ``raw_model_state`` explicitly.
    """

    raw_state = _model_state(model)
    ema_state = _model_state(ema_model) or raw_state
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    scaler_state = scaler.state_dict() if scaler is not None else None
    payload: dict[str, Any] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "raw_model_state": raw_state,
        "ema_model_state": ema_state,
        "optimizer_state": optimizer_state,
        "grad_scaler_state": scaler_state,
        "step": int(step),
        "best_validation_metric": best_validation_metric,
        "rng_states": rng_state if rng_state is not None else capture_rng_state(),
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "git_sha": git_sha,
        "training_protocol": training_protocol,
        "training_protocol_metadata": dict(protocol_metadata or {}),
        "data_state": dict(data_state or {}),
        "model_metadata": dict(model_metadata or {}),
        "provenance": dict(provenance or {}),
        "extra": dict(extra or {}),
        # Backward-compatible aliases.  Resume must not use these aliases.
        "model": ema_state,
        "optimizer": optimizer_state,
    }
    return _atomic_torch_save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    map_location: str | torch.device = "cpu",
    *,
    use_ema: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """Load weights for inference, preferring EMA for backward compatibility."""

    checkpoint = _torch_load(path, map_location=map_location)
    metadata = checkpoint.get("model_metadata") or {}
    expected_architecture = getattr(model, "resolved_model_config", {}).get("architecture")
    expected_model_hash = getattr(model, "model_config_hash", None)
    if metadata:
        if expected_architecture and metadata.get("architecture") != expected_architecture:
            raise ValueError(
                f"Checkpoint architecture {metadata.get('architecture')!r} does not match model {expected_architecture!r}"
            )
        if expected_model_hash and metadata.get("model_config_hash") != expected_model_hash:
            raise ValueError("Checkpoint model config hash does not match the constructed model")
    if use_ema:
        state = checkpoint.get("ema_model_state") or checkpoint.get("model")
    else:
        state = checkpoint.get("raw_model_state") or checkpoint.get("model")
    if state is None:
        raise KeyError(f"Checkpoint {path} contains no model state")
    model.load_state_dict(state, strict=strict)
    return checkpoint


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None = None,
    *,
    map_location: str | torch.device = "cpu",
    expected_config_hash: str | None = None,
    expected_manifest_hash: str | None = None,
    expected_model_config_hash: str | None = None,
    expected_architecture: str | None = None,
    restore_rng: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """Restore raw/EMA/optimizer/scaler and RNG state for an exact resume."""

    checkpoint = _torch_load(path, map_location=map_location)
    if expected_config_hash and checkpoint.get("config_hash") not in {None, expected_config_hash}:
        raise ValueError("Checkpoint config hash does not match this run")
    if expected_manifest_hash and checkpoint.get("manifest_hash") not in {None, expected_manifest_hash}:
        raise ValueError("Checkpoint manifest hash does not match this run")
    model_metadata = checkpoint.get("model_metadata") or {}
    if expected_architecture and model_metadata.get("architecture") not in {None, expected_architecture}:
        raise ValueError("Checkpoint architecture does not match this run")
    if expected_model_config_hash and model_metadata.get("model_config_hash") not in {None, expected_model_config_hash}:
        raise ValueError("Checkpoint model config hash does not match this run")

    raw_state = checkpoint.get("raw_model_state")
    if raw_state is None:
        # Schema-v1 checkpoints saved EMA in ``model``.  They can be resumed,
        # but exact raw/optimizer consistency cannot be guaranteed.
        raw_state = checkpoint.get("model")
    if raw_state is None:
        raise KeyError(f"Checkpoint {path} contains no raw model state")
    model.load_state_dict(raw_state, strict=strict)

    ema_state = checkpoint.get("ema_model_state") or checkpoint.get("model") or raw_state
    ema.load_state_dict({"decay": ema.decay, "model": ema_state}, strict=strict)
    optimizer_state = checkpoint.get("optimizer_state", checkpoint.get("optimizer"))
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    scaler_state = checkpoint.get("grad_scaler_state")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    if restore_rng:
        restore_rng_state(checkpoint.get("rng_states"))
    return checkpoint


def checkpoint_paths(
    checkpoint_path: str | Path | None,
    *,
    last_checkpoint_path: str | Path | None = None,
    best_checkpoint_path: str | Path | None = None,
) -> tuple[Path | None, Path | None, Path | None]:
    """Resolve canonical ``*_last.pt``/``*_best.pt`` and a legacy alias."""

    legacy = Path(checkpoint_path) if checkpoint_path is not None else None
    if last_checkpoint_path is not None:
        last = Path(last_checkpoint_path)
    elif legacy is None:
        last = None
    elif legacy.stem.endswith("_last"):
        last = legacy
    else:
        last = legacy.with_name(f"{legacy.stem}_last{legacy.suffix or '.pt'}")
    if best_checkpoint_path is not None:
        best = Path(best_checkpoint_path)
    elif last is None:
        best = None
    else:
        base_stem = last.stem[:-5] if last.stem.endswith("_last") else last.stem
        best = last.with_name(f"{base_stem}_best{last.suffix or '.pt'}")
    alias = legacy if legacy is not None and legacy != last else None
    return last, best, alias
