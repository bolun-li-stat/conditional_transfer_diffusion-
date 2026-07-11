from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed all process-level RNGs used by the image experiments.

    Sampling uses its own ``torch.Generator`` elsewhere; this function is for
    model initialization/training and intentionally does not create or retain a
    hidden generator.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def capture_rng_state() -> dict[str, Any]:
    """Return every RNG state required for an exact training resume."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore a state produced by :func:`capture_rng_state`.

    CUDA state is kept in CPU checkpoints but is skipped when resuming on a
    machine without CUDA.  This makes checkpoint inspection and CPU evaluation
    possible without weakening exact same-device resumes.
    """

    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda")
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


@contextlib.contextmanager
def preserve_rng_state() -> Iterator[None]:
    """Run code without allowing it to advance the training RNG streams."""

    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


@contextlib.contextmanager
def isolated_seed(seed: int, *, deterministic: bool = False) -> Iterator[None]:
    """Temporarily seed process RNGs, restoring their previous state on exit."""

    state = capture_rng_state()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    try:
        set_seed(seed, deterministic=deterministic)
        yield
    finally:
        restore_rng_state(state)
        torch.use_deterministic_algorithms(deterministic_before)


def make_torch_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """Create an explicit generator without changing global RNG state."""

    resolved = torch.device(device)
    generator_device = resolved if resolved.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(int(seed))
