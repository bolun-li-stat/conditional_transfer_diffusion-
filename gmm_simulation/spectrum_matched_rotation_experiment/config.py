"""Typed configuration for the spectrum-matched rotation experiment."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SpectrumConfig:
    lambda_high: float = 1.8
    lambda_low: float = 0.2
    rotation_degrees: tuple[int, ...] = (0, 45, 75)


@dataclass(frozen=True)
class CapacityConfig:
    time_embedding_dim: int
    class_embedding_dim: int
    hidden_width: int
    hidden_layers: int


CAPACITIES = {
    "standard": CapacityConfig(64, 32, 256, 4),
    "limited": CapacityConfig(64, 8, 128, 2),
}


@dataclass
class ExperimentConfig:
    seed: int = 0
    capacity: str = "standard"
    rotation_deg: int = 0
    K: int = 3
    d: int = 100
    n_target_train: int = 200
    n_aux_train: int = 200
    n_validation: int = 1000
    n_test: int = 2000
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    batch_size: int = 512
    learning_rate: float = 2e-4
    training_steps: int = 20_000
    score_risk_mc_samples: int = 5_000
    n_generated: int = 2_000
    device: str = "auto"
    results_dir: Path = Path("results")
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)

    def validate(self) -> None:
        if self.K != 3:
            raise ValueError("This experiment is intentionally fixed at K=3.")
        if self.d <= 0 or self.d % 2:
            raise ValueError("d must be a positive even integer.")
        if self.capacity not in CAPACITIES:
            raise ValueError(f"Unknown capacity: {self.capacity}")
        if self.rotation_deg not in self.spectrum.rotation_degrees:
            raise ValueError(f"Unsupported formal rotation: {self.rotation_deg}")


def smoke_config(results_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(d=8, n_target_train=24, n_aux_train=24, n_validation=32,
                            n_test=32, T=12, batch_size=16, training_steps=2,
                            score_risk_mc_samples=24, n_generated=16,
                            capacity="limited", results_dir=results_dir)
