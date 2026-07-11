"""Configuration objects for the DDPM Gaussian-mixture transfer experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


VarianceType = Literal["posterior", "beta"]
SamplingMode = Literal["balanced", "natural"]
ExperimentType = Literal["smoke", "low_target_data", "same_total_budget"]
CovarianceScenario = Literal["shared", "mismatch"]


@dataclass
class DiffusionConfig:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    variance_type: VarianceType = "posterior"


@dataclass
class DataConfig:
    d: int = 100
    K: int = 3
    Delta: float = 20.0
    target_class: int = 1
    min_pairwise_mean_distance: float = 15.0
    jitter: float = 0.0
    covariance_scenario: CovarianceScenario = "shared"
    rho: float | None = 0.2
    mismatch_level: str | None = None
    class_varying_target_rho: float = 0.1


@dataclass
class ModelConfig:
    time_embedding_dim: int = 64
    class_embedding_dim: int = 32
    hidden_width: int = 256
    hidden_layers: int = 4


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-4
    batch_size: int = 512
    training_steps: int = 20_000
    validation_interval: int = 500
    checkpoint_interval: int = 2_000
    save_checkpoints: bool = True
    resume_from_checkpoint: bool = False
    sampling_mode: SamplingMode = "balanced"
    device: str = "auto"
    num_workers: int = 0


@dataclass
class EvaluationConfig:
    n_test_target: int = 10_000
    n_generated: int = 5_000
    score_risk_mc_samples: int = 10_000
    mmd_max_samples: int = 2_000
    save_samples: bool = False


@dataclass
class ExperimentConfig:
    experiment_type: ExperimentType = "smoke"
    seed: int = 0
    n: int | None = None
    n_target_train: int | None = 100
    n_aux_train: int | None = 1_000
    results_dir: Path = Path("results_T1000_K3")
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def figure_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def checkpoint_dir(self) -> Path:
        return self.results_dir / "checkpoints"

    @property
    def config_dir(self) -> Path:
        return self.results_dir / "configs"

    @property
    def log_dir(self) -> Path:
        return self.results_dir / "logs"

    @property
    def sample_dir(self) -> Path:
        return self.results_dir / "samples"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def sqrt_alpha_bar_T(diffusion: DiffusionConfig) -> float:
    import math

    if diffusion.T <= 0:
        raise ValueError("Diffusion horizon T must be positive.")
    if diffusion.T == 1:
        betas = [diffusion.beta_start]
    else:
        step = (diffusion.beta_end - diffusion.beta_start) / (diffusion.T - 1)
        betas = [diffusion.beta_start + i * step for i in range(diffusion.T)]
    return float(math.sqrt(math.prod(1.0 - beta for beta in betas)))


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    out = _to_jsonable(asdict(config))
    out["sqrt_alpha_bar_T"] = sqrt_alpha_bar_T(config.diffusion)
    return out


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    cfg = ExperimentConfig()
    for key, value in data.items():
        if key == "diffusion":
            cfg.diffusion = DiffusionConfig(**value)
        elif key == "data":
            cfg.data = DataConfig(**value)
        elif key == "model":
            cfg.model = ModelConfig(**value)
        elif key == "training":
            cfg.training = TrainingConfig(**value)
        elif key == "evaluation":
            cfg.evaluation = EvaluationConfig(**value)
        elif key == "results_dir":
            cfg.results_dir = Path(value)
        else:
            setattr(cfg, key, value)
    return cfg


def default_smoke_config() -> ExperimentConfig:
    cfg = ExperimentConfig(experiment_type="smoke", seed=0, n_target_train=64, n_aux_train=64)
    cfg.training.training_steps = 200
    cfg.training.validation_interval = 50
    cfg.training.checkpoint_interval = 100
    cfg.training.batch_size = 128
    cfg.evaluation.n_test_target = 512
    cfg.evaluation.n_generated = 256
    cfg.evaluation.score_risk_mc_samples = 512
    cfg.evaluation.mmd_max_samples = 512
    cfg.evaluation.save_samples = True
    return cfg


def default_experiment1_configs(seeds: list[int] | None = None) -> list[ExperimentConfig]:
    seeds = list(range(20)) if seeds is None else seeds
    configs: list[ExperimentConfig] = []
    for seed in seeds:
        for rho in [0.2, 0.0, -0.2]:
            for n_target in [200, 500, 800]:
                cfg = ExperimentConfig(experiment_type="low_target_data", seed=seed, n_target_train=n_target, n_aux_train=n_target)
                cfg.data.covariance_scenario = "shared"
                cfg.data.rho = rho
                cfg.data.mismatch_level = None
                configs.append(cfg)
        for level in ["0", "mild", "medium", "strong"]:
            for n_target in [200, 500, 800]:
                cfg = ExperimentConfig(experiment_type="low_target_data", seed=seed, n_target_train=n_target, n_aux_train=n_target)
                cfg.data.covariance_scenario = "mismatch"
                cfg.data.rho = None
                cfg.data.mismatch_level = level
                configs.append(cfg)
    return configs


def default_experiment2_configs(seeds: list[int] | None = None) -> list[ExperimentConfig]:
    seeds = list(range(20)) if seeds is None else seeds
    configs: list[ExperimentConfig] = []
    for seed in seeds:
        for rho in [0.2, 0.0, -0.2]:
            for n in [200, 500, 800]:
                cfg = ExperimentConfig(experiment_type="same_total_budget", seed=seed, n=n, n_target_train=n, n_aux_train=n)
                cfg.data.covariance_scenario = "shared"
                cfg.data.rho = rho
                cfg.data.mismatch_level = None
                configs.append(cfg)
        for level in ["0", "mild", "medium", "strong"]:
            for n in [200, 500, 800]:
                cfg = ExperimentConfig(experiment_type="same_total_budget", seed=seed, n=n, n_target_train=n, n_aux_train=n)
                cfg.data.covariance_scenario = "mismatch"
                cfg.data.rho = None
                cfg.data.mismatch_level = level
                configs.append(cfg)
    return configs
