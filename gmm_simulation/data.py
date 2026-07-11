"""Synthetic Gaussian-mixture data generation for transfer diffusion experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianMixtureSpec:
    means: np.ndarray
    covariances: np.ndarray
    rhos: list[float]
    target_class: int = 1

    @property
    def d(self) -> int:
        return int(self.means.shape[1])

    @property
    def K(self) -> int:
        return int(self.means.shape[0])

    @property
    def target_index(self) -> int:
        return self.target_class - 1


def make_ar1_covariance(d: int, rho: float, jitter: float = 0.0) -> np.ndarray:
    idx = np.arange(d)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    cov = cov.astype(np.float64)
    if jitter:
        cov.flat[:: d + 1] += jitter
    return cov


def make_component_means(K: int, d: int, Delta: float, seed: int, min_pairwise_distance: float = 15.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    means = [np.zeros(d, dtype=np.float64)]
    attempts = 0
    while len(means) < K:
        attempts += 1
        if attempts > 20_000:
            raise RuntimeError("Could not sample well-separated component means; lower min_pairwise_distance.")
        v = rng.normal(size=d)
        v /= np.linalg.norm(v)
        candidate = Delta * v
        distances = [np.linalg.norm(candidate - prev) for prev in means]
        if min(distances) >= min_pairwise_distance:
            means.append(candidate)
    return np.stack(means, axis=0)


def _auxiliary_rhos(K: int, mismatch_level: str, seed: int, target_rho: float = 0.1) -> list[float]:
    rng = np.random.default_rng(seed)
    n_aux = K - 1
    if mismatch_level == "0":
        return [float(target_rho)] * n_aux
    intervals = {
        "mild": ((-0.10, 0.00), (0.20, 0.30)),
        "medium": ((-0.30, -0.10), (0.30, 0.50)),
        "strong": ((-0.55, -0.30), (0.50, 0.75)),
    }
    if mismatch_level not in intervals:
        raise ValueError(f"Unknown mismatch_level={mismatch_level!r}")
    lower, upper = intervals[mismatch_level]
    n_lower = n_aux // 2
    n_upper = n_aux - n_lower
    vals = [*rng.uniform(*lower, size=n_lower), *rng.uniform(*upper, size=n_upper)]
    rng.shuffle(vals)
    return [float(v) for v in vals]


def make_component_covariances(
    K: int,
    d: int,
    scenario: str,
    rho: float | None,
    mismatch_level: str | None,
    seed: int,
    jitter: float = 0.0,
    class_varying_target_rho: float = 0.1,
) -> tuple[np.ndarray, list[float]]:
    if scenario == "shared":
        if rho is None:
            raise ValueError("rho is required for the shared covariance scenario.")
        rhos = [float(rho)] * K
    elif scenario == "mismatch":
        level = "0" if mismatch_level is None else mismatch_level
        rhos = [float(class_varying_target_rho)] + _auxiliary_rhos(K, level, seed, class_varying_target_rho)
    else:
        raise ValueError(f"Unknown covariance scenario={scenario!r}")
    covariances = np.stack([make_ar1_covariance(d, r, jitter=jitter) for r in rhos], axis=0)
    return covariances, rhos


def make_gaussian_mixture_spec(
    K: int = 3,
    d: int = 100,
    Delta: float = 20.0,
    seed: int = 0,
    target_class: int = 1,
    min_pairwise_mean_distance: float = 15.0,
    covariance_scenario: str = "shared",
    rho: float | None = 0.2,
    mismatch_level: str | None = None,
    jitter: float = 0.0,
    class_varying_target_rho: float = 0.1,
) -> GaussianMixtureSpec:
    means = make_component_means(K, d, Delta, seed, min_pairwise_mean_distance)
    covs, rhos = make_component_covariances(K, d, covariance_scenario, rho, mismatch_level, seed + 10_000, jitter, class_varying_target_rho)
    return GaussianMixtureSpec(means=means, covariances=covs, rhos=rhos, target_class=target_class)


def sample_component(spec: GaussianMixtureSpec, class_index: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(spec.means[class_index], spec.covariances[class_index], size=n).astype(np.float32)


def sample_labeled_mixture(spec: GaussianMixtureSpec, counts_by_class: dict[int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for class_index, n in counts_by_class.items():
        xs.append(sample_component(spec, class_index, n, seed + 97 * (class_index + 1)))
        ys.append(np.full(n, class_index, dtype=np.int64))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    rng = np.random.default_rng(seed + 999)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def build_low_target_data_split(spec: GaussianMixtureSpec, n_target_train: int, n_aux_train: int, n_test_target: int, seed: int) -> dict[str, np.ndarray]:
    target_idx = spec.target_index
    target_train = sample_component(spec, target_idx, n_target_train, seed + 1)
    counts = {target_idx: n_target_train}
    counts.update({k: n_aux_train for k in range(spec.K) if k != target_idx})
    cond_x, cond_y = sample_labeled_mixture(spec, counts, seed + 2)
    # Replace the sampled target subset with the exact target_train set required by the design.
    non_target = cond_y != target_idx
    aux_x, aux_y = cond_x[non_target], cond_y[non_target]
    cond_x = np.concatenate([target_train, aux_x], axis=0)
    cond_y = np.concatenate([np.full(n_target_train, target_idx, dtype=np.int64), aux_y], axis=0)
    rng = np.random.default_rng(seed + 3)
    perm = rng.permutation(len(cond_y))
    target_val = sample_component(spec, target_idx, max(1024, min(n_test_target, 5000)), seed + 4)
    target_test = sample_component(spec, target_idx, n_test_target, seed + 5)
    return {
        "uncond_train_x": target_train,
        "cond_train_x": cond_x[perm],
        "cond_train_y": cond_y[perm],
        "target_val_x": target_val,
        "target_test_x": target_test,
    }


def build_same_total_budget_split(spec: GaussianMixtureSpec, n: int, n_test_target: int, seed: int) -> dict[str, np.ndarray]:
    target_idx = spec.target_index
    pool = sample_component(spec, target_idx, spec.K * n, seed + 11)
    cond_target = pool[:n]
    uncond_train = pool[: spec.K * n]
    xs = [cond_target]
    ys = [np.full(n, target_idx, dtype=np.int64)]
    for k in range(spec.K):
        if k == target_idx:
            continue
        xs.append(sample_component(spec, k, n, seed + 101 + k))
        ys.append(np.full(n, k, dtype=np.int64))
    cond_x = np.concatenate(xs, axis=0)
    cond_y = np.concatenate(ys, axis=0)
    rng = np.random.default_rng(seed + 12)
    perm = rng.permutation(len(cond_y))
    target_val = sample_component(spec, target_idx, max(1024, min(n_test_target, 5000)), seed + 13)
    target_test = sample_component(spec, target_idx, n_test_target, seed + 14)
    return {
        "uncond_train_x": uncond_train,
        "cond_train_x": cond_x[perm],
        "cond_train_y": cond_y[perm],
        "target_val_x": target_val,
        "target_test_x": target_test,
    }
