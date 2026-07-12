"""Spectrum-matched covariances and deterministic paired Gaussian data."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CovarianceFamily:
    target: np.ndarray
    auxiliary_1: np.ndarray
    auxiliary_2: np.ndarray
    rotation: np.ndarray


def canonical_orthogonal(d: int, seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed + 10_003).standard_normal((d, d)))
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs


def spectrum(d: int, high: float = 1.8, low: float = 0.2) -> np.ndarray:
    if d <= 0 or d % 2:
        raise ValueError("d must be positive and even")
    return np.concatenate([np.full(d // 2, high), np.full(d // 2, low)])


def paired_rotation(d: int, theta_deg: float) -> np.ndarray:
    if d <= 0 or d % 2:
        raise ValueError("d must be positive and even")
    h = d // 2
    c, s = np.cos(np.deg2rad(theta_deg)), np.sin(np.deg2rad(theta_deg))
    eye = np.eye(h)
    return np.block([[c * eye, -s * eye], [s * eye, c * eye]])


def _sym(x: np.ndarray) -> np.ndarray:
    return (x + x.T) / 2.0


def covariance_family(d: int, seed: int, theta_deg: float, high: float = 1.8,
                      low: float = 0.2) -> CovarianceFamily:
    u = canonical_orthogonal(d, seed)
    lam = np.diag(spectrum(d, high, low))
    r = paired_rotation(d, theta_deg)
    target = _sym(u @ lam @ u.T)
    aux1 = _sym(u @ r @ lam @ r.T @ u.T)
    rm = paired_rotation(d, -theta_deg)
    aux2 = _sym(u @ rm @ lam @ rm.T @ u.T)
    family = CovarianceFamily(target, aux1, aux2, r)
    validate_covariance_family(family, theta_deg)
    return family


def validate_covariance_family(family: CovarianceFamily, theta_deg: float,
                               atol: float = 1e-9) -> None:
    covs = [family.target, family.auxiliary_1, family.auxiliary_2]
    base_eigs = np.linalg.eigvalsh(covs[0])
    for cov in covs:
        assert np.allclose(cov, cov.T, atol=atol)
        eigs = np.linalg.eigvalsh(cov)
        assert eigs.min() > 0
        assert np.allclose(eigs, base_eigs, atol=atol, rtol=atol)
        assert np.isclose(np.trace(cov), np.trace(covs[0]), atol=atol)
        assert np.isclose(np.linalg.slogdet(cov)[1], np.linalg.slogdet(covs[0])[1], atol=atol)
        assert np.isclose(np.linalg.cond(cov), np.linalg.cond(covs[0]), atol=1e-7)
    assert np.allclose(family.rotation.T @ family.rotation, np.eye(len(family.rotation)), atol=atol)
    if theta_deg == 0:
        assert np.allclose(covs[0], covs[1], atol=atol)
        assert np.allclose(covs[0], covs[2], atol=atol)


def covariance_sqrt(cov: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(_sym(cov))
    if values.min() <= 0:
        raise ValueError("covariance is not SPD")
    return _sym((vectors * np.sqrt(values)) @ vectors.T)


def build_paired_split(d: int, seed: int, theta_deg: float, n_target: int,
                       n_aux: int, n_validation: int, n_test: int,
                       high: float = 1.8, low: float = 0.2) -> dict[str, np.ndarray]:
    fam = covariance_family(d, seed, theta_deg, high, low)
    # Target streams never include theta. Auxiliary streams reuse common Z across theta.
    target_rng = np.random.default_rng(seed + 20_011)
    target_z = target_rng.standard_normal((n_target + n_validation + n_test, d))
    target_all = target_z @ covariance_sqrt(fam.target).T
    aux_z = np.random.default_rng(seed + 30_013).standard_normal((2, n_aux, d))
    aux1 = aux_z[0] @ covariance_sqrt(fam.auxiliary_1).T
    aux2 = aux_z[1] @ covariance_sqrt(fam.auxiliary_2).T
    x = np.concatenate([target_all[:n_target], aux1, aux2]).astype(np.float32)
    y = np.concatenate([np.zeros(n_target), np.ones(n_aux), np.full(n_aux, 2)]).astype(np.int64)
    return {"target_train": target_all[:n_target].astype(np.float32), "joint_x": x,
            "joint_y": y, "target_val": target_all[n_target:n_target+n_validation].astype(np.float32),
            "target_test": target_all[-n_test:].astype(np.float32)}
