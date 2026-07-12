"""Auditable identities and concurrency-safe per-seed metric storage."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from filelock import FileLock
import pandas as pd


PAIR_FIELDS = (
    "seed", "capacity", "K", "d", "n_target_train", "n_aux_train",
    "n_validation", "n_test", "T", "beta_start", "beta_end", "batch_size",
    "learning_rate", "training_steps", "lambda_high", "lambda_low",
    "score_risk_mc_samples", "n_generated", "generation_enabled",
)


def canonical_hash(payload: Mapping[str, Any]) -> str:
    """Hash a canonical, human-auditable JSON mapping."""
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def pair_payload(cfg: Any, generation_enabled: bool) -> dict[str, Any]:
    payload = {
        "seed": cfg.seed, "capacity": cfg.capacity, "K": cfg.K, "d": cfg.d,
        "n_target_train": cfg.n_target_train, "n_aux_train": cfg.n_aux_train,
        "n_validation": cfg.n_validation, "n_test": cfg.n_test, "T": cfg.T,
        "beta_start": cfg.beta_start, "beta_end": cfg.beta_end,
        "batch_size": cfg.batch_size, "learning_rate": cfg.learning_rate,
        "training_steps": cfg.training_steps,
        "lambda_high": cfg.spectrum.lambda_high,
        "lambda_low": cfg.spectrum.lambda_low,
        "score_risk_mc_samples": cfg.score_risk_mc_samples,
        "n_generated": cfg.n_generated,
        "generation_enabled": bool(generation_enabled),
    }
    assert tuple(payload) == PAIR_FIELDS
    return payload


def pair_id(cfg: Any, generation_enabled: bool) -> str:
    return canonical_hash(pair_payload(cfg, generation_enabled))


def setting_id(pair_identifier: str, model_type: str,
               rotation_deg: int | None = None) -> str:
    if model_type not in {"target_only", "joint_conditional"}:
        raise ValueError(f"Unknown model_type={model_type!r}")
    payload: dict[str, Any] = {"pair_id": pair_identifier, "model_type": model_type}
    if model_type == "joint_conditional":
        if rotation_deg is None:
            raise ValueError("joint_conditional setting_id requires rotation_deg")
        payload["rotation_deg"] = int(rotation_deg)
    return canonical_hash(payload)


def seed_metrics_path(results_dir: Path, seed: int) -> Path:
    return results_dir / "metrics" / f"seed_{seed:03d}.csv"


def read_seed_metrics(results_dir: Path, seed: int) -> pd.DataFrame:
    path = seed_metrics_path(results_dir, seed)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def upsert_seed_metric(row: dict[str, Any], results_dir: Path) -> Path:
    """Atomically upsert one row while locking only its seed file."""
    path = seed_metrics_path(results_dir, int(row["seed"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path) + ".lock")
    with lock:
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if not old.empty:
            old = old[old["setting_id"].astype(str) != str(row["setting_id"])]
        combined = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                               dir=path.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            combined.to_csv(temporary, index=False)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return path


def consolidate_seed_metrics(results_dir: Path) -> pd.DataFrame:
    paths = sorted((results_dir / "metrics").glob("seed_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No per-seed metrics found under {results_dir / 'metrics'}")
    frames = [pd.read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(results_dir / "metrics.csv", index=False)
    return combined


def device_from_name(name: str):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
