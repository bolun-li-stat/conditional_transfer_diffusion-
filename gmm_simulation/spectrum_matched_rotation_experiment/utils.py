"""Experiment identities and concurrency-safe per-seed metric storage."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from filelock import FileLock
import pandas as pd


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def training_design_payload(cfg: Any) -> dict[str, Any]:
    from config import CAPACITIES
    architecture = CAPACITIES[cfg.capacity]
    return {
        "capacity": cfg.capacity,
        "time_embedding_dim": architecture.time_embedding_dim,
        "class_embedding_dim": architecture.class_embedding_dim,
        "hidden_width": architecture.hidden_width,
        "hidden_layers": architecture.hidden_layers,
        "K": cfg.K, "d": cfg.d,
        "n_target_train": cfg.n_target_train, "n_aux_train": cfg.n_aux_train,
        "T": cfg.T, "beta_start": cfg.beta_start, "beta_end": cfg.beta_end,
        "batch_size": cfg.batch_size, "learning_rate": cfg.learning_rate,
        "training_steps": cfg.training_steps, "sampling_mode": "balanced",
        "lambda_high": cfg.spectrum.lambda_high,
        "lambda_low": cfg.spectrum.lambda_low,
    }


def training_design_id(cfg: Any) -> str:
    return canonical_hash(training_design_payload(cfg))


def design_payload(cfg: Any) -> dict[str, Any]:
    return {
        **training_design_payload(cfg),
        "n_validation": cfg.n_validation, "n_test": cfg.n_test,
        "score_risk_mc_samples": cfg.score_risk_mc_samples,
        "n_generated": cfg.n_generated,
    }


def design_id(cfg: Any) -> str:
    return canonical_hash(design_payload(cfg))


def pair_id(design_identifier: str, seed: int) -> str:
    return canonical_hash({"design_id": design_identifier, "seed": int(seed)})


def checkpoint_id(training_design_identifier: str, seed: int, model_type: str,
                  rotation_deg: int | None = None) -> str:
    payload: dict[str, Any] = {
        "training_design_id": training_design_identifier,
        "seed": int(seed), "model_type": model_type,
    }
    if model_type == "joint_conditional":
        if rotation_deg is None:
            raise ValueError("joint_conditional checkpoint requires rotation_deg")
        payload["rotation_deg"] = int(rotation_deg)
    elif model_type != "target_only":
        raise ValueError(f"Unknown model_type={model_type!r}")
    return canonical_hash(payload)


def setting_id(pair_identifier: str, model_type: str,
               rotation_deg: int | None = None) -> str:
    payload: dict[str, Any] = {"pair_id": pair_identifier, "model_type": model_type}
    if model_type == "joint_conditional":
        if rotation_deg is None:
            raise ValueError("joint_conditional setting requires rotation_deg")
        payload["rotation_deg"] = int(rotation_deg)
    elif model_type != "target_only":
        raise ValueError(f"Unknown model_type={model_type!r}")
    return canonical_hash(payload)


def identity_payload(cfg: Any) -> dict[str, Any]:
    training_identifier = training_design_id(cfg)
    scientific_identifier = design_id(cfg)
    return {
        **design_payload(cfg),
        "training_design_id": training_identifier,
        "design_id": scientific_identifier,
        "pair_id": pair_id(scientific_identifier, cfg.seed),
        "seed": cfg.seed,
    }


def seed_metrics_path(results_dir: Path, seed: int) -> Path:
    return results_dir / "metrics" / f"seed_{seed:03d}.csv"


def read_seed_metrics(results_dir: Path, seed: int) -> pd.DataFrame:
    path = seed_metrics_path(results_dir, seed)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def upsert_seed_metric(row: dict[str, Any], results_dir: Path) -> Path:
    path = seed_metrics_path(results_dir, int(row["seed"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if not old.empty:
            old = old[old["setting_id"].astype(str) != str(row["setting_id"])]
        combined = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temporary = Path(name)
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
        raise FileNotFoundError(f"No per-seed metrics under {results_dir / 'metrics'}")
    combined = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    combined.to_csv(results_dir / "metrics.csv", index=False)
    return combined


def device_from_name(name: str):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
