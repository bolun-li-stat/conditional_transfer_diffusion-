"""Deterministic identifiers, devices, and atomic-ish CSV updates."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import pandas as pd


def stable_hash(parts: list[Any]) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def setting_id(seed: int, capacity: str, rotation_deg: int, model_type: str,
               training_steps: int, d: int, high: float, low: float) -> str:
    rotation = "common" if model_type == "target_only" else rotation_deg
    return stable_hash([seed, capacity, rotation, model_type, training_steps, d, high, low])


def device_from_name(name: str):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def upsert_csv(row: dict[str, Any], path: Path, key: str = "setting_id") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old[key].astype(str) != str(row[key])]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
