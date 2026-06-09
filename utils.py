"""General utilities for reproducible experiments, I/O, and numerical linear algebra."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def describe_device() -> str:
    available = torch.cuda.is_available()
    if available:
        name = torch.cuda.get_device_name(0)
        msg = f"CUDA available: True | GPU: {name}"
    else:
        msg = "CUDA available: False | using CPU"
    print(msg)
    return msg


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict[str, Any], path: Path | str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, indent=2, sort_keys=True))


def load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def stable_hash(parts: Iterable[Any], length: int = 12) -> str:
    text = "|".join(str(p) for p in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def append_row_to_csv(row: dict[str, Any], csv_path: Path | str, columns: list[str]) -> None:
    p = Path(csv_path)
    ensure_dir(p.parent)
    df = pd.DataFrame([{col: row.get(col, None) for col in columns}])
    df.to_csv(p, mode="a", index=False, header=not p.exists())


def read_existing_results(csv_path: Path | str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def covariance_np(x: np.ndarray) -> np.ndarray:
    return np.cov(np.asarray(x), rowvar=False, bias=False)


def symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def matrix_sqrt_psd(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    vals, vecs = np.linalg.eigh(symmetrize(a))
    vals = np.clip(vals, eps, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(np.nanstd(values, ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(values))))
