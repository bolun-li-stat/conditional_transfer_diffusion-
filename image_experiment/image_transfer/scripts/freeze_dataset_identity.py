"""Freeze the exact dataset inventory used by an image experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_transfer.config import load_resolved_config
from image_transfer.data.dataset_identity import freeze_dataset_identity
from image_transfer.utils.io import resolve_env_path


def _output_path(value: str | None, *, config_path: str | Path, configured: object) -> Path:
    raw = value if value is not None else str(configured or "")
    expanded = resolve_env_path(raw)
    if not expanded:
        raise ValueError("Provide --out or configure dataset_identity_path")
    path = Path(expanded).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config_path).resolve().parent / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    resolved = load_resolved_config(args.config)
    destination = _output_path(
        args.out,
        config_path=args.config,
        configured=resolved.resolved.get("dataset_identity_path"),
    )
    identity = freeze_dataset_identity(resolved.resolved, destination)
    print(json.dumps({
        "dataset_identity_path": str(destination),
        "dataset_identity_hash": identity["dataset_identity_hash"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "inventory_count": identity["inventory_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

