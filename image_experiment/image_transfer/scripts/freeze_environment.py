"""Freeze the environment installed on the current execution node."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from image_transfer.environment_lock import build_exact_environment_lock
from image_transfer.utils.io import atomic_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_environment(source_spec: str | Path, out: str | Path) -> dict:
    source = Path(source_spec).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"environment source specification does not exist: {source}")
    payload = build_exact_environment_lock(
        source_spec_path=source,
        source_spec_hash=_sha256(source),
    )
    atomic_write_json(payload, out)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = freeze_environment(args.source_spec, args.out)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
