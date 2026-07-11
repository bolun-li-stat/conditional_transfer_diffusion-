from __future__ import annotations

from pathlib import Path

import pytest

from image_transfer.data.dataset_identity import (
    DatasetIdentityError,
    freeze_dataset_identity,
    load_dataset_identity,
    verify_dataset_identity_file,
)


def _config(root: Path) -> dict:
    return {
        "dataset": "imagenet64",
        "use_fake_data": False,
        "data_root": str(root),
        "targets": [{
            "name": "target",
            "synset": "target",
            "auxiliary_sets": {"close": ["auxiliary"]},
        }],
        "data_split": {"eval_source": "train_holdout"},
    }


def test_frozen_identity_is_portable_and_detects_content_and_inventory_changes(tmp_path: Path):
    root = tmp_path / "images"
    for label, value in (("target", b"target-v1"), ("auxiliary", b"aux-v1")):
        directory = root / "train" / label
        directory.mkdir(parents=True)
        (directory / "image.bin").write_bytes(value)
    identity_path = tmp_path / "dataset-identity.json"
    frozen = freeze_dataset_identity(_config(root), identity_path)
    loaded = load_dataset_identity(identity_path)

    assert loaded == frozen
    assert all(not Path(item["relative_path"]).is_absolute() for item in loaded["inventory"])
    assert verify_dataset_identity_file(_config(root), identity_path)["dataset_identity_hash"] == frozen[
        "dataset_identity_hash"
    ]

    (root / "train" / "target" / "image.bin").write_bytes(b"target-v2")
    with pytest.raises(DatasetIdentityError, match="bytes or inventory"):
        verify_dataset_identity_file(_config(root), identity_path)

    (root / "train" / "target" / "image.bin").write_bytes(b"target-v1")
    (root / "train" / "target" / "additional.bin").write_bytes(b"new")
    with pytest.raises(DatasetIdentityError, match="bytes or inventory"):
        verify_dataset_identity_file(_config(root), identity_path)


def test_identity_requires_coverage_for_every_configured_class(tmp_path: Path):
    target = tmp_path / "train" / "target"
    target.mkdir(parents=True)
    (target / "image.bin").write_bytes(b"target")
    with pytest.raises(DatasetIdentityError, match="no training files"):
        freeze_dataset_identity(_config(tmp_path), tmp_path / "identity.json")
