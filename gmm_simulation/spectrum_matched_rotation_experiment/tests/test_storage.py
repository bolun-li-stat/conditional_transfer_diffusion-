from pathlib import Path

import pandas as pd

from utils import consolidate_seed_metrics, seed_metrics_path, upsert_seed_metric


def test_different_seed_files_consolidate(tmp_path: Path):
    upsert_seed_metric({"seed": 0, "setting_id": "a", "value": 1}, tmp_path)
    upsert_seed_metric({"seed": 1, "setting_id": "b", "value": 2}, tmp_path)
    combined = consolidate_seed_metrics(tmp_path)
    assert set(combined.seed) == {0, 1}
    assert seed_metrics_path(tmp_path, 0).exists() and seed_metrics_path(tmp_path, 1).exists()


def test_same_seed_upsert_has_no_duplicate(tmp_path: Path):
    upsert_seed_metric({"seed": 0, "setting_id": "a", "value": 1}, tmp_path)
    upsert_seed_metric({"seed": 0, "setting_id": "a", "value": 9}, tmp_path)
    frame = pd.read_csv(seed_metrics_path(tmp_path, 0))
    assert len(frame) == 1 and frame.iloc[0].value == 9


def test_atomic_writer_leaves_no_partial_file(tmp_path: Path):
    upsert_seed_metric({"seed": 0, "setting_id": "a", "value": 1}, tmp_path)
    path = seed_metrics_path(tmp_path, 0)
    assert pd.read_csv(path).iloc[0].setting_id == "a"
    assert list(path.parent.glob("*.tmp")) == []
