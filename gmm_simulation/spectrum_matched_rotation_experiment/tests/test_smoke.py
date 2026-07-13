from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from config import smoke_config
import train
from train import run_setting

def test_smoke_training(tmp_path: Path):
    cfg = smoke_config(tmp_path)
    target = run_setting(cfg, "target_only", skip_generation=True)
    joint = run_setting(cfg, "joint_conditional", skip_generation=True)
    assert target["model_type"] == "target_only"
    assert joint["model_type"] == "joint_conditional"
    assert (tmp_path / "metrics/seed_000.csv").exists()


def test_generation_only_updates_row_without_training(tmp_path: Path, monkeypatch):
    cfg = smoke_config(tmp_path)
    score_row = run_setting(cfg, "target_only", skip_generation=True)
    original_score = score_row["score_risk"]

    def forbidden_training(*args, **kwargs):
        raise AssertionError("generation-only attempted training")
    monkeypatch.setattr(train, "train_one", forbidden_training)
    generated = run_setting(cfg, "target_only", resume=True, generation_only=True)
    assert generated["score_risk"] == original_score
    assert np.isfinite(generated["gaussian_w2_squared"])
    assert bool(generated["generation_evaluation_complete"])
    stored = pd.read_csv(tmp_path / "metrics/seed_000.csv")
    assert len(stored) == 1 and np.isfinite(stored.iloc[0].gaussian_w2_squared)


def test_generation_only_missing_checkpoint_fails(tmp_path: Path):
    cfg = smoke_config(tmp_path)
    row = run_setting(cfg, "target_only", skip_generation=True)
    Path(row["checkpoint_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="requires checkpoint"):
        run_setting(cfg, "target_only", resume=True, generation_only=True)
