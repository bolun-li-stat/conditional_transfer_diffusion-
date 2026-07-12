from pathlib import Path
from config import smoke_config
from train import run_setting

def test_smoke_training(tmp_path: Path):
    cfg = smoke_config(tmp_path)
    target = run_setting(cfg, "target_only", skip_generation=True)
    joint = run_setting(cfg, "joint_conditional", skip_generation=True)
    assert target["model_type"] == "target_only"
    assert joint["model_type"] == "joint_conditional"
    assert (tmp_path / "metrics.csv").exists()
