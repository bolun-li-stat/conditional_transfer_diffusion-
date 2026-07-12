"""The relocated legacy CLI remains runnable from its own directory."""
from pathlib import Path
import os
import subprocess
import sys


def test_original_ar1_smoke_command(tmp_path: Path):
    old = Path(__file__).resolve().parents[2] / "original_ar1_experiment"
    env = dict(os.environ)
    completed = subprocess.run(
        [sys.executable, "train.py", "--experiment", "smoke", "--training-steps", "1",
         "--n-generated", "1", "--score-risk-mc-samples", "4",
         "--results-dir", str(tmp_path)], cwd=old, env=env, capture_output=True, text=True,
        timeout=180, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
