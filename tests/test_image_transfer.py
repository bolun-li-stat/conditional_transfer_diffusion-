from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, env=None):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(cmd, check=True, text=True, capture_output=True, env=merged)


def test_make_job_grid_abc(tmp_path: Path):
    env = {"RESULTS_ROOT": str(tmp_path / "results")}
    counts = {"A": 30, "B": 10, "C": 12}
    for exp, expected in counts.items():
        out = tmp_path / f"{exp}.csv"
        run_cmd([sys.executable, "-m", "image_transfer.scripts.make_job_grid", "--experiment", exp, "--config", "image_transfer/configs/cifar10_sanity.yaml", "--out", str(out)], env=env)
        with open(out, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == expected


def test_fake_data_train_and_run_job(tmp_path: Path):
    env = {"RESULTS_ROOT": str(tmp_path / "results")}
    jobs = tmp_path / "jobs.csv"
    run_cmd([sys.executable, "-m", "image_transfer.scripts.make_job_grid", "--experiment", "A", "--config", "image_transfer/configs/cifar10_sanity.yaml", "--out", str(jobs)], env=env)
    run_cmd([sys.executable, "-m", "image_transfer.scripts.run_job", "--jobs-csv", str(jobs), "--job-index", "0", "--device", "cpu", "--force"], env=env)
    assert (tmp_path / "results" / "A_equal_target" / "metrics.csv").exists()
