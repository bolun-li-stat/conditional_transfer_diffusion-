from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


FAKE_CONFIG = "image_transfer/configs/cifar10_fake_smoke.yaml"


def run_cmd(cmd, env=None):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(cmd, check=True, text=True, capture_output=True, env=merged)


def _grid(tmp_path: Path, experiment: str, env: dict[str, str]) -> tuple[Path, list[dict[str, str]]]:
    path = tmp_path / f"{experiment}.csv"
    run_cmd(
        [
            sys.executable,
            "-m",
            "image_transfer.scripts.make_job_grid",
            "--experiment",
            experiment,
            "--config",
            FAKE_CONFIG,
            "--out",
            str(path),
        ],
        env=env,
    )
    with open(path, newline="", encoding="utf-8") as handle:
        return path, list(csv.DictReader(handle))


def test_fake_smoke_job_grid_has_architecture_baselines_without_duplicates(tmp_path: Path):
    env = {"RESULTS_ROOT": str(tmp_path / "results")}
    expected = {"A": 3, "B": 3, "C": 4}
    expected_baselines = {
        "A": {"unconditional_n0", "conditional_target_only_n0"},
        "B": {"unconditional_equal_total", "conditional_target_only_equal_total"},
        "C": {"unconditional_n0", "conditional_target_only_n0"},
    }
    for experiment, count in expected.items():
        _, rows = _grid(tmp_path, experiment, env)
        assert len(rows) == count
        baselines = [row for row in rows if row["model_type"] in expected_baselines[experiment]]
        assert {row["model_type"] for row in baselines} == expected_baselines[experiment]
        assert len(baselines) == 2
        assert len({row["run_id"] for row in rows}) == len(rows)


def test_offline_cpu_pipeline_to_atomic_results_aggregation_and_plots(tmp_path: Path):
    results = tmp_path / "results"
    env = {
        "RESULTS_ROOT": str(results),
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
    }
    jobs, rows = _grid(tmp_path, "A", env)

    # Run both baselines and the auxiliary model, producing a complete primary
    # and legacy paired comparison without any network access.
    for index in range(len(rows)):
        run_cmd(
            [
                sys.executable,
                "-m",
                "image_transfer.scripts.run_job",
                "--jobs-csv",
                str(jobs),
                "--job-index",
                str(index),
                "--device",
                "cpu",
                "--force",
            ],
            env=env,
        )

    result_paths = sorted((results / "run_results").glob("*.json"))
    assert len(result_paths) == 3
    records = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    assert all(record["status"] == "completed" and record["schema_version"] == 1 for record in records)
    assert len({record["metadata"]["manifest_hash"] for record in records}) == 1
    assert len({record["metadata"]["target_eval_indices_hash"] for record in records}) == 1
    assert len({record["metadata"]["target_training_subset_hash"] for record in records}) == 1
    assert all("fid_target" not in record["metrics"] for record in records)
    assert all("debug_pooled_pixel_distance" in record["metrics"] for record in records)
    assert all(Path(record["metadata"]["last_checkpoint_path"]).exists() for record in records)
    assert all(Path(record["metadata"]["best_checkpoint_path"]).exists() for record in records)
    assert not list(results.rglob("metrics.csv"))

    # A terminal result is authoritative: --resume must not overwrite its
    # metrics with a zero-step continuation.
    terminal_before = result_paths[0].read_bytes()
    terminal_index = next(
        index for index, row in enumerate(rows) if row["run_id"] == result_paths[0].stem
    )
    run_cmd(
        [
            sys.executable,
            "-m",
            "image_transfer.scripts.run_job",
            "--jobs-csv",
            str(jobs),
            "--job-index",
            str(terminal_index),
            "--device",
            "cpu",
            "--resume",
        ],
        env=env,
    )
    assert result_paths[0].read_bytes() == terminal_before

    conflicting = subprocess.run(
        [
            sys.executable,
            "-m",
            "image_transfer.scripts.run_job",
            "--jobs-csv",
            str(jobs),
            "--job-index",
            str(terminal_index),
            "--force",
            "--resume",
        ],
        text=True,
        capture_output=True,
        env={**os.environ, **env},
    )
    assert conflicting.returncode != 0
    assert "not allowed with argument" in conflicting.stderr

    # A forced rerun starts from clean per-run artifacts rather than appending
    # duplicate training-log rows.
    run_cmd(
        [
            sys.executable,
            "-m",
            "image_transfer.scripts.run_job",
            "--jobs-csv",
            str(jobs),
            "--job-index",
            str(terminal_index),
            "--device",
            "cpu",
            "--force",
        ],
        env=env,
    )
    log_path = results / "A_equal_target" / "logs" / f"{result_paths[0].stem}_train_log.csv"
    with open(log_path, newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2

    run_cmd(
        [
            sys.executable,
            "-m",
            "image_transfer.scripts.aggregate_results",
            "--results-root",
            str(results),
            "--expected-jobs",
            str(jobs),
        ],
        env=env,
    )
    run_cmd(
        [
            sys.executable,
            "-m",
            "image_transfer.scripts.plot_results",
            "--results-root",
            str(results),
            "--experiment",
            "A",
        ],
        env=env,
    )

    for filename in (
        "all_metrics.csv",
        "summary_metrics.csv",
        "paired_transfer_gaps.csv",
        "job_completeness.csv",
        "failed_jobs.csv",
    ):
        assert (results / filename).exists()
    paired_text = (results / "paired_transfer_gaps.csv").read_text(encoding="utf-8")
    assert "conditional_target_only_n0" in paired_text
    assert "unconditional_n0" in paired_text
    assert list((results / "figures").glob("*.png"))
