from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


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


def test_job_grid_limit_refuses_to_write_without_explicit_override(tmp_path: Path):
    destination = tmp_path / "too_many.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "image_transfer.scripts.make_job_grid",
            "--experiment",
            "A",
            "--config",
            FAKE_CONFIG,
            "--out",
            str(destination),
            "--max-jobs",
            "2",
        ],
        text=True,
        capture_output=True,
        env={**os.environ, "RESULTS_ROOT": str(tmp_path / "results")},
    )
    assert completed.returncode != 0
    assert "Refusing to write" in completed.stderr
    assert not destination.exists()


def test_offline_cpu_pipeline_to_atomic_results_aggregation_and_plots(tmp_path: Path):
    results = tmp_path / "results"
    env = {
        "RESULTS_ROOT": str(results),
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
    }
    preflight_dir = tmp_path / "preflight"
    run_cmd(
        [
            sys.executable,
            "-m",
            "image_transfer.scripts.preflight_experiment",
            "--config",
            FAKE_CONFIG,
            "--out-dir",
            str(preflight_dir),
        ],
        env=env,
    )
    preflight = json.loads((preflight_dir / "preflight_report.json").read_text(encoding="utf-8"))
    assert preflight["status"] == "ready"
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
    assert all(record["status"] == "completed" and record["schema_version"] == 2 for record in records)
    assert len({record["metadata"]["manifest_hash"] for record in records}) == 1
    assert len({record["metadata"]["target_eval_indices_hash"] for record in records}) == 1
    assert len({record["metadata"]["target_training_subset_hash"] for record in records}) == 1
    assert all("fid_target" not in record["metrics"] for record in records)
    assert all("debug_pooled_pixel_distance" in record["metrics"] for record in records)
    assert all("train_epsilon_mse_target" in record["metrics"] for record in records)
    assert all("validation_epsilon_mse_target" in record["metrics"] for record in records)
    assert all("test_epsilon_mse_target" in record["metrics"] for record in records)
    assert all("train_validation_mse_gap" in record["metrics"] for record in records)
    assert all("train_test_mse_gap" in record["metrics"] for record in records)
    assert all("final_pooled_train_loss" in record["training"] for record in records)
    assert all("rolling_target_train_loss" in record["training"] for record in records)
    assert all(
        len(
            {
                record["metrics"]["train_corruption_bank_hash"],
                record["metrics"]["validation_corruption_bank_hash"],
                record["metrics"]["test_corruption_bank_hash"],
            }
        )
        == 3
        for record in records
    )
    assert all(Path(record["metadata"]["last_checkpoint_path"]).exists() for record in records)
    assert all(Path(record["metadata"]["best_checkpoint_path"]).exists() for record in records)
    assert all(Path(record["metadata"]["train_corruption_bank_path"]).exists() for record in records)
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
        "subset_level_summaries.csv",
        "target_level_summaries.csv",
        "hierarchical_summaries.csv",
        "job_completeness.csv",
        "failed_jobs.csv",
        "environment_summary.csv",
        "readiness_summary.json",
    ):
        assert (results / filename).exists()
    paired_text = (results / "paired_transfer_gaps.csv").read_text(encoding="utf-8")
    assert "conditional_target_only_n0" in paired_text
    assert "unconditional_n0" in paired_text
    assert list((results / "figures").glob("*.png"))


def test_v2_unequal_holdout_and_subset_seeds_run_grid_to_result(tmp_path: Path):
    """Regress the v2 seed bug through grid, builder, training, and result IO."""

    results_root = tmp_path / "unequal-seed-results"
    raw = yaml.safe_load(Path(FAKE_CONFIG).read_text(encoding="utf-8"))
    raw["output_root"] = str(results_root)
    raw["data_split"]["manifest_root"] = str(results_root / "manifests")
    raw["data_split"]["holdout_seed"] = 100
    raw["data_split"].pop("training_subset_seed", None)
    raw["seed_design"]["holdout_seed"] = 100
    raw["seed_design"]["training_subset_seeds"] = [0, 1]
    raw["fake_data_size"] = 64
    raw["data_split"]["target_eval_size"] = 4
    raw["data_split"]["target_val_size"] = 2
    raw["n0_values"] = [4]
    raw["experiments"]["A"]["n0_values"] = [4]
    raw["training"]["steps"] = 1
    raw["sampling"]["sampler"] = "ddim"
    raw["sampling"]["steps"] = 2
    raw["sampling"]["batch_size"] = 1
    raw["num_generated"] = 1
    raw["evaluation"]["real_eval_max"] = 4
    raw["evaluation"]["train_diagnostic_max_images"] = 4
    config_path = tmp_path / "unequal-seed-fake.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    jobs_path = tmp_path / "unequal-seed-jobs.csv"
    env = {
        "RESULTS_ROOT": str(results_root),
        "MPLCONFIGDIR": str(tmp_path / "matplotlib-unequal-seed"),
    }
    run_cmd(
        [
            sys.executable,
            "-m",
            "image_transfer.scripts.make_job_grid",
            "--experiment",
            "A",
            "--config",
            str(config_path),
            "--out",
            str(jobs_path),
        ],
        env=env,
    )
    with jobs_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 6
    assert {row["data_split_seed"] for row in rows} == {""}
    assert {int(row["holdout_seed"]) for row in rows} == {100}
    assert {int(row["training_subset_seed"]) for row in rows} == {0, 1}

    selected_indices = [
        index
        for index, row in enumerate(rows)
        if row["model_type"] in {"conditional_target_only_n0", "conditional_close"}
    ]
    assert len(selected_indices) == 4
    for index in selected_indices:
        run_cmd(
            [
                sys.executable,
                "-m",
                "image_transfer.scripts.run_job",
                "--jobs-csv",
                str(jobs_path),
                "--job-index",
                str(index),
                "--device",
                "cpu",
                "--force",
            ],
            env=env,
        )

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((results_root / "run_results").glob("*.json"))
    ]
    assert len(records) == 4
    assert all(record["status"] == "completed" for record in records)
    assert all(record["training"]["optimizer_steps"] == 1 for record in records)
    assert all(Path(record["metadata"]["last_checkpoint_path"]).is_file() for record in records)
    assert {record["metadata"]["holdout_seed"] for record in records} == {100}
    assert {record["metadata"]["training_subset_seed"] for record in records} == {0, 1}
    assert {record["job"]["data_split_seed"] for record in records} == {""}
    assert len({record["metadata"]["split_manifest_hash"] for record in records}) == 1
    assert len({record["metadata"]["target_eval_indices_hash"] for record in records}) == 1
    assert len(
        {record["metadata"]["target_validation_indices_hash"] for record in records}
    ) == 1
    assert len(
        {record["metadata"]["target_similarity_reference_hash"] for record in records}
    ) == 1

    by_subset: dict[int, list[dict]] = {0: [], 1: []}
    target_indices: dict[int, tuple[int, ...]] = {}
    for record in records:
        subset_seed = int(record["metadata"]["training_subset_seed"])
        by_subset[subset_seed].append(record)
        subset_manifest = json.loads(
            Path(record["metadata"]["subset_manifest_path"]).read_text(encoding="utf-8")
        )
        n0 = int(record["job"]["n0"])
        indices = tuple(str(value) for value in subset_manifest["target_training_order"][:n0])
        if subset_seed in target_indices:
            assert target_indices[subset_seed] == indices
        target_indices[subset_seed] = indices

    for paired_records in by_subset.values():
        assert len(paired_records) == 2
        assert {record["job"]["model_type"] for record in paired_records} == {
            "conditional_target_only_n0",
            "conditional_close",
        }
        assert len(
            {record["metadata"]["subset_manifest_hash"] for record in paired_records}
        ) == 1
        assert len(
            {record["metadata"]["target_training_subset_hash"] for record in paired_records}
        ) == 1

    assert target_indices[0] != target_indices[1]
    assert len({record["metadata"]["subset_manifest_hash"] for record in records}) == 2
    assert len(
        {record["metadata"]["target_training_subset_hash"] for record in records}
    ) == 2
