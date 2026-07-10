from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import ConcatDataset, TensorDataset

from image_transfer.scripts.aggregate_results import (
    aggregate_results,
    compute_paired_gaps,
    improvement_positive,
    similarity_correlations,
    t95_confidence_interval,
)
from image_transfer.scripts.plot_results import _plot_paired_axis
from image_transfer.training import trainer
from image_transfer.training.checkpointing import _torch_load
from image_transfer.utils.io import atomic_write_json, load_valid_result, write_failure_result, write_run_result


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.bias = nn.Parameter(torch.tensor(-0.1))

    def forward(self, x, t, y=None):
        label = 0.0 if y is None else y.float().view(-1, 1, 1, 1) * 0.01
        return self.scale * x + self.bias + label


class _TinyDiffusion:
    def __init__(self, timesteps=4, schedule="linear", device="cpu") -> None:
        self.timesteps = timesteps
        self.device = torch.device(device)

    def loss(self, model, x0, y=None):
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        return torch.nn.functional.mse_loss(model(x0, t, y), noise)


def _fixed_validation(model, diffusion, loader, device, label=None, **kwargs):
    value = float(sum(parameter.detach().square().sum() for parameter in model.parameters()))
    return {"all": value, "low": value + 1, "mid": value + 2, "high": value + 3}


@pytest.fixture
def tiny_training(monkeypatch):
    monkeypatch.setattr(trainer, "ImageUNet", lambda **kwargs: _TinyModel())
    monkeypatch.setattr(trainer, "ImageDDPM", _TinyDiffusion)


def _dataset(labels):
    x = torch.arange(len(labels) * 4, dtype=torch.float32).reshape(len(labels), 1, 2, 2) / 20
    return TensorDataset(x, torch.tensor(labels, dtype=torch.long))


def _train_kwargs(tmp_path: Path, checkpoint_name: str, steps: int) -> dict:
    return {
        "conditional": True,
        "num_classes": 3,
        "image_size": 2,
        "base_channels": 4,
        "channel_mults": [1],
        "timesteps": 4,
        "schedule": "linear",
        "steps": steps,
        "batch_size": 4,
        "lr": 1e-2,
        "device": "cpu",
        "ema_decay": 0.9,
        "checkpoint_path": tmp_path / checkpoint_name,
        "train_log_path": tmp_path / f"{checkpoint_name}.csv",
        "validation_interval": 2,
        "model_initialization_seed": 11,
        "training_seed": 17,
        "validation_evaluator": _fixed_validation,
        "deterministic_cpu": True,
        "config_hash": "config",
        "manifest_hash": "manifest",
        "git_sha": "deadbeef",
    }


def test_protocol_exposure_counters_and_balanced_auxiliary(tmp_path: Path, tiny_training):
    target = _dataset([0, 0, 0])
    auxiliary = _dataset([1, 1, 2, 2])
    pooled = ConcatDataset([target, auxiliary])

    _, _, conditional_metrics = trainer.train_image_model(
        pooled,
        target,
        **_train_kwargs(tmp_path, "conditional.pt", 5),
        target_dataset=target,
        auxiliary_dataset=auxiliary,
        training_protocol="target_exposure_matched",
        target_batch_size=4,
        auxiliary_batch_size=3,
        auxiliary_loss_weight=0.5,
    )
    _, _, baseline_metrics = trainer.train_image_model(
        target,
        target,
        **_train_kwargs(tmp_path, "baseline.pt", 5),
        target_dataset=target,
        training_protocol="target_exposure_matched",
        target_batch_size=4,
    )

    assert conditional_metrics["target_examples_seen"] == baseline_metrics["target_examples_seen"] == 20
    assert conditional_metrics["auxiliary_examples_seen"] == 15
    assert conditional_metrics["total_examples_seen"] == 35
    by_class = conditional_metrics["auxiliary_examples_seen_by_class"]
    assert max(by_class.values()) - min(by_class.values()) <= 1
    assert conditional_metrics["auxiliary_loss_weight"] == 0.5


def test_natural_protocol_counts_actual_pooled_exposure(tmp_path: Path, tiny_training):
    target = _dataset([0, 0])
    auxiliary = _dataset([1, 1, 1, 2, 2, 2])
    pooled = ConcatDataset([target, auxiliary])
    kwargs = _train_kwargs(tmp_path, "natural.pt", 4)
    kwargs["batch_size"] = 8
    _, _, metrics = trainer.train_image_model(
        pooled,
        target,
        **kwargs,
        training_protocol="natural_compute_matched",
    )
    assert metrics["target_examples_seen"] == 8
    assert metrics["auxiliary_examples_seen"] == 24
    assert metrics["effective_target_fraction"] == pytest.approx(0.25)


def test_checkpoint_resume_matches_continuous_cpu_training(tmp_path: Path, tiny_training):
    data = _dataset([0, 0, 1, 1, 2, 2])
    target = _dataset([0, 0, 0])
    continuous_kwargs = _train_kwargs(tmp_path, "continuous.pt", 6)
    trainer.train_image_model(data, target, **continuous_kwargs)

    split_kwargs = _train_kwargs(tmp_path, "resumed.pt", 3)
    trainer.train_image_model(data, target, **split_kwargs)
    resume_kwargs = _train_kwargs(tmp_path, "resumed.pt", 6)
    trainer.train_image_model(data, target, **resume_kwargs, resume=True)

    continuous = _torch_load(tmp_path / "continuous_last.pt")
    resumed = _torch_load(tmp_path / "resumed_last.pt")
    assert continuous["step"] == resumed["step"] == 6
    for section in ("raw_model_state", "ema_model_state"):
        assert continuous[section].keys() == resumed[section].keys()
        for name in continuous[section]:
            torch.testing.assert_close(continuous[section][name], resumed[section][name], rtol=0, atol=0)
    continuous_metadata = dict(continuous["training_protocol_metadata"])
    resumed_metadata = dict(resumed["training_protocol_metadata"])
    continuous_wallclock = continuous_metadata.pop("wallclock_train_seconds")
    resumed_wallclock = resumed_metadata.pop("wallclock_train_seconds")
    assert continuous_metadata == resumed_metadata
    assert continuous_wallclock > 0 and resumed_wallclock > 0
    assert resumed["optimizer_state"] is not None
    assert "python" in resumed["rng_states"] and "numpy" in resumed["rng_states"]
    assert resumed["config_hash"] == "config"
    assert resumed["manifest_hash"] == "manifest"
    assert resumed["git_sha"] == "deadbeef"
    assert (tmp_path / "resumed_best.pt").exists()

    with open(tmp_path / "continuous.pt.csv", newline="", encoding="utf-8") as handle:
        continuous_log = list(csv.DictReader(handle))
    with open(tmp_path / "resumed.pt.csv", newline="", encoding="utf-8") as handle:
        resumed_log = list(csv.DictReader(handle))
    assert continuous_log == resumed_log


def _result(run_id: str, model_type: str, seed: int, *, fid: float, top1: float) -> dict:
    return {
        "status": "completed",
        "job": {
            "experiment": "A",
            "target_synset": "dog",
            "model_type": model_type,
            "aux_set": "close" if model_type == "conditional_close" else "none",
            "n0": 50,
            "m_per_aux": 50,
            "K_aux": 1,
            "data_split_seed": seed,
            "model_initialization_seed": seed,
            "training_seed": seed,
            "sampling_seed": seed,
            "evaluation_seed": seed,
            "sampler": "ddim",
            "sampling_steps": 4,
            "effective_run_spec_hash": "spec",
            "config_hash": "config",
            "training_protocol": "natural_compute_matched",
        },
        "metadata": {
            "manifest_hash": "manifest",
            "config_hash": "config",
            "git_sha": "deadbeef",
            "selected_checkpoint_path": "checkpoint.pt",
            "target_eval_indices_hash": "eval",
            "aux_synsets": json.dumps(["n02108915"] if model_type == "conditional_close" else []),
        },
        "training": {
            "optimizer_steps": 10,
            "target_examples_seen": 40,
            "auxiliary_examples_seen": 0 if "target_only" in model_type or model_type.startswith("unconditional") else 40,
            "total_examples_seen": 40 if "target_only" in model_type or model_type.startswith("unconditional") else 80,
        },
        "metrics": {
            "evaluation_mode": "strict",
            "metric_backend": "offline-test",
            "num_generated": 100,
            "num_real_eval": 100,
            "test_corruption_bank_hash": "bank",
            "fid_target": fid,
            "classifier_target_top1_acc": top1,
            "auxiliary_leakage_rate": 0.05 if model_type == "conditional_close" else float("nan"),
            "top1_prediction_histogram_json": json.dumps({"245": 2, "1": 8}),
            "test_epsilon_mse_target": fid / 20.0,
        },
    }


def test_atomic_result_failure_and_paired_aggregation(tmp_path: Path):
    malformed_path = tmp_path / "malformed.json"
    atomic_write_json(
        {"schema_version": 1, "run_id": "truncated", "status": "completed"},
        malformed_path,
    )
    with pytest.raises(ValueError, match="job and metadata"):
        load_valid_result(malformed_path)

    for seed in (0, 1):
        write_run_result(tmp_path, f"primary-{seed}", _result(f"primary-{seed}", "conditional_target_only_n0", seed, fid=10 + seed, top1=0.6))
        write_run_result(tmp_path, f"legacy-{seed}", _result(f"legacy-{seed}", "unconditional_n0", seed, fid=12 + seed, top1=0.5))
        write_run_result(tmp_path, f"model-{seed}", _result(f"model-{seed}", "conditional_close", seed, fid=8 + seed, top1=0.7))

    # Same run_id in a nested result directory: newest wins, run_id is deduped.
    duplicate = _result("model-0", "conditional_close", 0, fid=8.0, top1=0.7)
    write_run_result(tmp_path / "A_equal_target", "model-0", duplicate)
    failure_path = write_failure_result(tmp_path, "failed-run", RuntimeError("boom"), job={"experiment": "A"}, git_sha="abc")
    assert failure_path.exists()
    assert not list((tmp_path / "run_results").glob("*.tmp"))
    assert load_valid_result(tmp_path / "run_results" / "model-0.json", expected_run_id="model-0")["status"] == "completed"

    outputs = aggregate_results(tmp_path)
    assert len(outputs["all_metrics"]) == 6
    pairs = outputs["paired_transfer_gaps"]
    primary_fid = pairs[(pairs["model_type"] == "conditional_close") & (pairs["baseline_kind"] == "primary") & (pairs["metric"] == "fid_target")]
    legacy_fid = pairs[(pairs["model_type"] == "conditional_close") & (pairs["baseline_kind"] == "legacy") & (pairs["metric"] == "fid_target")]
    primary_top1 = pairs[(pairs["baseline_kind"] == "primary") & (pairs["metric"] == "classifier_target_top1_acc")]
    primary_test_mse = pairs[(pairs["baseline_kind"] == "primary") & (pairs["metric"] == "test_epsilon_mse_target")]
    primary_leakage = pairs[(pairs["baseline_kind"] == "primary") & (pairs["metric"] == "auxiliary_leakage_rate")]
    assert set(primary_fid["improvement_positive"]) == {2.0}
    assert set(legacy_fid["improvement_positive"]) == {4.0}
    assert set(primary_top1["improvement_positive"].round(8)) == {0.1}
    assert set(primary_test_mse["improvement_positive"].round(8)) == {0.1}
    assert set(primary_leakage["improvement_positive"].round(8)) == {0.15}
    assert improvement_positive(8, 10, "fid_target") == 2
    assert improvement_positive(0.7, 0.6, "classifier_target_top1_acc") == pytest.approx(0.1)
    assert (tmp_path / "summary_metrics.csv").exists()
    assert "failed-run" in set(outputs["failed_jobs"]["run_id"])
    completeness = outputs["job_completeness"].set_index("run_id")
    assert completeness.loc["model-0", "duplicate_result_count"] == 1


def test_deduplicated_baselines_broadcast_across_auxiliary_factorizations():
    shared = {
        "target_synset": "target",
        "data_split_seed": 1,
        "model_initialization_seed": 2,
        "training_seed": 3,
        "training_protocol": "natural_compute_matched",
        "sampling_seed": 4,
        "evaluation_seed": 5,
        "sampler": "ddim",
        "sampling_steps": 10,
        "effective_run_spec_hash": "spec",
        "config_hash": "config",
        "status": "completed",
    }
    rows = [
        {
            **shared,
            "run_id": "a-baseline",
            "experiment": "A",
            "model_type": "conditional_target_only_n0",
            "n0": 4,
            "m_per_aux": 0,
            "K_aux": 0,
            "total_auxiliary_budget": 0,
            "baseline_target_count": 4,
            "fid_target": 10.0,
        },
        {
            **shared,
            "run_id": "a-candidate",
            "experiment": "A",
            "model_type": "conditional_close",
            "aux_set": "close",
            "n0": 4,
            "m_per_aux": 2,
            "K_aux": 3,
            "total_auxiliary_budget": 6,
            "baseline_target_count": 4,
            "fid_target": 8.0,
        },
        {
            **shared,
            "run_id": "b-baseline",
            "experiment": "B",
            "model_type": "conditional_target_only_equal_total",
            "n0": 4,
            "m_per_aux": 6,
            "K_aux": 1,
            "total_auxiliary_budget": 6,
            "baseline_target_count": 10,
            "fid_target": 12.0,
        },
        {
            **shared,
            "run_id": "b-candidate",
            "experiment": "B",
            "model_type": "conditional_close",
            "aux_set": "close",
            "n0": 4,
            "m_per_aux": 3,
            "K_aux": 2,
            "total_auxiliary_budget": 6,
            "baseline_target_count": 10,
            "fid_target": 9.0,
        },
    ]
    pairs = compute_paired_gaps(pd.DataFrame(rows))
    primary_fid = pairs[
        (pairs["baseline_kind"] == "primary")
        & (pairs["metric"] == "fid_target")
        & (pairs["pair_status"] == "completed")
    ].set_index("run_id")
    assert primary_fid.loc["a-candidate", "baseline_run_id"] == "a-baseline"
    assert primary_fid.loc["a-candidate", "improvement_positive"] == 2.0
    assert primary_fid.loc["b-candidate", "baseline_run_id"] == "b-baseline"
    assert primary_fid.loc["b-candidate", "improvement_positive"] == 3.0


def test_t_confidence_interval_and_plot_raw_points_are_not_connected():
    stats = t95_confidence_interval([1.0, 3.0])
    assert stats["mean"] == 2.0
    assert stats["standard_deviation"] == pytest.approx(2**0.5)
    assert stats["standard_error"] == pytest.approx(1.0)
    assert stats["ci95_lower"] == pytest.approx(2.0 - 12.706)

    pairs = pd.DataFrame(
        [
            {
                "experiment": "A",
                "model_type": "conditional_close",
                "aux_set": "close",
                "training_protocol": "natural_compute_matched",
                "baseline_kind": "primary",
                "metric": "fid_target",
                "pair_status": "completed",
                "n0": n0,
                "improvement_positive": value,
            }
            for n0, value in [(50, 1.0), (50, 2.0), (100, 2.5), (100, 3.0)]
        ]
    )
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    _plot_paired_axis(axis, pairs, x_column="n0", metric="fid_target", show_raw=True)
    assert axis.collections  # raw paired values are scatter collections
    assert not any(" raw" in line.get_label() for line in axis.lines)
    assert any(np.allclose(line.get_ydata(), [0, 0]) for line in axis.lines if len(line.get_ydata()) == 2)
    plt.close(figure)


def test_similarity_analysis_reports_spearman_and_cluster_bootstrap():
    pairs = pd.DataFrame(
        [
            {
                "experiment": "C",
                "target_synset": f"target-{index % 2}",
                "data_split_seed": index % 2,
                "training_seed": index,
                "model_type": "similarity_balanced_mix",
                "training_protocol": "natural_compute_matched",
                "baseline_kind": "primary",
                "metric": "test_epsilon_mse_target",
                "pair_status": "completed",
                "average_auxiliary_similarity": float(index),
                "improvement_positive": float(index),
            }
            for index in range(1, 7)
        ]
    )
    result = similarity_correlations(pairs, bootstrap_samples=30)
    assert len(result) == 1
    assert result.iloc[0]["spearman_correlation"] == pytest.approx(1.0)
    assert result.iloc[0]["n"] == 6
