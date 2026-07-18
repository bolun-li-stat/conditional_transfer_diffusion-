import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from analyze_results import (THREE_MODEL_PAIRED_COLUMNS,
                             THREE_MODEL_SUMMARY_COLUMNS, analyze,
                             metric_status_family, pair_three_models,
                             summarize_three_models)
from config import ExperimentConfig, smoke_config
from train import (normalize_model_types, run_unconditional_setting,
                   train_unconditional)
from unconditional_model import LabelIgnoringAdapter, UnconditionalDenoiser
from utils import (checkpoint_id, design_id, pair_id, setting_id,
                   training_design_id, upsert_seed_metric)


def test_unconditional_core_is_label_free_and_adapter_has_no_extra_parameters():
    core = UnconditionalDenoiser(4, 8, 12, 1)
    assert list(inspect.signature(core.forward).parameters) == ["x", "t"]
    assert not any("class" in name or "label" in name for name, _ in core.named_parameters())
    adapter = LabelIgnoringAdapter(core)
    x = torch.randn(2, 4); t = torch.zeros(2, dtype=torch.long)
    assert torch.equal(adapter(x, t, torch.zeros(2, dtype=torch.long)),
                       adapter(x, t, torch.ones(2, dtype=torch.long)))
    assert sum(p.numel() for p in adapter.parameters()) == sum(
        p.numel() for p in core.parameters())


def test_legacy_identity_snapshots_and_unconditional_angle_independence():
    cfg = ExperimentConfig()
    tid, did = training_design_id(cfg), design_id(cfg)
    pid = pair_id(did, cfg.seed)
    assert tid == "74fa8c0f002df99d"
    assert did == "c989a2c1328858a9"
    assert pid == "ea1b310b44b4e5c9"
    assert checkpoint_id(tid, 0, "target_only") == "8beb0f58275e33fa"
    assert setting_id(pid, "target_only") == "afa850ba5141e4b3"
    assert checkpoint_id(tid, 0, "joint_conditional", 45) == "9f9db481c5e7abf2"
    assert setting_id(pid, "joint_conditional", 45) == "7d4c22bbc6474c64"
    assert checkpoint_id(tid, 0, "unconditional", 0) == checkpoint_id(
        tid, 0, "unconditional", 75)
    assert setting_id(pid, "unconditional", 0) == setting_id(
        pid, "unconditional", 75)
    assert normalize_model_types(None) == ("target_only", "joint_conditional")


def test_unconditional_score_then_generation_only_is_independent(tmp_path: Path):
    cfg = smoke_config(tmp_path); cfg.device = "cpu"
    score = run_unconditional_setting(cfg, skip_generation=True)
    assert score["model_type"] == "unconditional"
    assert np.isnan(score["rotation_deg"])
    assert not bool(score["generation_evaluation_complete"])
    generated = run_unconditional_setting(cfg, resume=True, generation_only=True)
    assert generated["setting_id"] == score["setting_id"]
    assert generated["checkpoint_path"] == score["checkpoint_path"]
    assert np.isfinite(generated["gaussian_w2_squared"])
    stored = pd.read_csv(tmp_path / "metrics/seed_000.csv")
    assert list(stored.model_type) == ["unconditional"]


def test_unconditional_uses_only_target_array(tmp_path: Path, monkeypatch):
    cfg = smoke_config(tmp_path); cfg.training_steps = 1
    observed = {}

    def recording_train(config, target_train, diffusion, resume):
        observed["rows"] = target_train.copy()
        return train_unconditional(config, target_train, diffusion, resume)

    import train
    monkeypatch.setattr(train, "train_unconditional", recording_train)
    row = train.run_unconditional_setting(cfg, skip_generation=True)
    from data import build_paired_split
    split = build_paired_split(cfg.d, cfg.seed, cfg.rotation_deg,
                               cfg.n_target_train, cfg.n_aux_train,
                               cfg.n_validation, cfg.n_test)
    assert np.array_equal(observed["rows"], split["target_train"])
    assert row["model_type"] == "unconditional"


def test_full_cli_runs_unconditional_once_and_no_legacy(monkeypatch, tmp_path):
    import sys
    import train
    calls = []
    monkeypatch.setattr(train, "run_setting",
                        lambda *args, **kwargs: calls.append(("legacy", args[1])))
    monkeypatch.setattr(train, "run_unconditional_setting",
                        lambda cfg, *args, **kwargs: calls.append(
                            ("unconditional", cfg.rotation_deg)))
    monkeypatch.setattr(sys, "argv", ["train.py", "--experiment", "full",
                        "--capacity", "limited", "--seeds", "0",
                        "--model-types", "unconditional", "--results-dir",
                        str(tmp_path)])
    train.main()
    assert calls == [("unconditional", 0)]


def test_unconditional_is_identical_alone_or_after_legacy(tmp_path: Path):
    import train
    after_cfg = smoke_config(tmp_path / "after"); after_cfg.device = "cpu"
    train.run_setting(after_cfg, "target_only", skip_generation=True)
    train.run_setting(after_cfg, "joint_conditional", skip_generation=True)
    after = train.run_unconditional_setting(after_cfg, skip_generation=True)
    alone_cfg = smoke_config(tmp_path / "alone"); alone_cfg.device = "cpu"
    alone = train.run_unconditional_setting(alone_cfg, skip_generation=True)
    after_state = torch.load(after["checkpoint_path"], map_location="cpu")["model"]
    alone_state = torch.load(alone["checkpoint_path"], map_location="cpu")["model"]
    assert all(torch.equal(after_state[key], alone_state[key]) for key in after_state)
    for metric in ("final_train_loss", "score_risk", "validation_epsilon_mse"):
        assert after[metric] == alone[metric]


def _row(seed, model, rotation, value):
    cfg = ExperimentConfig(seed=seed, capacity="limited", rotation_deg=rotation or 0)
    from utils import identity_payload
    identity = identity_payload(cfg)
    row = {**identity, "setting_id": setting_id(identity["pair_id"], model, rotation),
           "model_type": model,
           "rotation_deg": np.nan if model != "joint_conditional" else rotation}
    for metric in ("score_risk", "low_noise_score_risk", "mid_noise_score_risk",
                   "high_noise_score_risk", "validation_epsilon_mse",
                   "gaussian_w2_squared", "mean_error", "covariance_error"):
        row[metric] = value
    return row


def test_rotation_three_model_gaps_and_missing_unconditional():
    rows = []
    for seed in (0, 1):
        rows += [_row(seed, "unconditional", None, 3),
                 _row(seed, "target_only", None, 2),
                 _row(seed, "joint_conditional", 45, 1)]
    paired = pair_three_models(pd.DataFrame(rows))
    score = paired[paired.metric == "score_risk"]
    assert score.groupby("comparison").gap.mean().to_dict() == {
        "joint_conditional_minus_target_only_conditional": -1.0,
        "joint_conditional_minus_unconditional": -2.0,
        "target_only_conditional_minus_unconditional": -1.0,
    }
    summary = summarize_three_models(paired, expected_seeds=[0, 1])
    assert set(summary.completeness) == {"complete"}
    missing = pair_three_models(pd.DataFrame(
        [row for row in rows if row["model_type"] != "unconditional"]))
    legacy = missing[
        missing.comparison == "joint_conditional_minus_target_only_conditional"]
    primary = missing[
        missing.comparison == "joint_conditional_minus_unconditional"]
    assert legacy.gap.notna().all() and primary.gap.isna().all()


def test_rotation_metric_status_families_are_disjoint():
    rows = []
    for seed in (0, 1):
        rows += [_row(seed, "unconditional", None, 3),
                 _row(seed, "target_only", None, 2),
                 _row(seed, "joint_conditional", 45, 1)]
    summary = summarize_three_models(
        pair_three_models(pd.DataFrame(rows)), expected_seeds=[0, 1])
    expected = {
        "score_risk": "score", "low_noise_score_risk": "score",
        "gaussian_w2_squared": "sample", "validation_epsilon_mse": "diagnostic",
        "mean_error": "diagnostic", "covariance_error": "diagnostic",
    }
    for metric, family in expected.items():
        assert metric_status_family(metric) == family
        row = summary[
            (summary.metric == metric)
            & (summary.comparison == "joint_conditional_minus_unconditional")
        ].iloc[0]
        statuses = {"score": row.score_transfer_status,
                    "sample": row.sample_transfer_status,
                    "diagnostic": row.diagnostic_status}
        assert statuses[family]
        assert all(not status for name, status in statuses.items() if name != family)


@pytest.mark.parametrize("model", ["unconditional", "target_only",
                                    "joint_conditional"])
def test_analysis_single_model_directory_is_safe(tmp_path: Path, model: str):
    rotation = 45 if model == "joint_conditional" else None
    row = _row(0, model, rotation, 1)
    for name in ("grad_cos_target_aux1_init", "grad_cos_target_aux2_init",
                 "grad_cos_target_aux_mean_init", "covariance_distance",
                 "noised_score_map_distance"):
        row[name] = 0.0
    upsert_seed_metric(row, tmp_path)
    analyze(tmp_path, expected_seeds=[0])
    paired = pd.read_csv(tmp_path / "paired_three_model_gaps.csv")
    summary = pd.read_csv(tmp_path / "summary_three_model_gaps.csv")
    assert list(paired.columns) == THREE_MODEL_PAIRED_COLUMNS
    assert list(summary.columns) == THREE_MODEL_SUMMARY_COLUMNS
    assert paired.gap.dropna().empty
    assert summary.paired_mean.dropna().empty


def test_analysis_runs_when_target_only_baseline_is_missing(tmp_path: Path):
    for row in [_row(0, "unconditional", None, 3),
                _row(0, "joint_conditional", 45, 1)]:
        for name in ("grad_cos_target_aux1_init", "grad_cos_target_aux2_init",
                     "grad_cos_target_aux_mean_init", "covariance_distance",
                     "noised_score_map_distance"):
            row[name] = 0.0
        upsert_seed_metric(row, tmp_path)
    legacy, _ = analyze(tmp_path, expected_seeds=[0])
    assert legacy.empty
    three = pd.read_csv(tmp_path / "paired_three_model_gaps.csv")
    primary = three[
        three.comparison == "joint_conditional_minus_unconditional"]
    auxiliary = three[
        three.comparison == "joint_conditional_minus_target_only_conditional"]
    assert primary.gap.notna().all() and auxiliary.gap.isna().all()
