from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train
from analyze_three_models import (COMPARISONS, pair_three_models,
                                  summarize_three_models)
from conditional_model import ConditionalDenoiser
from config import ExperimentConfig, default_smoke_config
from data import (build_low_target_data_split, build_same_total_budget_split,
                  make_gaussian_mixture_spec)
from diffusion import DDPM


def tiny_config(results: Path, experiment: str = "smoke") -> ExperimentConfig:
    cfg = ExperimentConfig(experiment_type=experiment, results_dir=results,
                           seed=0, n_target_train=4, n_aux_train=4)
    cfg.data.d = 4; cfg.data.Delta = 2.0
    cfg.data.min_pairwise_mean_distance = 1.0
    cfg.diffusion.T = 3
    cfg.model.time_embedding_dim = 8; cfg.model.class_embedding_dim = 4
    cfg.model.hidden_width = 12; cfg.model.hidden_layers = 1
    cfg.training.device = "cpu"; cfg.training.batch_size = 4
    cfg.training.training_steps = 1; cfg.training.validation_interval = 1
    cfg.training.checkpoint_interval = 1
    cfg.evaluation.n_test_target = 8; cfg.evaluation.n_generated = 4
    cfg.evaluation.score_risk_mc_samples = 4; cfg.evaluation.mmd_max_samples = 8
    return cfg


def test_legacy_identity_snapshots_and_default_selection():
    cfg = default_smoke_config()
    assert train.common_setting_id(cfg) == "79296a054e44"
    assert train.setting_id(cfg, "unconditional") == "cbcb6cf0fdca"
    assert train.setting_id(cfg, "conditional") == "7ffcfa099b88"
    assert train.normalize_model_types(None) == ("unconditional", "conditional")
    assert train.normalize_model_types(["all"]) == (
        "unconditional", "conditional", "target_only_conditional")


def test_fixed_label_trainer_uses_full_embedding_and_no_balanced_sampler(tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path)
    model = ConditionalDenoiser(4, 3, 8, 4, 12, 1)
    assert model.class_embedding.num_embeddings == 3
    diffusion = DDPM(cfg.diffusion, torch.device("cpu"))
    seen = []
    original = diffusion.epsilon_loss

    def recording_loss(network, x, labels):
        seen.append(labels.detach().clone())
        return original(network, x, labels)

    diffusion.epsilon_loss = recording_loss
    monkeypatch.setattr(train, "ConditionalBatchSampler",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("balanced sampler used")))
    x = np.zeros((4, 4), dtype=np.float32)
    train.train_target_only_conditional(model, diffusion, x, x, cfg, target_index=0)
    assert seen and all(torch.equal(labels, torch.zeros_like(labels)) for labels in seen)


def test_target_rows_match_both_budget_designs():
    spec = make_gaussian_mixture_spec(K=3, d=4, Delta=2, seed=0,
                                      min_pairwise_mean_distance=1)
    low = build_low_target_data_split(spec, 4, 4, 8, 0)
    assert len(low["uncond_train_x"]) == 4
    conditional_target = low["cond_train_x"][low["cond_train_y"] == spec.target_index]
    assert {tuple(row) for row in conditional_target} == {
        tuple(row) for row in low["uncond_train_x"]}
    budget = build_same_total_budget_split(spec, 4, 8, 0)
    assert len(budget["uncond_train_x"]) == 12
    # C_T consumes this exact array, so it has K*n rows just like U_T.
    assert np.array_equal(budget["uncond_train_x"], budget["uncond_train_x"])


def test_new_model_runs_alone_and_upserts_without_legacy_rows(tmp_path):
    cfg = tiny_config(tmp_path)
    first = train.run_single_setting(
        cfg, model_types=("target_only_conditional",))
    second = train.run_single_setting(
        cfg, model_types=("target_only_conditional",))
    stored = pd.read_csv(tmp_path / "metrics.csv")
    assert list(first.model_type) == ["target_only_conditional"]
    assert second.empty and list(stored.model_type) == ["target_only_conditional"]
    assert not any(path.name in {"unconditional_last.pt", "conditional_last.pt"}
                   for path in tmp_path.rglob("*.pt"))


def test_all_orchestration_finishes_legacy_before_new(monkeypatch, tmp_path):
    import sys
    cfgs = [tiny_config(tmp_path / "a"), tiny_config(tmp_path / "b")]
    calls = []
    monkeypatch.setattr(train, "build_cli_configs", lambda args: cfgs)
    monkeypatch.setattr(train, "describe_device", lambda: None)
    monkeypatch.setattr(train, "run_single_setting",
                        lambda cfg, force, model_types: calls.append(model_types)
                        or pd.DataFrame())
    monkeypatch.setattr(sys, "argv", ["train.py", "--model-types", "all"])
    train.main()
    assert calls == [("unconditional", "conditional"),
                     ("unconditional", "conditional"),
                     ("target_only_conditional",),
                     ("target_only_conditional",)]


def test_control_is_identical_alone_or_after_legacy(tmp_path):
    after_cfg = tiny_config(tmp_path / "after")
    train.run_single_setting(after_cfg, model_types=("unconditional", "conditional"))
    after = train.run_single_setting(
        after_cfg, model_types=("target_only_conditional",)).iloc[0]
    alone_cfg = tiny_config(tmp_path / "alone")
    alone = train.run_single_setting(
        alone_cfg, model_types=("target_only_conditional",)).iloc[0]
    after_state = torch.load(after.checkpoint_path, map_location="cpu")
    alone_state = torch.load(alone.checkpoint_path, map_location="cpu")
    assert all(torch.equal(after_state["model_state_dict"][key],
                           alone_state["model_state_dict"][key])
               for key in after_state["model_state_dict"])
    for metric in ("final_train_loss", "score_risk", "validation_epsilon_mse",
                   "gaussian_w2_squared", "mmd_rbf"):
        assert after[metric] == alone[metric]


def _metric_rows(include_control=True):
    rows = []
    values = {"unconditional": 3.0, "target_only_conditional": 2.0,
              "conditional": 1.0}
    for seed in (0, 1):
        for model, value in values.items():
            if model == "target_only_conditional" and not include_control:
                continue
            row = {key: 0 for key in train.RESULT_COLUMNS}
            row.update({"experiment_type": "smoke", "covariance_scenario": "shared",
                        "seed": seed, "model_type": model,
                        "sampling_mode": "balanced", "training_steps": 1})
            for metric in ("score_risk", "validation_epsilon_mse",
                           "gaussian_w2_squared", "mmd_rbf", "mean_error",
                           "covariance_error"):
                row[metric] = value
            rows.append(row)
    return pd.DataFrame(rows)


def test_three_gaps_are_paired_and_additive_with_missing_control_supported():
    paired = pair_three_models(_metric_rows())
    score = paired[paired.metric == "score_risk"]
    means = score.groupby("comparison").gap.mean().to_dict()
    assert means["joint_conditional_minus_unconditional"] == -2
    assert means["joint_conditional_minus_target_only_conditional"] == -1
    assert means["target_only_conditional_minus_unconditional"] == -1
    summary = summarize_three_models(paired, expected_seeds=[0, 1])
    assert set(summary.completeness) == {"complete"}
    missing = pair_three_models(_metric_rows(include_control=False))
    total = missing[missing.comparison == "joint_conditional_minus_unconditional"]
    auxiliary = missing[
        missing.comparison == "joint_conditional_minus_target_only_conditional"]
    assert total.gap.notna().all() and auxiliary.gap.isna().all()
