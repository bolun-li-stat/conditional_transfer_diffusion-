from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

import train
from analyze_three_models import (_gap_plot, pair_three_models,
                                  prepare_gap_plot_series,
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


def test_low_target_rows_are_shared():
    spec = make_gaussian_mixture_spec(K=3, d=4, Delta=2, seed=0,
                                      min_pairwise_mean_distance=1)
    low = build_low_target_data_split(spec, 4, 4, 8, 0)
    assert len(low["uncond_train_x"]) == 4
    conditional_target = low["cond_train_x"][low["cond_train_y"] == spec.target_index]
    assert {tuple(row) for row in conditional_target} == {
        tuple(row) for row in low["uncond_train_x"]}


def test_same_total_budget_control_receives_exact_k_n_target_rows(
        tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path, experiment="same_total_budget")
    cfg.n = 4
    cfg.n_target_train = None
    cfg.n_aux_train = None
    captured = {}
    observed_labels = []
    original_trainer = train.train_target_only_conditional

    def recording_trainer(model, diffusion, train_x, val_x, config, target_index):
        captured["train_x"] = train_x.copy()
        original_loss = diffusion.epsilon_loss

        def recording_loss(network, x, labels):
            observed_labels.append(labels.detach().clone())
            return original_loss(network, x, labels)

        diffusion.epsilon_loss = recording_loss
        return original_trainer(
            model, diffusion, train_x, val_x, config, target_index)

    monkeypatch.setattr(train, "train_target_only_conditional", recording_trainer)
    monkeypatch.setattr(train, "ConditionalBatchSampler",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("balanced sampler used")))
    train.run_single_setting(cfg, model_types=("target_only_conditional",))

    spec = make_gaussian_mixture_spec(
        K=cfg.data.K, d=cfg.data.d, Delta=cfg.data.Delta, seed=cfg.seed,
        target_class=cfg.data.target_class,
        min_pairwise_mean_distance=cfg.data.min_pairwise_mean_distance,
        covariance_scenario=cfg.data.covariance_scenario, rho=cfg.data.rho,
        mismatch_level=cfg.data.mismatch_level, jitter=cfg.data.jitter,
        class_varying_target_rho=cfg.data.class_varying_target_rho)
    expected = build_same_total_budget_split(
        spec, cfg.n, cfg.evaluation.n_test_target, cfg.seed)
    assert len(captured["train_x"]) == cfg.data.K * cfg.n == 12
    assert np.array_equal(captured["train_x"], expected["uncond_train_x"])
    assert not np.array_equal(captured["train_x"], expected["cond_train_x"])
    assert observed_labels and all(
        torch.equal(labels, torch.full_like(labels, spec.target_index))
        for labels in observed_labels)


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


def test_mismatch_summary_combines_realized_auxiliary_rhos_across_seeds():
    rows = []
    auxiliary_by_seed = {0: "-0.08;0.22", 1: "-0.03;0.29"}
    values_by_seed = {
        0: {"unconditional": 4.0, "target_only_conditional": 3.0,
            "conditional": 2.0},
        1: {"unconditional": 6.0, "target_only_conditional": 3.0,
            "conditional": 2.0},
    }
    for seed, values in values_by_seed.items():
        for model, value in values.items():
            row = {key: 0 for key in train.RESULT_COLUMNS}
            row.update({"experiment_type": "low_target_data",
                        "covariance_scenario": "mismatch", "rho": np.nan,
                        "mismatch_level": "mild", "target_rho": 0.1,
                        "auxiliary_rhos": auxiliary_by_seed[seed], "K": 3,
                        "d": 4, "Delta": 2.0, "n": np.nan,
                        "n_target_train": 4, "n_aux_train": 4, "seed": seed,
                        "model_type": model, "sampling_mode": "balanced",
                        "training_steps": 1})
            for metric in ("score_risk", "validation_epsilon_mse",
                           "gaussian_w2_squared", "mmd_rbf", "mean_error",
                           "covariance_error"):
                row[metric] = value
            rows.append(row)
    paired = pair_three_models(pd.DataFrame(rows))
    assert set(paired.auxiliary_rhos) == set(auxiliary_by_seed.values())
    summary = summarize_three_models(paired, expected_seeds=[0, 1])
    primary = summary[
        (summary.comparison == "joint_conditional_minus_unconditional")
        & (summary.metric == "score_risk")]
    assert len(primary) == 1
    result = primary.iloc[0]
    assert result.observed_n == 2 and result.completeness == "complete"
    assert result.paired_mean == -3.0
    expected_se = 1.0
    critical = student_t.ppf(.975, 1)
    assert np.isclose(result.standard_error, expected_se)
    assert np.isclose(result.ci95_low, -3.0 - critical * expected_se)
    assert np.isclose(result.ci95_high, -3.0 + critical * expected_se)


def test_gap_plot_series_split_experiments_settings_and_use_student_t_ci(tmp_path):
    rows = []
    for experiment, x_name, x_value in [
        ("low_target_data", "n_target_train", 200),
        ("same_total_budget", "n", 200),
    ]:
        settings = [
            ("shared", 0.2, np.nan, "rho=0.2"),
            ("shared", 0.0, np.nan, "rho=0.0"),
            ("mismatch", np.nan, "mild", "mismatch=mild"),
            ("mismatch", np.nan, "strong", "mismatch=strong"),
        ]
        for scenario, rho, mismatch, _ in settings:
            for comparison in ["joint_conditional_minus_unconditional",
                               "target_only_conditional_minus_unconditional"]:
                row = {"experiment_type": experiment,
                       "covariance_scenario": scenario, "rho": rho,
                       "mismatch_level": mismatch, "comparison": comparison,
                       "metric": "score_risk", "paired_mean": 10.0,
                       "standard_error": 999.0, "ci95_low": 7.0,
                       "ci95_high": 12.0, "n": np.nan,
                       "n_target_train": np.nan}
                row[x_name] = x_value
                rows.append(row)
    summary = pd.DataFrame(rows)
    low = prepare_gap_plot_series(summary, "low_target_data", "score_risk")
    budget = prepare_gap_plot_series(summary, "same_total_budget", "score_risk")
    assert len(low) == len(budget) == 8
    assert {curve["x_name"] for curve in low} == {"n_target_train"}
    assert {curve["x_name"] for curve in budget} == {"n"}
    assert all(np.array_equal(curve["lower_error"], [3.0]) for curve in low)
    assert all(np.array_equal(curve["upper_error"], [2.0]) for curve in low)
    assert len({(curve["comparison"], curve["covariance_scenario"],
                 curve["covariance_setting"]) for curve in low}) == 8
    for experiment in ("low_target_data", "same_total_budget"):
        path = tmp_path / f"{experiment}.png"
        _gap_plot(summary, experiment, "score_risk", path)
        assert path.exists()
