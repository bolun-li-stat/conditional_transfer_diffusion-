from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analyze_results import analyze, pair_metrics, summarize
from config import ExperimentConfig
from utils import identity_payload, setting_id, upsert_seed_metric


def metric_row(seed: int, model: str, rotation: int | None, score: float,
               w2: float = np.nan, training_steps: int = 20_000) -> dict[str, object]:
    cfg = ExperimentConfig(seed=seed, capacity="limited", rotation_deg=rotation or 0,
                           training_steps=training_steps)
    identity = identity_payload(cfg)
    row = {**identity, "setting_id": setting_id(identity["pair_id"], model, rotation),
           "model_type": model, "rotation_deg": np.nan if model == "target_only" else rotation}
    for metric in ("score_risk", "low_noise_score_risk", "mid_noise_score_risk",
                   "high_noise_score_risk", "validation_epsilon_mse"):
        row[metric] = score
    row["gaussian_w2_squared"] = w2
    for diagnostic in ("grad_cos_target_aux1_init", "grad_cos_target_aux2_init",
                       "grad_cos_target_aux_mean_init", "covariance_distance",
                       "noised_score_map_distance"):
        row[diagnostic] = 0.0
    return row


def paired_fixture(seeds, joint_score=1.0, baseline_w2=2.0, joint_w2=np.nan,
                   training_steps=20_000):
    rows = []
    for seed in seeds:
        rows.extend([metric_row(seed, "target_only", None, 2.0, baseline_w2, training_steps),
                     metric_row(seed, "joint_conditional", 0, joint_score, joint_w2, training_steps)])
    return pair_metrics(pd.DataFrame(rows))


def test_mixed_training_step_seeds_remain_two_incomplete_designs():
    rows = []
    for seed in range(10):
        rows += [metric_row(seed, "target_only", None, 2.0, 2.0, 2_000),
                 metric_row(seed, "joint_conditional", 0, 1.0, 1.0, 2_000)]
    for seed in range(10, 20):
        rows += [metric_row(seed, "target_only", None, 2.0, 2.0, 20_000),
                 metric_row(seed, "joint_conditional", 0, 1.0, 1.0, 20_000)]
    summary = summarize(pair_metrics(pd.DataFrame(rows)))
    assert summary.design_id.nunique() == 2
    assert set(summary.observed_n) == {10}
    assert set(summary.score_transfer_status.dropna().replace("", np.nan).dropna()) == {"incomplete"}


def test_score_complete_while_sample_incomplete():
    summary = summarize(paired_fixture(range(20)))
    score = summary[summary.metric == "gap_score"].iloc[0]
    sample = summary[summary.metric == "gap_w2"].iloc[0]
    assert score.score_transfer_status == "positive"
    assert sample.sample_transfer_status == "incomplete"


def test_complete_sample_transfer_signs_and_metric_disagreement():
    positive = summarize(paired_fixture(range(20), joint_w2=1.0))
    assert positive[positive.metric == "gap_w2"].iloc[0].sample_transfer_status == "positive"
    negative = summarize(paired_fixture(range(20), joint_score=1.0, joint_w2=3.0))
    assert negative[negative.metric == "gap_score"].iloc[0].score_transfer_status == "positive"
    assert negative[negative.metric == "gap_w2"].iloc[0].sample_transfer_status == "negative"
    assert "overall_status" not in negative.columns


def test_analysis_end_to_end_and_design_filter(tmp_path: Path):
    for seed in range(3):
        upsert_seed_metric(metric_row(seed, "target_only", None, 2.0, 2.0), tmp_path)
        upsert_seed_metric(metric_row(seed, "joint_conditional", 0, 1.0, 1.0), tmp_path)
    paired, summary = analyze(tmp_path)
    design = paired.design_id.iloc[0]
    assert set(summary.score_transfer_status.replace("", np.nan).dropna()) == {"incomplete"}
    analyze(tmp_path, selected_design_id=design)
    with pytest.raises(ValueError, match="Unknown design_id"):
        analyze(tmp_path, selected_design_id="missing")
    expected = ["metrics.csv", "design_manifest.csv", "paired_gaps.csv",
                "summary_by_angle_capacity.csv", "tables/rotation_transfer_summary.tex",
                "figures/score_gap_by_rotation.png", "figures/noise_bin_gaps.png",
                "figures/w2_gap_by_rotation.png", "figures/negative_score_fraction.png",
                "figures/negative_sample_fraction.png", "figures/gradient_cosine.png",
                "figures/mismatch_distances.png"]
    assert all((tmp_path / item).exists() for item in expected)
