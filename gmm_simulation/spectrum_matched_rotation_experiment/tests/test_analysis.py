from pathlib import Path

import numpy as np
import pandas as pd

from analyze_results import analyze, pair_metrics, summarize
from config import ExperimentConfig
from utils import pair_id, pair_payload, setting_id, upsert_seed_metric


def metric_row(seed: int, model: str, rotation: int | None, risk: float,
               generation_enabled: bool = True) -> dict[str, object]:
    cfg = ExperimentConfig(seed=seed, capacity="limited", rotation_deg=rotation or 0)
    pid = pair_id(cfg, generation_enabled)
    row = {**pair_payload(cfg, generation_enabled), "pair_id": pid,
           "setting_id": setting_id(pid, model, rotation), "model_type": model,
           "rotation_deg": np.nan if model == "target_only" else rotation}
    for metric in ("score_risk", "low_noise_score_risk", "mid_noise_score_risk",
                   "high_noise_score_risk", "validation_epsilon_mse"):
        row[metric] = risk
    row["gaussian_w2_squared"] = risk if generation_enabled else np.nan
    for diagnostic in ("grad_cos_target_aux1_init", "grad_cos_target_aux2_init",
                       "grad_cos_target_aux_mean_init", "covariance_distance",
                       "noised_score_map_distance"):
        row[diagnostic] = 0.0
    return row


def paired_fixture(seeds: range, joint_risk: float = 1.0) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        rows.extend([metric_row(seed, "target_only", None, 2.0),
                     metric_row(seed, "joint_conditional", 0, joint_risk)])
    return pair_metrics(pd.DataFrame(rows))


def test_three_seed_fixture_is_incomplete():
    summary = summarize(paired_fixture(range(3)))
    assert set(summary.transfer_status) == {"incomplete"}


def test_complete_twenty_seed_fixture_can_be_positive():
    summary = summarize(paired_fixture(range(20)))
    score = summary[summary.metric == "gap_score"].iloc[0]
    assert score.transfer_status == "positive"
    assert score.missing_seeds == ""


def test_missing_seed_is_reported():
    summary = summarize(paired_fixture(range(19)))
    score = summary[summary.metric == "gap_score"].iloc[0]
    assert score.transfer_status == "incomplete"
    assert score.missing_seeds == "19"


def test_negative_sign_convention():
    summary = summarize(paired_fixture(range(20), joint_risk=3.0))
    assert summary[summary.metric == "gap_score"].iloc[0].transfer_status == "negative"


def test_analysis_end_to_end_with_nan_w2(tmp_path: Path):
    for seed in range(3):
        upsert_seed_metric(metric_row(seed, "target_only", None, 2.0, False), tmp_path)
        upsert_seed_metric(metric_row(seed, "joint_conditional", 0, 1.0, False), tmp_path)
    paired, summary = analyze(tmp_path)
    assert not paired.empty
    assert set(summary.transfer_status) == {"incomplete"}
    w2 = summary[summary.metric == "gap_w2"].iloc[0]
    assert w2.n_available == 0 and np.isnan(w2.mean_gap)
    expected = [tmp_path / "metrics.csv", tmp_path / "paired_gaps.csv",
                tmp_path / "summary_by_angle_capacity.csv",
                tmp_path / "tables/rotation_transfer_summary.tex",
                tmp_path / "figures/score_gap_by_rotation_limited.png",
                tmp_path / "figures/noise_bin_gaps.png", tmp_path / "figures/w2_gap_by_rotation.png",
                tmp_path / "figures/negative_fraction.png", tmp_path / "figures/gradient_cosine.png",
                tmp_path / "figures/mismatch_distances.png"]
    assert all(path.exists() for path in expected)
