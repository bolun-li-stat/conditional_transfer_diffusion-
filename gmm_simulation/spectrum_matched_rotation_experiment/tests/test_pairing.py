import numpy as np
import pandas as pd
import pytest

from analyze_results import pair_metrics
from config import ExperimentConfig, SpectrumConfig
from data import build_paired_split
from utils import pair_id, pair_payload, setting_id


def test_reproducible_split():
    args = (8, 2, 45, 16, 16, 12, 12)
    first, second = build_paired_split(*args), build_paired_split(*args)
    assert all(np.array_equal(first[key], second[key]) for key in first)


def test_target_samples_shared_by_models():
    split = build_paired_split(8, 2, 45, 16, 16, 12, 12)
    assert np.array_equal(split["target_train"], split["joint_x"][:16])


def test_target_data_invariant_across_rotation():
    zero = build_paired_split(8, 2, 0, 16, 16, 12, 12)
    rotated = build_paired_split(8, 2, 75, 16, 16, 12, 12)
    for key in ("target_train", "target_val", "target_test"):
        assert np.array_equal(zero[key], rotated[key])


def test_target_only_setting_id_ignores_rotation():
    cfg = ExperimentConfig()
    pid = pair_id(cfg, True)
    assert setting_id(pid, "target_only", 0) == setting_id(pid, "target_only", 75)


def test_joint_setting_id_contains_rotation():
    cfg = ExperimentConfig()
    pid = pair_id(cfg, True)
    assert setting_id(pid, "joint_conditional", 0) != setting_id(pid, "joint_conditional", 75)


def _row(cfg: ExperimentConfig, model: str, rotation: int | None) -> dict[str, object]:
    pid = pair_id(cfg, True)
    row = {**pair_payload(cfg, True), "pair_id": pid,
           "setting_id": setting_id(pid, model, rotation), "model_type": model,
           "rotation_deg": np.nan if model == "target_only" else rotation}
    for metric in ("score_risk", "low_noise_score_risk", "mid_noise_score_risk",
                   "high_noise_score_risk", "validation_epsilon_mse", "gaussian_w2_squared"):
        row[metric] = 1.0
    for diagnostic in ("grad_cos_target_aux1_init", "grad_cos_target_aux2_init",
                       "grad_cos_target_aux_mean_init", "covariance_distance",
                       "noised_score_map_distance"):
        row[diagnostic] = np.nan
    return row


def test_training_steps_cannot_cross_pair():
    baseline_cfg = ExperimentConfig(training_steps=2_000)
    joint_cfg = ExperimentConfig(training_steps=20_000)
    with pytest.raises(ValueError, match="no exact target-only"):
        pair_metrics(pd.DataFrame([_row(baseline_cfg, "target_only", None),
                                   _row(joint_cfg, "joint_conditional", 0)]))


def test_spectrum_cannot_cross_pair():
    baseline_cfg = ExperimentConfig(spectrum=SpectrumConfig(1.8, 0.2))
    joint_cfg = ExperimentConfig(spectrum=SpectrumConfig(1.7, 0.3))
    with pytest.raises(ValueError, match="no exact target-only"):
        pair_metrics(pd.DataFrame([_row(baseline_cfg, "target_only", None),
                                   _row(joint_cfg, "joint_conditional", 0)]))


def test_duplicate_baseline_fails():
    cfg = ExperimentConfig()
    baseline = _row(cfg, "target_only", None)
    with pytest.raises(ValueError, match="exactly one"):
        pair_metrics(pd.DataFrame([baseline, baseline, _row(cfg, "joint_conditional", 0)]))
