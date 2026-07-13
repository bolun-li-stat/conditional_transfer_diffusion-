import numpy as np
import pandas as pd
import pytest

from analyze_results import pair_metrics
from config import CAPACITIES, CapacityConfig, ExperimentConfig, SpectrumConfig
from data import build_paired_split
from utils import (checkpoint_id, design_id, identity_payload, pair_id,
                   setting_id, training_design_id)


def test_reproducible_split_and_shared_target():
    args = (8, 2, 45, 16, 16, 12, 12)
    first, second = build_paired_split(*args), build_paired_split(*args)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert np.array_equal(first["target_train"], first["joint_x"][:16])


def test_target_data_invariant_across_rotation():
    zero = build_paired_split(8, 2, 0, 16, 16, 12, 12)
    rotated = build_paired_split(8, 2, 75, 16, 16, 12, 12)
    for key in ("target_train", "target_val", "target_test"):
        assert np.array_equal(zero[key], rotated[key])


def test_target_and_joint_rotation_identity_rules():
    cfg = ExperimentConfig()
    tid, did = training_design_id(cfg), design_id(cfg)
    pid = pair_id(did, cfg.seed)
    assert checkpoint_id(tid, 0, "target_only", 0) == checkpoint_id(tid, 0, "target_only", 75)
    assert checkpoint_id(tid, 0, "joint_conditional", 0) != checkpoint_id(tid, 0, "joint_conditional", 75)
    assert setting_id(pid, "target_only", 0) == setting_id(pid, "target_only", 75)
    assert setting_id(pid, "joint_conditional", 0) != setting_id(pid, "joint_conditional", 75)


def test_evaluation_options_change_design_not_checkpoint():
    base = ExperimentConfig()
    more_score = ExperimentConfig(score_risk_mc_samples=9_000)
    more_samples = ExperimentConfig(n_generated=3_000)
    for changed in (more_score, more_samples):
        assert training_design_id(base) == training_design_id(changed)
        assert checkpoint_id(training_design_id(base), 0, "target_only") == checkpoint_id(
            training_design_id(changed), 0, "target_only")
        assert design_id(base) != design_id(changed)


def test_skip_and_generation_only_share_checkpoint_identity():
    cfg = ExperimentConfig()
    tid = training_design_id(cfg)
    # Execution stage is intentionally absent from the checkpoint payload.
    skip_stage = checkpoint_id(tid, cfg.seed, "joint_conditional", 45)
    generation_stage = checkpoint_id(tid, cfg.seed, "joint_conditional", 45)
    assert skip_stage == generation_stage


def test_training_steps_change_checkpoint_and_design():
    base, changed = ExperimentConfig(), ExperimentConfig(training_steps=2_000)
    assert training_design_id(base) != training_design_id(changed)
    assert design_id(base) != design_id(changed)


def test_architecture_fields_change_ids(monkeypatch):
    cfg = ExperimentConfig(capacity="standard")
    original_training, original_design = training_design_id(cfg), design_id(cfg)
    monkeypatch.setitem(CAPACITIES, "standard", CapacityConfig(64, 32, 300, 4))
    assert training_design_id(cfg) != original_training and design_id(cfg) != original_design
    monkeypatch.setitem(CAPACITIES, "standard", CapacityConfig(64, 16, 256, 4))
    assert training_design_id(cfg) != original_training and design_id(cfg) != original_design


def _row(cfg: ExperimentConfig, model: str, rotation: int | None) -> dict[str, object]:
    identity = identity_payload(cfg)
    row = {**identity, "setting_id": setting_id(identity["pair_id"], model, rotation),
           "model_type": model, "rotation_deg": np.nan if model == "target_only" else rotation}
    for metric in ("score_risk", "low_noise_score_risk", "mid_noise_score_risk",
                   "high_noise_score_risk", "validation_epsilon_mse", "gaussian_w2_squared"):
        row[metric] = 1.0
    for diagnostic in ("grad_cos_target_aux1_init", "grad_cos_target_aux2_init",
                       "grad_cos_target_aux_mean_init", "covariance_distance",
                       "noised_score_map_distance"):
        row[diagnostic] = np.nan
    return row


def test_target_and_joint_strictly_pair_by_pair_id():
    baseline_cfg = ExperimentConfig(training_steps=2_000)
    joint_cfg = ExperimentConfig(training_steps=20_000)
    with pytest.raises(ValueError, match="no exact target-only"):
        pair_metrics(pd.DataFrame([_row(baseline_cfg, "target_only", None),
                                   _row(joint_cfg, "joint_conditional", 0)]))


def test_spectrum_cannot_cross_pair_and_duplicate_baseline_fails():
    baseline_cfg = ExperimentConfig(spectrum=SpectrumConfig(1.8, 0.2))
    joint_cfg = ExperimentConfig(spectrum=SpectrumConfig(1.7, 0.3))
    with pytest.raises(ValueError, match="no exact target-only"):
        pair_metrics(pd.DataFrame([_row(baseline_cfg, "target_only", None),
                                   _row(joint_cfg, "joint_conditional", 0)]))
    baseline = _row(baseline_cfg, "target_only", None)
    with pytest.raises(ValueError, match="exactly one"):
        pair_metrics(pd.DataFrame([baseline, baseline,
                                   _row(baseline_cfg, "joint_conditional", 0)]))
