import pandas as pd
from analyze_results import pair_metrics, summarize

def _row(model, seed, rotation, risk):
    return {"model_type": model, "seed": seed, "capacity": "limited", "rotation_deg": rotation,
            "score_risk": risk, "low_noise_score_risk": risk, "mid_noise_score_risk": risk,
            "high_noise_score_risk": risk, "validation_epsilon_mse": risk, "gaussian_w2_squared": risk,
            "grad_cos_target_aux1_init": 0, "grad_cos_target_aux2_init": 0,
            "grad_cos_target_aux_mean_init": 0, "covariance_distance": 0, "noised_score_map_distance": 0}

def _paired_fixture():
    rows = []
    for seed in range(3):
        rows += [_row("target_only", seed, float("nan"), 2.0), _row("joint_conditional", seed, 0, 1.0), _row("joint_conditional", seed, 75, 1.0)]
    return pair_metrics(pd.DataFrame(rows))

def test_common_baseline_merge():
    paired = _paired_fixture()
    assert len(paired) == 6 and (paired.gap_score == -1).all()

def test_positive_sign_convention():
    paired = _paired_fixture()
    summary = summarize(paired)
    assert (summary[summary.metric == "gap_score"].transfer_status == "positive").all()
