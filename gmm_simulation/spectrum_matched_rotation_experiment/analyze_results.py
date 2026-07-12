"""Strict paired seed-level analysis for spectrum-matched transfer."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from utils import consolidate_seed_metrics

METRICS = {
    "score_risk": "gap_score",
    "low_noise_score_risk": "gap_low_noise",
    "mid_noise_score_risk": "gap_mid_noise",
    "high_noise_score_risk": "gap_high_noise",
    "validation_epsilon_mse": "gap_val_epsilon",
    "gaussian_w2_squared": "gap_w2",
}
DIAGNOSTICS = [
    "grad_cos_target_aux1_init", "grad_cos_target_aux2_init",
    "grad_cos_target_aux_mean_init", "covariance_distance",
    "noised_score_map_distance",
]


def pair_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"pair_id", "setting_id", "model_type", "rotation_deg", "seed", "capacity"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Metrics are missing pairing columns: {sorted(missing)}")
    baseline = metrics[metrics.model_type == "target_only"]
    counts = baseline.groupby("pair_id", dropna=False).size()
    duplicate_ids = counts[counts != 1]
    if not duplicate_ids.empty:
        raise ValueError(f"Each pair_id requires exactly one target-only baseline: {duplicate_ids.to_dict()}")
    joint = metrics[metrics.model_type == "joint_conditional"]
    duplicated_joint = joint.duplicated(["pair_id", "rotation_deg"], keep=False)
    if duplicated_joint.any():
        bad = joint.loc[duplicated_joint, ["pair_id", "rotation_deg"]].to_dict("records")
        raise ValueError(f"Duplicate joint pair_id/rotation rows: {bad}")
    paired = joint.merge(baseline, on="pair_id", suffixes=("_joint", "_target"),
                         how="left", validate="many_to_one", indicator=True)
    unmatched = paired[paired["_merge"] != "both"]
    if not unmatched.empty:
        raise ValueError(f"Joint rows have no exact target-only pair_id: {unmatched['pair_id'].tolist()}")
    for column in ("seed", "capacity"):
        if not (paired[f"{column}_joint"] == paired[f"{column}_target"]).all():
            raise ValueError(f"pair_id collision detected for {column}")
    out = paired[["pair_id", "seed_joint", "capacity_joint", "rotation_deg_joint"]].rename(
        columns={"seed_joint": "seed", "capacity_joint": "capacity",
                 "rotation_deg_joint": "rotation_deg"})
    for metric, gap in METRICS.items():
        out[gap] = paired[f"{metric}_joint"] - paired[f"{metric}_target"]
    for column in DIAGNOSTICS:
        out[column] = paired[f"{column}_joint"]
    return out


def summarize(paired: pd.DataFrame, expected_seeds: Iterable[int] = range(20)) -> pd.DataFrame:
    expected = {int(seed) for seed in expected_seeds}
    if not expected:
        raise ValueError("expected_seeds cannot be empty")
    rows: list[dict[str, object]] = []
    for (capacity, rotation), group in paired.groupby(["capacity", "rotation_deg"], sort=True):
        observed = {int(seed) for seed in group.seed.unique()}
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        complete = observed == expected
        for gap in METRICS.values():
            values = group[gap].dropna().to_numpy(dtype=float)
            n = len(values)
            mean = float(values.mean()) if n else np.nan
            se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else np.nan
            lo, hi = ((mean - critical * se, mean + critical * se)
                      if n > 1 else (np.nan, np.nan))
            if not complete or n != len(expected):
                status = "incomplete"
            elif hi < 0:
                status = "positive"
            elif lo > 0:
                status = "negative"
            else:
                status = "inconclusive"
            rows.append({
                "capacity": capacity, "rotation_deg": rotation, "metric": gap,
                "expected_n": len(expected), "observed_n": len(observed),
                "n_available": n, "missing_seeds": ";".join(map(str, missing)),
                "extra_seeds": ";".join(map(str, extra)), "mean_gap": mean,
                "standard_error": se, "ci95_low": lo, "ci95_high": hi,
                "n_gap_lt_zero": int((values < 0).sum()),
                "n_gap_gt_zero": int((values > 0).sum()),
                "transfer_status": status,
            })
    return pd.DataFrame(rows)


def _line(paired: pd.DataFrame, columns: list[str], ylabel: str, path: Path) -> None:
    plotted = False
    for capacity, group in paired.groupby("capacity"):
        for column in columns:
            available = group.dropna(subset=[column])
            if available.empty:
                continue
            stats = available.groupby("rotation_deg")[column].agg(["mean", "sem"])
            plt.errorbar(stats.index, stats["mean"], yerr=1.96 * stats["sem"].fillna(0),
                         marker="o", label=f"{capacity}: {column}")
            plotted = True
    plt.axhline(0, color="black", linewidth=.8)
    plt.xlabel("Rotation (degrees)")
    plt.ylabel(ylabel)
    if plotted:
        plt.legend(fontsize=7)
    else:
        plt.text(0.5, 0.5, "No available values", ha="center", va="center",
                 transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def analyze(results_dir: Path, expected_seeds: Iterable[int] = range(20)) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = consolidate_seed_metrics(results_dir)
    paired = pair_metrics(metrics)
    summary = summarize(paired, expected_seeds)
    paired.to_csv(results_dir / "paired_gaps.csv", index=False)
    summary.to_csv(results_dir / "summary_by_angle_capacity.csv", index=False)
    tables, figures = results_dir / "tables", results_dir / "figures"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    summary[summary.metric == "gap_score"].to_latex(
        tables / "rotation_transfer_summary.tex", index=False, float_format="%.4g")
    for capacity in ("standard", "limited"):
        subset = paired[paired.capacity == capacity]
        if not subset.empty:
            _line(subset, ["gap_score"], "Score-risk gap (joint - target only)",
                  figures / f"score_gap_by_rotation_{capacity}.png")
    _line(paired, ["gap_low_noise", "gap_mid_noise", "gap_high_noise"],
          "Score-risk gap", figures / "noise_bin_gaps.png")
    _line(paired, ["gap_w2"], "Gaussian W2 squared gap", figures / "w2_gap_by_rotation.png")
    _line(paired.assign(negative=(paired.gap_score > 0).astype(float)), ["negative"],
          "Fraction of negative-transfer seeds", figures / "negative_fraction.png")
    _line(paired, ["grad_cos_target_aux_mean_init"], "Initial shared-gradient cosine",
          figures / "gradient_cosine.png")
    _line(paired, ["covariance_distance", "noised_score_map_distance"],
          "Mismatch distance", figures / "mismatch_distances.png")
    return paired, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--expected-seeds", nargs="*", type=int, default=list(range(20)))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    analyze(Path(arguments.results_dir), arguments.expected_seeds)
