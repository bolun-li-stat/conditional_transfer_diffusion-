"""Paired seed-level analysis for spectrum-matched transfer."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS = {"score_risk": "gap_score", "low_noise_score_risk": "gap_low_noise",
           "mid_noise_score_risk": "gap_mid_noise", "high_noise_score_risk": "gap_high_noise",
           "validation_epsilon_mse": "gap_val_epsilon", "gaussian_w2_squared": "gap_w2"}


def pair_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics.model_type == "target_only"].drop_duplicates(["seed", "capacity"])
    joint = metrics[metrics.model_type == "joint_conditional"]
    paired = joint.merge(base, on=["seed", "capacity"], suffixes=("_joint", "_target"), validate="many_to_one")
    out = paired[["seed", "capacity", "rotation_deg_joint"]].rename(columns={"rotation_deg_joint":"rotation_deg"})
    for metric, gap in METRICS.items():
        out[gap] = paired[f"{metric}_joint"] - paired[f"{metric}_target"]
    for col in ["grad_cos_target_aux1_init", "grad_cos_target_aux2_init", "grad_cos_target_aux_mean_init",
                "covariance_distance", "noised_score_map_distance"]:
        out[col] = paired[f"{col}_joint"]
    return out


def summarize(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (capacity, rotation), group in paired.groupby(["capacity", "rotation_deg"], sort=True):
        for gap in METRICS.values():
            values = group[gap].dropna().to_numpy()
            n = len(values); mean = float(values.mean()) if n else np.nan
            se = float(values.std(ddof=1)/np.sqrt(n)) if n > 1 else np.nan
            lo, hi = (mean-1.96*se, mean+1.96*se) if n > 1 else (np.nan, np.nan)
            status = "positive" if hi < 0 else "negative" if lo > 0 else "inconclusive"
            rows.append({"capacity": capacity, "rotation_deg": rotation, "metric": gap, "n": n,
                         "mean_gap": mean, "standard_error": se, "ci95_low": lo, "ci95_high": hi,
                         "n_gap_lt_zero": int((values < 0).sum()), "n_gap_gt_zero": int((values > 0).sum()),
                         "transfer_status": status})
    return pd.DataFrame(rows)


def _line(paired: pd.DataFrame, columns: list[str], ylabel: str, path: Path) -> None:
    for capacity, group in paired.groupby("capacity"):
        for column in columns:
            stats = group.groupby("rotation_deg")[column].agg(["mean", "sem"])
            plt.errorbar(stats.index, stats["mean"], yerr=1.96*stats["sem"], marker="o", label=f"{capacity}: {column}")
    plt.axhline(0, color="black", linewidth=.8); plt.xlabel("Rotation (degrees)"); plt.ylabel(ylabel); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def analyze(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = pair_metrics(pd.read_csv(results_dir / "metrics.csv")); summary = summarize(paired)
    paired.to_csv(results_dir / "paired_gaps.csv", index=False); summary.to_csv(results_dir / "summary_by_angle_capacity.csv", index=False)
    tables = results_dir / "tables"; figures = results_dir / "figures"; tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    summary[summary.metric == "gap_score"].to_latex(tables / "rotation_transfer_summary.tex", index=False, float_format="%.4g")
    for capacity in ("standard", "limited"):
        subset = paired[paired.capacity == capacity]
        if not subset.empty:
            _line(subset, ["gap_score"], "Score-risk gap (joint - target only)",
                  figures / f"score_gap_by_rotation_{capacity}.png")
    _line(paired, ["gap_low_noise","gap_mid_noise","gap_high_noise"], "Score-risk gap", figures / "noise_bin_gaps.png")
    _line(paired, ["gap_w2"], "Gaussian W2 squared gap", figures / "w2_gap_by_rotation.png")
    fraction = paired.assign(negative=paired.gap_score > 0)
    _line(fraction, ["negative"], "Fraction of negative-transfer seeds", figures / "negative_fraction.png")
    _line(paired, ["grad_cos_target_aux_mean_init"], "Initial shared-gradient cosine", figures / "gradient_cosine.png")
    _line(paired, ["covariance_distance","noised_score_map_distance"], "Mismatch distance", figures / "mismatch_distances.png")
    return paired, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--results-dir", default="results")
    analyze(Path(parser.parse_args().results_dir))
