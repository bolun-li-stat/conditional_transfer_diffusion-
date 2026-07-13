"""Design-isolated paired analysis for score and sample transfer."""
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
    "score_risk": "gap_score", "low_noise_score_risk": "gap_low_noise",
    "mid_noise_score_risk": "gap_mid_noise", "high_noise_score_risk": "gap_high_noise",
    "validation_epsilon_mse": "gap_val_epsilon",
    "gaussian_w2_squared": "gap_w2",
}
DIAGNOSTICS = ["grad_cos_target_aux1_init", "grad_cos_target_aux2_init",
               "grad_cos_target_aux_mean_init", "covariance_distance",
               "noised_score_map_distance"]
MANIFEST_FIELDS = ["design_id", "training_design_id", "capacity",
                   "time_embedding_dim", "class_embedding_dim", "hidden_width",
                   "hidden_layers", "K", "d", "n_target_train", "n_aux_train",
                   "T", "beta_start", "beta_end", "batch_size", "learning_rate",
                   "training_steps", "sampling_mode", "lambda_high", "lambda_low",
                   "n_validation", "n_test", "score_risk_mc_samples", "n_generated"]


def pair_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"design_id", "pair_id", "setting_id", "model_type",
                "rotation_deg", "seed", "capacity"}
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"Metrics missing pairing columns: {sorted(missing)}")
    baseline = metrics[metrics.model_type == "target_only"]
    counts = baseline.groupby("pair_id", dropna=False).size()
    if (counts != 1).any():
        raise ValueError(f"Each pair_id requires exactly one target-only baseline: {counts[counts != 1].to_dict()}")
    joint = metrics[metrics.model_type == "joint_conditional"]
    duplicated = joint.duplicated(["pair_id", "rotation_deg"], keep=False)
    if duplicated.any():
        raise ValueError("Each pair_id/rotation requires exactly one joint row")
    paired = joint.merge(baseline, on="pair_id", suffixes=("_joint", "_target"),
                         how="left", validate="many_to_one", indicator=True)
    if (paired._merge != "both").any():
        raise ValueError("Joint rows have no exact target-only pair_id")
    for column in ("design_id", "seed", "capacity"):
        if not (paired[f"{column}_joint"] == paired[f"{column}_target"]).all():
            raise ValueError(f"pair_id collision for {column}")
    out = paired[["design_id_joint", "pair_id", "seed_joint", "capacity_joint",
                  "rotation_deg_joint"]].rename(columns={
                      "design_id_joint": "design_id", "seed_joint": "seed",
                      "capacity_joint": "capacity", "rotation_deg_joint": "rotation_deg"})
    for metric, gap in METRICS.items():
        out[gap] = paired[f"{metric}_joint"] - paired[f"{metric}_target"]
    for column in DIAGNOSTICS:
        out[column] = paired[f"{column}_joint"]
    return out


def _status_column(metric: str) -> str:
    if metric == "gap_score":
        return "score_transfer_status"
    if metric == "gap_w2":
        return "sample_transfer_status"
    return "diagnostic_status"


def summarize(paired: pd.DataFrame, expected_seeds: Iterable[int] = range(20)) -> pd.DataFrame:
    expected = {int(seed) for seed in expected_seeds}
    if not expected:
        raise ValueError("expected_seeds cannot be empty")
    rows: list[dict[str, object]] = []
    grouping = ["design_id", "capacity", "rotation_deg"]
    for (design, capacity, rotation), group in paired.groupby(grouping, sort=True):
        observed = {int(seed) for seed in group.seed.unique()}
        missing, extra = sorted(expected - observed), sorted(observed - expected)
        seed_complete = observed == expected
        for gap in METRICS.values():
            available = group.dropna(subset=[gap])
            values = available[gap].to_numpy(dtype=float)
            available_seeds = {int(seed) for seed in available.seed.unique()}
            n = len(values)
            mean = float(values.mean()) if n else np.nan
            se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else np.nan
            lo, hi = ((mean - critical * se, mean + critical * se)
                      if n > 1 else (np.nan, np.nan))
            complete = seed_complete and available_seeds == expected and n == len(expected)
            status = ("incomplete" if not complete else "positive" if hi < 0
                      else "negative" if lo > 0 else "inconclusive")
            status_column = _status_column(gap)
            row: dict[str, object] = {
                "design_id": design, "capacity": capacity, "rotation_deg": rotation,
                "metric": gap, "expected_n": len(expected), "observed_n": len(observed),
                "n_available": n, "missing_seeds": ";".join(map(str, missing)),
                "extra_seeds": ";".join(map(str, extra)), "mean_gap": mean,
                "standard_error": se, "ci95_low": lo, "ci95_high": hi,
                "n_gap_lt_zero": int((values < 0).sum()),
                "n_gap_gt_zero": int((values > 0).sum()),
                "score_transfer_status": "", "sample_transfer_status": "",
                "diagnostic_status": "",
            }
            row[status_column] = status
            rows.append(row)
    return pd.DataFrame(rows)


def design_manifest(metrics: pd.DataFrame) -> pd.DataFrame:
    missing = set(MANIFEST_FIELDS) - set(metrics)
    if missing:
        raise ValueError(f"Metrics missing design manifest fields: {sorted(missing)}")
    manifest = metrics[MANIFEST_FIELDS].drop_duplicates()
    counts = manifest.groupby("design_id").size()
    if (counts != 1).any():
        raise ValueError("A design_id maps to multiple parameter payloads")
    return manifest.sort_values("design_id").reset_index(drop=True)


def _interval_stats(group: pd.DataFrame, column: str) -> pd.DataFrame:
    records = []
    for rotation, values_frame in group.groupby("rotation_deg"):
        values = values_frame[column].dropna().to_numpy(dtype=float)
        n = len(values); mean = float(values.mean()) if n else np.nan
        se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else 0.0
        records.append({"rotation_deg": rotation, "mean": mean,
                        "half_width": critical * se})
    return pd.DataFrame(records)


def _line(paired: pd.DataFrame, columns: list[str], ylabel: str, path: Path) -> None:
    plotted = False
    for (design, capacity), group in paired.groupby(["design_id", "capacity"]):
        for column in columns:
            stats = _interval_stats(group, column).dropna(subset=["mean"])
            if stats.empty:
                continue
            label = f"{capacity}/{design[:8]}: {column}"
            plt.errorbar(stats.rotation_deg, stats["mean"], yerr=stats.half_width,
                         marker="o", label=label)
            plotted = True
    plt.axhline(0, color="black", linewidth=.8); plt.xlabel("Rotation (degrees)")
    plt.ylabel(ylabel)
    if plotted:
        plt.legend(fontsize=7)
    else:
        plt.text(.5, .5, "No available values", ha="center", va="center",
                 transform=plt.gca().transAxes)
    plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def _latex_table(summary: pd.DataFrame, path: Path) -> None:
    score = summary[summary.metric == "gap_score"].copy()
    sample = summary[summary.metric == "gap_w2"].copy()
    keys = ["design_id", "capacity", "rotation_deg"]
    score = score[keys + ["mean_gap", "standard_error", "ci95_low", "ci95_high",
                          "score_transfer_status"]].rename(columns={
                              "mean_gap": "score_mean", "standard_error": "score_se",
                              "ci95_low": "score_ci_low", "ci95_high": "score_ci_high"})
    sample = sample[keys + ["mean_gap", "standard_error", "ci95_low", "ci95_high",
                            "sample_transfer_status"]].rename(columns={
                                "mean_gap": "w2_mean", "standard_error": "w2_se",
                                "ci95_low": "w2_ci_low", "ci95_high": "w2_ci_high"})
    score.merge(sample, on=keys, how="outer").to_latex(path, index=False,
                                                        float_format="%.4g")


def analyze(results_dir: Path, expected_seeds: Iterable[int] = range(20),
            selected_design_id: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = consolidate_seed_metrics(results_dir)
    manifest = design_manifest(metrics)
    if selected_design_id is not None:
        if selected_design_id not in set(metrics.design_id.astype(str)):
            raise ValueError(f"Unknown design_id={selected_design_id}")
        metrics = metrics[metrics.design_id.astype(str) == selected_design_id]
        manifest = manifest[manifest.design_id.astype(str) == selected_design_id]
    manifest.to_csv(results_dir / "design_manifest.csv", index=False)
    paired = pair_metrics(metrics); summary = summarize(paired, expected_seeds)
    paired.to_csv(results_dir / "paired_gaps.csv", index=False)
    summary.to_csv(results_dir / "summary_by_angle_capacity.csv", index=False)
    tables, figures = results_dir / "tables", results_dir / "figures"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    _latex_table(summary, tables / "rotation_transfer_summary.tex")
    _line(paired, ["gap_score"], "Integrated score-risk gap",
          figures / "score_gap_by_rotation.png")
    _line(paired, ["gap_low_noise", "gap_mid_noise", "gap_high_noise"],
          "Score-risk gap", figures / "noise_bin_gaps.png")
    _line(paired, ["gap_w2"], "Gaussian W2 squared gap",
          figures / "w2_gap_by_rotation.png")
    _line(paired.assign(negative_score=(paired.gap_score > 0).astype(float)),
          ["negative_score"], "Negative score-transfer seed fraction",
          figures / "negative_score_fraction.png")
    sample_fraction = paired.assign(
        negative_sample=np.where(paired.gap_w2.isna(), np.nan,
                                 (paired.gap_w2 > 0).astype(float)))
    _line(sample_fraction, ["negative_sample"],
          "Negative sample-transfer seed fraction",
          figures / "negative_sample_fraction.png")
    _line(paired, ["grad_cos_target_aux_mean_init"],
          "Initial shared-gradient cosine", figures / "gradient_cosine.png")
    _line(paired, ["covariance_distance", "noised_score_map_distance"],
          "Mismatch distance", figures / "mismatch_distances.png")
    return paired, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--expected-seeds", nargs="*", type=int, default=list(range(20)))
    parser.add_argument("--design-id")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(Path(args.results_dir), args.expected_seeds, args.design_id)
