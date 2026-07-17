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
THREE_MODEL_METRICS = [*METRICS.keys(), "mean_error", "covariance_error"]
THREE_MODEL_COMPARISONS = {
    "joint_conditional_minus_unconditional": ("joint_conditional", "unconditional"),
    "joint_conditional_minus_target_only_conditional": (
        "joint_conditional", "target_only"),
    "target_only_conditional_minus_unconditional": ("target_only", "unconditional"),
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


def pair_three_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return long-form gaps while retaining unavailable comparisons as NaN."""
    required = {"design_id", "pair_id", "setting_id", "model_type",
                "rotation_deg", "seed", "capacity"}
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"Metrics missing pairing columns: {sorted(missing)}")
    baselines = metrics[metrics.model_type.isin(["target_only", "unconditional"])]
    duplicated = baselines.duplicated(["pair_id", "model_type"], keep=False)
    if duplicated.any():
        raise ValueError("Each pair_id requires at most one row per angle-independent baseline")
    joint = metrics[metrics.model_type == "joint_conditional"]
    if joint.duplicated(["pair_id", "rotation_deg"], keep=False).any():
        raise ValueError("Each pair_id/rotation requires exactly one joint row")
    rows: list[dict[str, object]] = []
    for _, joint_row in joint.iterrows():
        same_pair = metrics[metrics.pair_id.astype(str) == str(joint_row.pair_id)]
        models = {str(row.model_type): row for _, row in same_pair.iterrows()
                  if row.model_type != "joint_conditional"}
        models["joint_conditional"] = joint_row
        for comparison, (left, right) in THREE_MODEL_COMPARISONS.items():
            has_models = left in models and right in models
            for metric in THREE_MODEL_METRICS:
                if metric not in metrics:
                    continue
                gap = np.nan
                if has_models:
                    left_value, right_value = models[left][metric], models[right][metric]
                    if pd.notna(left_value) and pd.notna(right_value):
                        gap = float(left_value) - float(right_value)
                rows.append({"design_id": joint_row.design_id,
                             "pair_id": joint_row.pair_id,
                             "seed": int(joint_row.seed),
                             "capacity": joint_row.capacity,
                             "rotation_deg": joint_row.rotation_deg,
                             "comparison": comparison, "metric": metric,
                             "gap": gap, "models_available": has_models})
    paired = pd.DataFrame(rows)
    if paired.empty:
        return paired
    index = ["design_id", "pair_id", "seed", "capacity", "rotation_deg", "metric"]
    wide = paired.pivot(index=index, columns="comparison", values="gap")
    names = list(THREE_MODEL_COMPARISONS)
    if all(name in wide for name in names):
        complete = wide[names].notna().all(axis=1)
        if complete.any() and not np.allclose(
            wide.loc[complete, names[0]],
            wide.loc[complete, names[1]] + wide.loc[complete, names[2]],
            rtol=1e-10, atol=1e-12):
            raise ValueError("Three-model gap identity failed")
    return paired


def summarize_three_models(
    paired: pd.DataFrame, expected_seeds: Iterable[int] = range(20),
) -> pd.DataFrame:
    expected = {int(seed) for seed in expected_seeds}
    if not expected:
        raise ValueError("expected_seeds cannot be empty")
    rows: list[dict[str, object]] = []
    grouping = ["design_id", "capacity", "rotation_deg", "comparison", "metric"]
    for values, group in paired.groupby(grouping, sort=True):
        base = dict(zip(grouping, values))
        available = group.dropna(subset=["gap"])
        observed = {int(seed) for seed in available.seed.unique()}
        gaps = available.gap.to_numpy(dtype=float)
        n = len(gaps)
        mean = float(gaps.mean()) if n else np.nan
        se = float(gaps.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        critical = float(student_t.ppf(.975, n - 1)) if n > 1 else np.nan
        lo, hi = ((mean - critical * se, mean + critical * se)
                  if n > 1 else (np.nan, np.nan))
        complete = observed == expected and n == len(expected)
        sample_metric = base["metric"] in {"gaussian_w2_squared", "mean_error",
                                            "covariance_error"}
        status = ("incomplete" if not complete else "positive" if hi < 0
                  else "negative" if lo > 0 else "inconclusive")
        rows.append({**base, "expected_n": len(expected), "observed_n": n,
                     "missing_seeds": ";".join(map(str, sorted(expected - observed))),
                     "paired_mean": mean, "standard_error": se,
                     "ci95_low": lo, "ci95_high": hi,
                     "n_gap_lt_zero": int((gaps < 0).sum()),
                     "n_gap_gt_zero": int((gaps > 0).sum()),
                     "completeness": "complete" if complete else "incomplete",
                     "score_transfer_status": (status if base["metric"].endswith("score_risk") else ""),
                     "sample_transfer_status": (status if sample_metric else ""),
                     "diagnostic_status": (status if not sample_metric and not base["metric"].endswith("score_risk") else "")})
    return pd.DataFrame(rows)


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


def _three_model_line(paired: pd.DataFrame, metric: str, comparison: str,
                      path: Path) -> None:
    data = paired[(paired.metric == metric) & (paired.comparison == comparison)]
    plt.figure(figsize=(7, 4.5))
    plotted = False
    for (design, capacity), group in data.groupby(["design_id", "capacity"]):
        records = []
        for rotation, values in group.groupby("rotation_deg"):
            gaps = values.gap.dropna().to_numpy(dtype=float)
            if not len(gaps):
                continue
            se = float(gaps.std(ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else 0.0
            critical = float(student_t.ppf(.975, len(gaps) - 1)) if len(gaps) > 1 else 0.0
            records.append((rotation, float(gaps.mean()), critical * se))
        if records:
            frame = pd.DataFrame(records, columns=["rotation", "mean", "half_width"])
            plt.errorbar(frame.rotation, frame["mean"], yerr=frame.half_width,
                         marker="o", label=f"{capacity}/{design[:8]}")
            plotted = True
    plt.axhline(0, color="black", linewidth=.8)
    plt.xlabel("Rotation (degrees)"); plt.ylabel(f"{metric} gap")
    if plotted:
        plt.legend(fontsize=7)
    else:
        plt.text(.5, .5, "Comparison unavailable", ha="center", va="center",
                 transform=plt.gca().transAxes)
    plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def write_three_model_outputs(metrics: pd.DataFrame, results_dir: Path,
                              expected_seeds: Iterable[int]) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = pair_three_models(metrics)
    summary = summarize_three_models(paired, expected_seeds)
    paired.to_csv(results_dir / "paired_three_model_gaps.csv", index=False)
    summary.to_csv(results_dir / "summary_three_model_gaps.csv", index=False)
    tables, figures = results_dir / "tables", results_dir / "figures"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    primary = summary[
        summary.comparison == "joint_conditional_minus_unconditional"]
    primary.to_latex(tables / "rotation_primary_unconditional_summary.tex",
                     index=False, float_format="%.4g")
    summary.to_latex(tables / "rotation_three_model_summary.tex", index=False,
                     float_format="%.4g")
    figure_specs = [
        ("score_risk", "joint_conditional_minus_unconditional",
         "score_gap_joint_minus_unconditional.png"),
        ("gaussian_w2_squared", "joint_conditional_minus_unconditional",
         "w2_gap_joint_minus_unconditional.png"),
        ("score_risk", "target_only_conditional_minus_unconditional",
         "parameterization_score_gap.png"),
        ("gaussian_w2_squared", "target_only_conditional_minus_unconditional",
         "parameterization_w2_gap.png"),
    ]
    for metric, comparison, filename in figure_specs:
        _three_model_line(paired, metric, comparison, figures / filename)
    return paired, summary


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
    # Keep pair_metrics strict and unchanged, but omit joint rows whose historical
    # C_T baseline is absent so the additive three-model analysis can still run.
    target_counts = (metrics[metrics.model_type == "target_only"]
                     .groupby("pair_id").size())
    valid_pair_ids = set(target_counts[target_counts == 1].index.astype(str))
    legacy_metrics = metrics[
        (metrics.model_type == "target_only")
        | ((metrics.model_type == "joint_conditional")
           & metrics.pair_id.astype(str).isin(valid_pair_ids))]
    if (legacy_metrics.model_type == "joint_conditional").any():
        paired = pair_metrics(legacy_metrics)
        summary = summarize(paired, expected_seeds)
    else:
        paired = pd.DataFrame(columns=["design_id", "pair_id", "seed", "capacity",
                                       "rotation_deg", *METRICS.values(), *DIAGNOSTICS])
        summary = pd.DataFrame(columns=[
            "design_id", "capacity", "rotation_deg", "metric", "expected_n",
            "observed_n", "n_available", "missing_seeds", "extra_seeds",
            "mean_gap", "standard_error", "ci95_low", "ci95_high",
            "n_gap_lt_zero", "n_gap_gt_zero", "score_transfer_status",
            "sample_transfer_status", "diagnostic_status"])
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
    write_three_model_outputs(metrics, results_dir, expected_seeds)
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
