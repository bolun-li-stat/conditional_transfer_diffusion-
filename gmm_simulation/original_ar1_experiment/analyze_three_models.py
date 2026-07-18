"""Additive paired analysis for U_T, C_T, and C_J in the AR(1) experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from plot_results import load_metrics


METRICS = ["score_risk", "validation_epsilon_mse", "gaussian_w2_squared",
           "mmd_rbf", "mean_error", "covariance_error"]
COMPARISONS = {
    "joint_conditional_minus_unconditional": ("conditional", "unconditional"),
    "joint_conditional_minus_target_only_conditional": (
        "conditional", "target_only_conditional"),
    "target_only_conditional_minus_unconditional": (
        "target_only_conditional", "unconditional"),
}
PAIR_KEYS = ["experiment_type", "covariance_scenario", "rho", "mismatch_level",
             "target_rho", "auxiliary_rhos", "sqrt_alpha_bar_T", "K", "d",
             "Delta", "n", "n_target_train", "n_aux_train", "seed",
             "sampling_mode", "training_steps"]
SUMMARY_KEYS = [
    key for key in PAIR_KEYS if key not in {"seed", "auxiliary_rhos"}
]


def pair_three_models(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [key for key in PAIR_KEYS if key in metrics]
    required = {"model_type", "seed", *METRICS} - set(metrics)
    if required:
        raise ValueError(f"Metrics missing columns: {sorted(required)}")
    duplicates = metrics.duplicated(keys + ["model_type"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate model rows prevent strict seed/setting pairing")
    rows: list[dict[str, object]] = []
    for values, group in metrics.groupby(keys, dropna=False, sort=True):
        base = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        indexed = group.set_index("model_type")
        for comparison, (left, right) in COMPARISONS.items():
            available = left in indexed.index and right in indexed.index
            for metric in METRICS:
                gap = (float(indexed.loc[left, metric]) - float(indexed.loc[right, metric])
                       if available else np.nan)
                rows.append({**base, "comparison": comparison, "metric": metric,
                             "gap": gap, "available": available})
    paired = pd.DataFrame(rows)
    if paired.empty:
        return paired
    for _, group in paired.groupby(keys + ["metric"], dropna=False):
        gaps = group.set_index("comparison").gap
        if set(COMPARISONS).issubset(gaps.index) and gaps.notna().all():
            if not np.isclose(gaps.iloc[0], gaps.iloc[1] + gaps.iloc[2],
                              rtol=1e-10, atol=1e-12):
                raise ValueError(
                    f"Three-model gap identity failed for {group.metric.iloc[0]}")
    return paired


def summarize_three_models(
    paired: pd.DataFrame, expected_seeds: Iterable[int] = range(20),
) -> pd.DataFrame:
    expected = {int(seed) for seed in expected_seeds}
    if not expected:
        raise ValueError("expected_seeds cannot be empty")
    grouping = [key for key in SUMMARY_KEYS if key in paired]
    grouping += ["comparison", "metric"]
    rows: list[dict[str, object]] = []
    for values, group in paired.groupby(grouping, dropna=False, sort=True):
        base = dict(zip(grouping, values if isinstance(values, tuple) else (values,)))
        available = group.dropna(subset=["gap"])
        observed = {int(seed) for seed in available.seed.unique()}
        gaps = available.gap.to_numpy(dtype=float)
        n = len(gaps)
        mean = float(gaps.mean()) if n else np.nan
        se = float(gaps.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        critical = float(student_t.ppf(.975, n - 1)) if n > 1 else np.nan
        lo, hi = ((mean - critical * se, mean + critical * se)
                  if n > 1 else (np.nan, np.nan))
        complete = observed == expected and n == len(expected) and n > 1
        rows.append({**base, "paired_mean": mean, "standard_error": se,
                     "ci95_low": lo, "ci95_high": hi,
                     "n_gap_lt_zero": int((gaps < 0).sum()),
                     "n_gap_gt_zero": int((gaps > 0).sum()),
                     "expected_n": len(expected), "observed_n": n,
                     "missing_seeds": ";".join(map(str, sorted(expected - observed))),
                     "completeness": "complete" if complete else "incomplete",
                     "status": ("unavailable" if n == 0 else
                                "complete" if complete else "incomplete")})
    return pd.DataFrame(rows)


def _covariance_setting(row: pd.Series) -> str:
    if row["covariance_scenario"] == "shared":
        return f"rho={row['rho']}"
    return f"mismatch={row['mismatch_level']}"


def prepare_gap_plot_series(
    summary: pd.DataFrame, experiment_type: str, metric: str,
) -> list[dict[str, object]]:
    """Prepare scientifically distinct curves and paired Student-t error bars."""
    x_name = "n_target_train" if experiment_type == "low_target_data" else "n"
    data = summary[
        (summary.experiment_type == experiment_type) & (summary.metric == metric)
    ].copy()
    if data.empty:
        return []
    data["covariance_setting"] = data.apply(_covariance_setting, axis=1)
    group_keys = ["comparison", "covariance_scenario", "covariance_setting"]
    series: list[dict[str, object]] = []
    for keys, group in data.groupby(group_keys, dropna=False, sort=True):
        valid = group[
            np.isfinite(group["paired_mean"])
            & np.isfinite(group["ci95_low"])
            & np.isfinite(group["ci95_high"])
            & group[x_name].notna()
        ].sort_values(x_name)
        if valid.empty:
            continue
        comparison, scenario, setting = keys
        series.append({
            "label": f"{comparison} | {scenario} | {setting}",
            "comparison": comparison,
            "covariance_scenario": scenario,
            "covariance_setting": setting,
            "x_name": x_name,
            "x": valid[x_name].to_numpy(dtype=float),
            "mean": valid["paired_mean"].to_numpy(dtype=float),
            "lower_error": (
                valid["paired_mean"] - valid["ci95_low"]
            ).to_numpy(dtype=float),
            "upper_error": (
                valid["ci95_high"] - valid["paired_mean"]
            ).to_numpy(dtype=float),
        })
    return series


def _gap_plot(summary: pd.DataFrame, experiment_type: str, metric: str,
              path: Path) -> None:
    series = prepare_gap_plot_series(summary, experiment_type, metric)
    plt.figure(figsize=(9, 4.8))
    if not series:
        plt.text(.5, .5, "No available paired values", ha="center", va="center",
                 transform=plt.gca().transAxes)
    else:
        for curve in series:
            plt.errorbar(
                curve["x"], curve["mean"],
                yerr=np.vstack([curve["lower_error"], curve["upper_error"]]),
                marker="o", label=curve["label"])
        plt.axhline(0, color="black", linewidth=.8)
        plt.legend(fontsize=7)
    x_name = "n_target_train" if experiment_type == "low_target_data" else "n"
    plt.xlabel(x_name)
    plt.ylabel(f"Paired {metric} gap")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def analyze(results_dir: Path, expected_seeds: Iterable[int] = range(20)) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = load_metrics(results_dir)
    if metrics.empty:
        raise FileNotFoundError(f"No metrics found under {results_dir}")
    paired = pair_three_models(metrics)
    summary = summarize_three_models(paired, expected_seeds)
    paired.to_csv(results_dir / "paired_three_model_gaps.csv", index=False)
    summary.to_csv(results_dir / "summary_three_model_gaps.csv", index=False)
    tables, figures = results_dir / "tables", results_dir / "figures"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    summary.to_latex(tables / "three_model_transfer_summary.tex", index=False,
                     float_format="%.4g")
    for experiment_type in ("low_target_data", "same_total_budget"):
        _gap_plot(
            summary, experiment_type, "score_risk",
            figures / f"{experiment_type}_three_model_score_gaps.png")
        _gap_plot(
            summary, experiment_type, "gaussian_w2_squared",
            figures / f"{experiment_type}_three_model_w2_gaps.png")
    return paired, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results_T1000_K3")
    parser.add_argument("--expected-seeds", nargs="*", type=int,
                        default=list(range(20)))
    args = parser.parse_args()
    analyze(Path(args.results_dir), args.expected_seeds)
