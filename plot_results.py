"""Plot and summarize DDPM Gaussian-mixture transfer experiment results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils import ensure_dir, standard_error

METRICS = ["score_risk", "gaussian_w2_squared", "covariance_error", "mean_error", "mmd_rbf", "validation_epsilon_mse"]


def load_metrics(results_dir: Path) -> pd.DataFrame:
    files = [p for p in results_dir.glob("*.csv") if "metrics" in p.name or "smoke" in p.name or "results" in p.name]
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if "model_type" in df.columns and "score_risk" in df.columns:
            frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["experiment_type", "covariance_scenario", "rho", "mismatch_level", "K", "n", "n_target_train", "model_type", "sampling_mode"] if c in df.columns]
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        for metric in METRICS:
            if metric in sub.columns:
                vals = sub[metric].to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(np.nanmean(vals))
                row[f"{metric}_se"] = standard_error(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_metric_vs(df: pd.DataFrame, x: str, metric: str, experiment: str, out: Path) -> None:
    sub = df[df["experiment_type"] == experiment]
    if sub.empty or x not in sub.columns:
        return
    for scenario, ss in sub.groupby("covariance_scenario", dropna=False):
        plt.figure(figsize=(7, 4.5))
        for model, mm in ss.groupby("model_type"):
            mean_col, se_col = f"{metric}_mean", f"{metric}_se"
            if mean_col not in mm:
                continue
            mm = mm.sort_values(x)
            plt.errorbar(mm[x], mm[mean_col], yerr=mm.get(se_col, 0), marker="o", capsize=3, label=model)
        plt.xlabel(x)
        plt.ylabel(metric)
        plt.title(f"{metric} vs {x} | {experiment} | {scenario}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"{experiment}_{scenario}_{metric}_vs_{x}.png", dpi=160)
        plt.close()


def plot_metric_curves(agg: pd.DataFrame, figure_dir: Path) -> None:
    ensure_dir(figure_dir)
    _plot_metric_vs(agg, "n_target_train", "score_risk", "low_target_data", figure_dir)
    _plot_metric_vs(agg, "n", "score_risk", "same_total_budget", figure_dir)
    _plot_metric_vs(agg, "n_target_train", "gaussian_w2_squared", "low_target_data", figure_dir)
    _plot_metric_vs(agg, "n", "gaussian_w2_squared", "same_total_budget", figure_dir)
    _plot_metric_vs(agg, "n", "validation_epsilon_mse", "same_total_budget", figure_dir)
    _plot_metric_vs(agg, "n_target_train", "validation_epsilon_mse", "low_target_data", figure_dir)


def plot_mismatch_differences(df: pd.DataFrame, figure_dir: Path) -> None:
    ensure_dir(figure_dir)
    if "mismatch_level" not in df.columns:
        return
    for experiment in df["experiment_type"].dropna().unique():
        sub = df[(df["experiment_type"] == experiment) & (df["covariance_scenario"] == "mismatch")]
        if sub.empty:
            continue
        pivot_keys = [c for c in ["seed", "n", "n_target_train", "mismatch_level"] if c in sub.columns]
        for metric in ["score_risk", "gaussian_w2_squared", "covariance_error", "mean_error", "mmd_rbf"]:
            if metric not in sub.columns:
                continue
            wide = sub.pivot_table(index=pivot_keys, columns="model_type", values=metric, aggfunc="mean").reset_index()
            if not {"conditional", "unconditional"}.issubset(wide.columns):
                continue
            wide["difference"] = wide["conditional"] - wide["unconditional"]
            summary = wide.groupby("mismatch_level")["difference"].agg(["mean", standard_error]).reset_index()
            order = ["0", "mild", "medium", "strong"]
            summary["mismatch_level"] = pd.Categorical(summary["mismatch_level"], categories=order, ordered=True)
            summary = summary.sort_values("mismatch_level")
            plt.figure(figsize=(7, 4.5))
            plt.axhline(0, color="black", linewidth=1)
            plt.errorbar(summary["mismatch_level"].astype(str), summary["mean"], yerr=summary["standard_error"], marker="o", capsize=3)
            plt.ylabel(f"conditional - unconditional {metric}")
            plt.xlabel("mismatch_level")
            plt.title(f"Transfer gap by mismatch | {experiment}")
            plt.tight_layout()
            plt.savefig(figure_dir / f"{experiment}_{metric}_conditional_minus_unconditional_by_mismatch.png", dpi=160)
            plt.close()


def plot_summary_bars(agg: pd.DataFrame, figure_dir: Path) -> None:
    if agg.empty:
        return
    for metric in ["score_risk", "gaussian_w2_squared", "covariance_error", "mean_error", "mmd_rbf"]:
        mean_col = f"{metric}_mean"
        if mean_col not in agg:
            continue
        sub = agg.groupby(["experiment_type", "model_type"], dropna=False)[mean_col].mean().reset_index()
        if sub.empty:
            continue
        plt.figure(figsize=(8, 4.5))
        labels = sorted(sub["experiment_type"].dropna().unique())
        x = np.arange(len(labels))
        width = 0.35
        for i, model in enumerate(["unconditional", "conditional"]):
            vals = [sub[(sub["experiment_type"] == lab) & (sub["model_type"] == model)][mean_col].mean() for lab in labels]
            plt.bar(x + (i - 0.5) * width, vals, width=width, label=model)
        plt.xticks(x, labels, rotation=20)
        plt.ylabel(metric)
        plt.title(f"Overall model comparison: {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / f"summary_bar_{metric}.png", dpi=160)
        plt.close()


def plot_training_losses(results_dir: Path, figure_dir: Path) -> None:
    logs = list((results_dir / "logs").glob("*_train_log.csv"))[:12]
    if not logs:
        return
    plt.figure(figsize=(8, 5))
    for p in logs:
        df = pd.read_csv(p)
        if {"step", "train_loss"}.issubset(df.columns):
            plt.plot(df["step"], df["train_loss"], label=p.stem[:24], alpha=0.8)
    plt.xlabel("step")
    plt.ylabel("training loss")
    plt.title("Representative training loss curves")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(figure_dir / "representative_training_loss_curves.png", dpi=160)
    plt.close()


def plot_pca_samples(results_dir: Path, figure_dir: Path) -> None:
    sample_dir = results_dir / "samples"
    sample_files = list(sample_dir.glob("*_unconditional_samples.npy"))
    for uncond in sample_files[:4]:
        stem = uncond.name.replace("_unconditional_samples.npy", "")
        cond = sample_dir / f"{stem}_conditional_samples.npy"
        target = sample_dir / f"{stem}_target_test.npy"
        if not cond.exists() or not target.exists():
            continue
        x_true = np.load(target)[:1000]
        x_u = np.load(uncond)[:1000]
        x_c = np.load(cond)[:1000]
        all_x = np.concatenate([x_true, x_u, x_c], axis=0)
        z = PCA(n_components=2).fit_transform(all_x)
        n = len(x_true)
        plt.figure(figsize=(6, 5))
        plt.scatter(z[:n, 0], z[:n, 1], s=8, alpha=0.5, label="true target")
        plt.scatter(z[n : n + len(x_u), 0], z[n : n + len(x_u), 1], s=8, alpha=0.5, label="unconditional")
        plt.scatter(z[n + len(x_u) :, 0], z[n + len(x_u) :, 1], s=8, alpha=0.5, label="conditional")
        plt.legend()
        plt.title("PCA: true vs generated target samples")
        plt.tight_layout()
        plt.savefig(figure_dir / f"pca_{stem}.png", dpi=160)
        plt.close()


def textual_summary(df: pd.DataFrame) -> str:
    if df.empty or "score_risk" not in df:
        return "No score-risk results found."
    lines = []
    keys = [c for c in ["experiment_type", "covariance_scenario", "mismatch_level", "n", "n_target_train"] if c in df.columns]
    for group, sub in df.groupby(keys, dropna=False):
        means = sub.groupby("model_type")["score_risk"].mean()
        if {"conditional", "unconditional"}.issubset(means.index):
            diff = means["conditional"] - means["unconditional"]
            verdict = "conditional better" if diff < 0 else "negative transfer / unconditional better" if diff > 0 else "tied"
            lines.append(f"{dict(zip(keys, group if isinstance(group, tuple) else (group,)))}: diff={diff:.4g} ({verdict})")
    return "\n".join(lines) if lines else "Need paired conditional and unconditional rows for summaries."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--make-pca", action="store_true")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    figure_dir = ensure_dir(results_dir / "figures")
    df = load_metrics(results_dir)
    if df.empty:
        print("No metrics CSV files found.")
        return
    agg = aggregate(df)
    agg.to_csv(results_dir / "aggregated_metrics.csv", index=False)
    plot_metric_curves(agg, figure_dir)
    plot_mismatch_differences(df, figure_dir)
    plot_summary_bars(agg, figure_dir)
    plot_training_losses(results_dir, figure_dir)
    if args.make_pca:
        plot_pca_samples(results_dir, figure_dir)
    print(textual_summary(df))
    print(f"Saved figures to {figure_dir}")


if __name__ == "__main__":
    main()
