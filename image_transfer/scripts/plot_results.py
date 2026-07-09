from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXP_DIR = {"A": "A_equal_target", "B": "B_equal_total", "C": "C_similarity_sweep"}
METRICS = ["fid_target", "kid_target_mean", "validation_epsilon_mse_target", "classifier_target_top1_acc", "auxiliary_leakage_rate"]


def _plot_metric_curves(df: pd.DataFrame, figdir: Path, experiment: str) -> None:
    for metric in METRICS:
        if metric not in df:
            continue
        plt.figure(figsize=(7, 4))
        for name, group in df.groupby("model_type"):
            group = group.sort_values("n0")
            plt.plot(group["n0"], group[metric], marker="o", label=name)
        plt.xlabel("n0")
        plt.ylabel(metric)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(figdir / f"{experiment}_{metric}_curves.png", dpi=160)
        plt.close()


def _plot_gap(df: pd.DataFrame, figdir: Path, experiment: str) -> None:
    if "fid_target" not in df:
        return
    baseline_name = "unconditional_n0" if experiment == "A" else "unconditional_equal_total"
    baseline = df[df["model_type"] == baseline_name][["target_synset", "n0", "seed", "fid_target"]]
    if baseline.empty:
        return
    merged = df.merge(baseline, on=["target_synset", "n0", "seed"], suffixes=("", "_baseline"))
    merged = merged[~merged["model_type"].eq(baseline_name)]
    if merged.empty:
        return
    merged["delta_fid"] = merged["fid_target_baseline"] - merged["fid_target"]
    plt.figure(figsize=(7, 4))
    for name, group in merged.groupby("model_type"):
        group = group.sort_values("n0")
        plt.plot(group["n0"], group["delta_fid"], marker="o", label=name)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("n0")
    plt.ylabel("FID baseline - FID model")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figdir / f"{experiment}_transfer_gap.png", dpi=160)
    plt.close()


def _plot_composition(df: pd.DataFrame, figdir: Path) -> None:
    order = ["close_only", "mostly_close", "balanced_mix", "mostly_far", "far_only"]
    cdf = df[df["aux_set"].isin(order)].copy()
    if cdf.empty:
        return
    cdf["composition_order"] = cdf["aux_set"].map({name: i for i, name in enumerate(order)})
    for metric in ["fid_target", "validation_epsilon_mse_target", "auxiliary_leakage_rate"]:
        if metric not in cdf:
            continue
        summary = cdf.groupby(["aux_set", "composition_order"])[metric].mean(numeric_only=True).reset_index().sort_values("composition_order")
        plt.figure(figsize=(7, 4))
        plt.plot(summary["aux_set"], summary[metric], marker="o")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(figdir / f"C_composition_{metric}.png", dpi=160)
        plt.close()
    if "average_auxiliary_similarity" in cdf and "fid_target" in cdf:
        plt.figure(figsize=(5, 4))
        plt.scatter(cdf["average_auxiliary_similarity"], cdf["fid_target"])
        plt.xlabel("average auxiliary feature similarity")
        plt.ylabel("FID")
        plt.tight_layout()
        plt.savefig(figdir / "C_similarity_fid_scatter.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--experiment", choices=["A", "B", "C"], required=True)
    args = parser.parse_args()
    exp_dir = Path(args.results_root) / EXP_DIR[args.experiment]
    metrics_path = exp_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    figdir = exp_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_path)
    _plot_metric_curves(df, figdir, args.experiment)
    _plot_gap(df, figdir, args.experiment)
    if args.experiment == "C":
        _plot_composition(df, figdir)
    print(figdir)


if __name__ == "__main__":
    main()
