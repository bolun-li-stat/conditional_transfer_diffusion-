from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from image_transfer.scripts.aggregate_results import aggregate_results, t95_confidence_interval

EXP_DIR = {"A": "A_equal_target", "B": "B_equal_total", "C": "C_similarity_sweep"}
TRANSFER_METRICS = [
    "test_epsilon_mse_target",
    "fid_target",
    "kid_target_mean",
    "classifier_target_top1_acc",
    "auxiliary_leakage_rate",
    "density_target",
    "coverage_target",
]
NOISE_BINS = [
    "test_epsilon_mse_low_noise",
    "test_epsilon_mse_mid_noise",
    "test_epsilon_mse_high_noise",
]


def _safe_name(value: Any) -> str:
    return str(value).replace("/", "-").replace(" ", "_")


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _plot_summary(frame: pd.DataFrame, x_column: str, identity_columns: list[str]) -> pd.DataFrame:
    group_columns = [column for column in [*identity_columns, "metric", x_column] if column in frame]
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row.update(t95_confidence_interval(pd.to_numeric(group["improvement_positive"], errors="coerce").tolist()))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_paired_axis(
    ax,
    pairs: pd.DataFrame,
    *,
    x_column: str,
    metric: str,
    show_raw: bool = True,
) -> pd.DataFrame:
    """Plot paired summaries; raw seeds are points only and are never connected."""

    frame = pairs[(pairs["pair_status"] == "completed") & (pairs["metric"] == metric)].copy()
    if frame.empty or x_column not in frame:
        return pd.DataFrame()
    frame[x_column] = pd.to_numeric(frame[x_column], errors="coerce")
    frame = frame.dropna(subset=[x_column, "improvement_positive"])
    conditioning = {
        "n0": ("auxiliary_ratio", "K_aux"),
        "m_per_aux": ("n0", "K_aux"),
        "K_aux": ("n0", "total_auxiliary_budget"),
        "average_auxiliary_similarity": ("n0", "m_per_aux", "K_aux"),
    }.get(x_column, ())
    identity_candidates = (
        ("training_protocol", "baseline_kind", *conditioning)
        if x_column == "average_auxiliary_similarity"
        else ("model_type", "aux_set", "training_protocol", "baseline_kind", *conditioning)
    )
    identity_columns = [column for column in identity_candidates if column in frame]
    summary = _plot_summary(frame, x_column, identity_columns)
    for identity, group in frame.groupby(identity_columns, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        label = " | ".join(str(value) for value in identity)
        aggregate = summary
        for column, value in zip(identity_columns, identity):
            aggregate = aggregate[aggregate[column].eq(value)]
        aggregate = aggregate.sort_values(x_column)
        if show_raw:
            ax.scatter(
                group[x_column],
                group["improvement_positive"],
                s=16,
                alpha=0.22,
                label=f"{label} raw",
            )
        means = pd.to_numeric(aggregate["mean"], errors="coerce").to_numpy(dtype=float)
        lower = pd.to_numeric(aggregate["ci95_lower"], errors="coerce").to_numpy(dtype=float)
        upper = pd.to_numeric(aggregate["ci95_upper"], errors="coerce").to_numpy(dtype=float)
        yerr = np.vstack([means - lower, upper - means])
        # A one-run group has no valid t interval; display the mean without an
        # invented confidence bar.
        yerr[~np.isfinite(yerr)] = 0.0
        ax.errorbar(
            aggregate[x_column].to_numpy(dtype=float),
            means,
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=f"{label} mean (95% t CI)",
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel(x_column)
    ax.set_ylabel(f"improvement-positive: {metric}")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=6)
    return summary


def plot_transfer_improvement(
    pairs: pd.DataFrame,
    figdir: Path,
    *,
    x_column: str,
    filename_stem: str,
    metrics: list[str] | None = None,
    show_raw: bool = True,
) -> list[Path]:
    outputs = []
    metrics = metrics or TRANSFER_METRICS
    for metric in metrics:
        if pairs.empty or metric not in set(pairs.get("metric", [])) or x_column not in pairs:
            continue
        figure, axis = plt.subplots(figsize=(8, 4.5))
        summary = _plot_paired_axis(axis, pairs, x_column=x_column, metric=metric, show_raw=show_raw)
        if summary.empty:
            plt.close(figure)
            continue
        figure.tight_layout()
        output = figdir / f"{filename_stem}_{_safe_name(metric)}.png"
        figure.savefig(output, dpi=180)
        plt.close(figure)
        outputs.append(output)
    return outputs


def _plot_noise_bins(metrics: pd.DataFrame, figdir: Path) -> Path | None:
    available = [column for column in NOISE_BINS if column in metrics]
    if metrics.empty or not available:
        return None
    identity_columns = [
        column
        for column in ("experiment", "target_synset", "model_type", "training_protocol", "n0", "m_per_aux", "K_aux")
        if column in metrics
    ]
    rows = []
    for identity, group in metrics.groupby(identity_columns, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        for column in available:
            stats = t95_confidence_interval(pd.to_numeric(group[column], errors="coerce").tolist())
            rows.append({**dict(zip(identity_columns, identity)), "noise_bin": column, **stats})
    summary = pd.DataFrame(rows)
    if summary.empty:
        return None
    figure, axis = plt.subplots(figsize=(9, 4.5))
    labels = [" | ".join(str(row[column]) for column in identity_columns) for _, row in summary.drop_duplicates(identity_columns).iterrows()]
    x = np.arange(len(available), dtype=float)
    width = 0.8 / max(len(labels), 1)
    for index, (identity, group) in enumerate(summary.groupby(identity_columns, dropna=False)):
        identity = identity if isinstance(identity, tuple) else (identity,)
        group = group.set_index("noise_bin").reindex(available)
        means = group["mean"].to_numpy(dtype=float)
        lower = group["ci95_lower"].to_numpy(dtype=float)
        upper = group["ci95_upper"].to_numpy(dtype=float)
        error = np.vstack([means - lower, upper - means])
        error[~np.isfinite(error)] = 0.0
        axis.bar(x + (index - (len(labels) - 1) / 2) * width, means, width, yerr=error, capsize=2, label=" | ".join(map(str, identity)))
    axis.set_xticks(x, [name.replace("test_epsilon_mse_", "").replace("_noise", "") for name in available])
    axis.set_ylabel("target epsilon MSE (mean, 95% t CI)")
    axis.legend(fontsize=6)
    figure.tight_layout()
    output = figdir / "target_denoising_mse_by_noise_bin.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def _plot_semantic_leakage(metrics: pd.DataFrame, figdir: Path) -> Path | None:
    required = {"classifier_target_top1_acc", "auxiliary_leakage_rate"}
    if metrics.empty or not required.issubset(metrics.columns):
        return None
    frame = metrics.dropna(subset=list(required))
    if frame.empty:
        return None
    figure, axis = plt.subplots(figsize=(6, 5))
    identity_columns = [column for column in ("model_type", "training_protocol") if column in frame]
    for identity, group in frame.groupby(identity_columns, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        axis.scatter(group["auxiliary_leakage_rate"], group["classifier_target_top1_acc"], alpha=0.65, label=" | ".join(map(str, identity)))
    axis.set_xlabel("auxiliary leakage rate (lower is better)")
    axis.set_ylabel("target classifier top-1 accuracy")
    axis.legend(fontsize=6)
    figure.tight_layout()
    output = figdir / "semantic_fidelity_vs_leakage.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def _plot_density_coverage(metrics: pd.DataFrame, figdir: Path) -> Path | None:
    if metrics.empty or not {"density_target", "coverage_target"}.issubset(metrics.columns):
        return None
    frame = metrics.dropna(subset=["density_target", "coverage_target"])
    if frame.empty:
        return None
    figure, axis = plt.subplots(figsize=(6, 5))
    identity_columns = [column for column in ("model_type", "training_protocol") if column in frame]
    for identity, group in frame.groupby(identity_columns, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        axis.scatter(group["coverage_target"], group["density_target"], alpha=0.65, label=" | ".join(map(str, identity)))
    axis.set_xlabel("coverage_target")
    axis.set_ylabel("density_target")
    axis.legend(fontsize=6)
    figure.tight_layout()
    output = figdir / "density_vs_coverage.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def _plot_sample_grid(results_root: Path, figdir: Path, max_models: int = 8, max_per_model: int = 8) -> Path | None:
    try:
        import torch
        from torchvision.utils import make_grid, save_image
    except Exception:
        return None
    sample_files = sorted(results_root.glob("**/samples/*_samples.pt"))[:max_models]
    rows = []
    for sample_path in sample_files:
        try:
            try:
                samples = torch.load(sample_path, map_location="cpu", weights_only=True)[:max_per_model]
            except TypeError:  # pragma: no cover - older PyTorch
                samples = torch.load(sample_path, map_location="cpu")[:max_per_model]
        except Exception:
            continue
        if len(samples):
            rows.append(samples)
    if not rows:
        return None
    output = figdir / "sample_grid.png"
    grid = make_grid(torch.cat(rows, dim=0), nrow=max_per_model, normalize=True, value_range=(-1, 1))
    save_image(grid, output)
    return output


def plot_results(results_root: str | Path, *, experiment: str | None = None, show_raw: bool = True) -> list[Path]:
    root = Path(results_root)
    pairs_path = root / "paired_transfer_gaps.csv"
    metrics_path = root / "all_metrics.csv"
    if not pairs_path.exists() or not metrics_path.exists():
        aggregate_results(root)
    pairs = _read_csv_or_empty(pairs_path)
    metrics = _read_csv_or_empty(metrics_path)
    if experiment is not None:
        if "experiment" in pairs:
            pairs = pairs[pairs["experiment"].astype(str) == experiment]
        if "experiment" in metrics:
            metrics = metrics[metrics["experiment"].astype(str) == experiment]
    # Primary architecture-matched gaps are the main figures; legacy gaps remain
    # in the CSV and can be plotted by filtering explicitly.
    if "baseline_kind" in pairs and (pairs["baseline_kind"] == "primary").any():
        pairs = pairs[pairs["baseline_kind"] == "primary"]
    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for x_column, stem in (
        ("n0", "transfer_improvement_vs_n0"),
        ("m_per_aux", "transfer_improvement_vs_auxiliary_sample_size"),
        ("K_aux", "transfer_improvement_vs_K_aux"),
        ("average_auxiliary_similarity", "transfer_improvement_vs_auxiliary_similarity"),
    ):
        outputs.extend(
            plot_transfer_improvement(
                pairs,
                figdir,
                x_column=x_column,
                filename_stem=stem,
                show_raw=show_raw,
            )
        )
    sample_root = root / EXP_DIR[experiment] if experiment is not None else root
    for output in (
        _plot_noise_bins(metrics, figdir),
        _plot_semantic_leakage(metrics, figdir),
        _plot_density_coverage(metrics, figdir),
        _plot_sample_grid(sample_root, figdir),
    ):
        if output is not None:
            outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--experiment", choices=["A", "B", "C"])
    parser.add_argument("--hide-raw", action="store_true")
    args = parser.parse_args()
    outputs = plot_results(args.results_root, experiment=args.experiment, show_raw=not args.hide_raw)
    print(f"wrote {len(outputs)} figures to {Path(args.results_root) / 'figures'}")


if __name__ == "__main__":
    main()
