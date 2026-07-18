"""Reproduce exact tiny-CPU legacy regression checks across two worktrees.

This helper never enters the normal experiment pipeline. It creates small
temporary result directories inside each supplied worktree, compares the loaded
artifacts, prints a JSON summary, and removes only the directories it created.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch


RESULT_NAMES = {
    "original_low_target_data": "golden_regression_original_low",
    "original_same_total_budget": "golden_regression_original_budget",
    "rotation": "golden_regression_rotation",
}


def _run_original() -> None:
    sys.path.insert(0, str(Path.cwd()))
    from config import ExperimentConfig
    from train import run_single_setting

    def tiny(experiment: str, results: str) -> ExperimentConfig:
        cfg = ExperimentConfig(
            experiment_type=experiment, seed=0,
            n=4 if experiment == "same_total_budget" else None,
            n_target_train=None if experiment == "same_total_budget" else 4,
            n_aux_train=None if experiment == "same_total_budget" else 4,
            results_dir=Path(results))
        cfg.data.d = 4
        cfg.data.K = 3
        cfg.data.Delta = 2.0
        cfg.data.min_pairwise_mean_distance = 1.0
        cfg.diffusion.T = 4
        cfg.model.time_embedding_dim = 8
        cfg.model.class_embedding_dim = 4
        cfg.model.hidden_width = 16
        cfg.model.hidden_layers = 1
        cfg.training.device = "cpu"
        cfg.training.batch_size = 4
        cfg.training.training_steps = 2
        cfg.training.validation_interval = 1
        cfg.training.checkpoint_interval = 1
        cfg.evaluation.n_test_target = 8
        cfg.evaluation.n_generated = 4
        cfg.evaluation.score_risk_mc_samples = 8
        cfg.evaluation.mmd_max_samples = 8
        cfg.evaluation.save_samples = True
        return cfg

    run_single_setting(tiny(
        "low_target_data", RESULT_NAMES["original_low_target_data"]), force=True)
    run_single_setting(tiny(
        "same_total_budget", RESULT_NAMES["original_same_total_budget"]),
        force=True)


def _run_rotation() -> None:
    sys.path.insert(0, str(Path.cwd()))
    from config import smoke_config
    from train import run_setting

    cfg = smoke_config(Path(RESULT_NAMES["rotation"]))
    cfg.device = "cpu"
    run_setting(cfg, "target_only", force=True)
    run_setting(cfg, "joint_conditional", force=True)


def _assert_nested_equal(left: Any, right: Any, location: str) -> None:
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or not torch.equal(left, right):
            raise AssertionError(f"Tensor mismatch at {location}")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"Mapping mismatch at {location}")
        for key in left:
            _assert_nested_equal(left[key], right[key], f"{location}.{key}")
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f"Sequence mismatch at {location}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_nested_equal(
                left_item, right_item, f"{location}[{index}]")
    elif isinstance(left, float) and np.isnan(left):
        if not isinstance(right, float) or not np.isnan(right):
            raise AssertionError(f"NaN mismatch at {location}")
    elif left != right:
        raise AssertionError(f"Value mismatch at {location}: {left!r} != {right!r}")


def _identity_values(root: Path) -> dict[str, list[str]]:
    identities: dict[str, set[str]] = {
        name: set() for name in (
            "training_design_id", "design_id", "pair_id", "checkpoint_id",
            "setting_id", "checkpoint_path")
    }
    for csv_path in root.rglob("*.csv"):
        frame = pd.read_csv(csv_path)
        for name in identities:
            if name in frame:
                identities[name].update(frame[name].dropna().astype(str))
    checkpoint_root = root / "checkpoints"
    for path in checkpoint_root.rglob("*.pt"):
        if path.parent == checkpoint_root:
            identities["checkpoint_id"].add(path.stem)
        else:
            identities["setting_id"].add(path.parent.name)
    return {name: sorted(values) for name, values in identities.items() if values}


def _compare_result_trees(base: Path, modified: Path) -> dict[str, Any]:
    base_files = sorted(path.relative_to(base) for path in base.rglob("*")
                        if path.is_file())
    modified_files = sorted(path.relative_to(modified)
                            for path in modified.rglob("*") if path.is_file())
    if base_files != modified_files:
        raise AssertionError(
            f"Artifact file sets differ: {base_files!r} != {modified_files!r}")
    counts = {"checkpoints": 0, "csv_files": 0, "sample_arrays": 0,
              "config_files": 0, "other_files": 0}
    for relative in base_files:
        left, right = base / relative, modified / relative
        if relative.suffix == ".pt":
            _assert_nested_equal(
                torch.load(left, map_location="cpu"),
                torch.load(right, map_location="cpu"), str(relative))
            counts["checkpoints"] += 1
        elif relative.suffix == ".csv":
            pd.testing.assert_frame_equal(
                pd.read_csv(left), pd.read_csv(right), check_exact=True)
            counts["csv_files"] += 1
        elif relative.suffix == ".npy":
            if not np.array_equal(np.load(left), np.load(right)):
                raise AssertionError(f"Sample mismatch at {relative}")
            counts["sample_arrays"] += 1
        elif relative.suffix == ".json":
            if json.loads(left.read_text()) != json.loads(right.read_text()):
                raise AssertionError(f"Config mismatch at {relative}")
            counts["config_files"] += 1
        else:
            if left.read_bytes() != right.read_bytes():
                raise AssertionError(f"File mismatch at {relative}")
            counts["other_files"] += 1
    base_identities = _identity_values(base)
    modified_identities = _identity_values(modified)
    if base_identities != modified_identities:
        raise AssertionError("Legacy identity values changed")
    return {"status": "exact", "counts": counts,
            "identities": base_identities}


def _module_dir(worktree: Path, module: str) -> Path:
    suffix = ("original_ar1_experiment" if module == "original" else
              "spectrum_matched_rotation_experiment")
    return worktree / "gmm_simulation" / suffix


def _run_hidden(script: Path, worktree: Path, module: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(script), "--run-module", module],
        cwd=_module_dir(worktree, module), capture_output=True, text=True,
        check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Golden {module} run failed under {worktree}:\n"
            f"{completed.stdout}\n{completed.stderr}")


def compare_worktrees(base_worktree: Path, modified_worktree: Path) \
        -> dict[str, Any]:
    base_worktree = base_worktree.resolve()
    modified_worktree = modified_worktree.resolve()
    script = Path(__file__).resolve()
    targets = []
    for worktree in (base_worktree, modified_worktree):
        for key, name in RESULT_NAMES.items():
            module = "rotation" if key == "rotation" else "original"
            target = _module_dir(worktree, module) / name
            if target.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing golden directory: {target}")
            targets.append(target)
    try:
        for worktree in (base_worktree, modified_worktree):
            _run_hidden(script, worktree, "original")
            _run_hidden(script, worktree, "rotation")
        comparisons = {}
        for key, name in RESULT_NAMES.items():
            module = "rotation" if key == "rotation" else "original"
            comparisons[key] = _compare_result_trees(
                _module_dir(base_worktree, module) / name,
                _module_dir(modified_worktree, module) / name)
        return {"base_worktree": str(base_worktree),
                "modified_worktree": str(modified_worktree),
                "comparisons": comparisons, "overall_status": "exact"}
    finally:
        for target in targets:
            if target.exists():
                shutil.rmtree(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-worktree", type=Path)
    parser.add_argument("--modified-worktree", type=Path)
    parser.add_argument("--run-module", choices=["original", "rotation"],
                        help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run_module == "original":
        _run_original()
    elif args.run_module == "rotation":
        _run_rotation()
    else:
        if args.base_worktree is None or args.modified_worktree is None:
            raise SystemExit(
                "--base-worktree and --modified-worktree are required")
        print(json.dumps(compare_worktrees(
            args.base_worktree, args.modified_worktree), indent=2,
            sort_keys=True))
