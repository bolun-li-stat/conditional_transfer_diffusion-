from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from image_transfer.config import load_resolved_config, resolve_config
from image_transfer.scripts.make_job_grid import job_breakdown, rows_for_experiment


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "image_transfer" / "configs"


def _four_target_file(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    targets = []
    for index in range(4):
        targets.append(
            {
                "name": f"target-{index}",
                "synset": f"target-{index}",
                "supercategory": "animals" if index < 2 else "objects",
                "selection_rationale": "Fixed synthetic identity used only for grid-shape validation.",
                "auxiliary_sets": {
                    "close": [f"close-{index}-{value}" for value in range(6)],
                    "medium": [f"medium-{index}-{value}" for value in range(6)],
                    "far": [f"far-{index}-{value}" for value in range(6)],
                },
            }
        )
    destination = tmp_path / "reviewed_targets.yaml"
    destination.write_text(
        yaml.safe_dump(
            {
                "target_set_id": "grid-shape-test",
                "target_set_version": "1",
                "reviewed": True,
                "frozen": True,
                "reviewer": "test",
                "review_date": "2026-07-11",
                "targets": targets,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return destination


def _single_target_file(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_load(
        (CONFIG_DIR / "targets" / "french_bulldog_pilot.yaml").read_text(encoding="utf-8")
    )
    payload.update(
        {
            "target_set_id": "reviewed-primary-grid-shape-test",
            "target_set_version": "1",
            "reviewed": True,
            "frozen": True,
            "reviewer": "test",
            "review_date": "2026-07-11",
        }
    )
    destination = tmp_path / "reviewed_primary_target.yaml"
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def _resolved_main_design(name: str, tmp_path: Path):
    source = CONFIG_DIR / name
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    single_target_designs = {
        "imagenet64_optimization_stability.yaml",
        "imagenet64_k_sensitivity.yaml",
        "imagenet64_capacity_sensitivity.yaml",
    }
    raw["target_set_path"] = str(
        _single_target_file(tmp_path) if name in single_target_designs else _four_target_file(tmp_path)
    )
    return resolve_config(raw, source_path=source)


def _shape_rows(name: str, tmp_path: Path):
    source = CONFIG_DIR / name
    if name in {
        "imagenet64_main_template.yaml",
        "imagenet64_core_confirmatory.yaml",
        "imagenet64_optimization_stability.yaml",
        "imagenet64_k_sensitivity.yaml",
        "imagenet64_capacity_sensitivity.yaml",
    }:
        resolved = _resolved_main_design(name, tmp_path)
        return rows_for_experiment(
            "A",
            resolved.raw,
            str(source),
            allow_disabled=True,
            resolved_info=resolved,
            override_readiness_gate=True,
        )
    resolved = load_resolved_config(source)
    return rows_for_experiment(
        "A",
        resolved.raw,
        str(source),
        allow_disabled=True,
        resolved_info=resolved,
    )


@pytest.mark.parametrize(
    "name",
    [
        "imagenet64_optimization_stability.yaml",
        "imagenet64_k_sensitivity.yaml",
        "imagenet64_capacity_sensitivity.yaml",
    ],
)
def test_single_target_scientific_designs_are_blocked_until_target_review(name: str):
    with pytest.raises(ValueError, match="reviewed and frozen target set"):
        load_resolved_config(CONFIG_DIR / name)


@pytest.mark.parametrize(
    "name",
    [
        "imagenet64_optimization_stability.yaml",
        "imagenet64_k_sensitivity.yaml",
        "imagenet64_capacity_sensitivity.yaml",
    ],
)
def test_single_target_scientific_designs_require_readiness_after_review(
    name: str, tmp_path: Path
):
    source = CONFIG_DIR / name
    resolved = _resolved_main_design(name, tmp_path / name)
    with pytest.raises(RuntimeError, match="readiness gate failed"):
        rows_for_experiment(
            "A",
            resolved.raw,
            str(source),
            allow_disabled=True,
            resolved_info=resolved,
        )


def test_staged_design_job_counts_and_seed_roles(tmp_path: Path):
    expected = {
        "imagenet64_release_pilot.yaml": 12,
        "imagenet64_main_template.yaml": 320,
        "imagenet64_core_confirmatory.yaml": 480,
        "imagenet64_optimization_stability.yaml": 60,
        "imagenet64_k_sensitivity.yaml": 165,
        "imagenet64_capacity_sensitivity.yaml": 54,
    }
    grids = {name: _shape_rows(name, tmp_path / name) for name in expected}
    assert {name: len(rows) for name, rows in grids.items()} == expected

    main = grids["imagenet64_main_template.yaml"]
    assert {row["K_aux"] for row in main if row["aux_set"] != "none"} == {3}
    assert {row["training_subset_seed"] for row in main} == set(range(5))
    assert {row["training_protocol"] for row in main} == {"natural_compute_matched"}

    core = grids["imagenet64_core_confirmatory.yaml"]
    assert {row["training_subset_seed"] for row in core} == set(range(10))
    assert {row["training_protocol"] for row in core} == {
        "natural_compute_matched",
        "target_exposure_matched",
    }

    stability = grids["imagenet64_optimization_stability.yaml"]
    assert {row["training_subset_seed"] for row in stability} == set(range(5))
    assert {
        (row["model_initialization_seed"], row["training_seed"]) for row in stability
    } == {(0, 0), (1, 1)}


def test_k_sensitivity_preserves_budget_labels_and_unique_draws(tmp_path: Path):
    rows = _shape_rows("imagenet64_k_sensitivity.yaml", tmp_path)
    candidates = [row for row in rows if row["aux_set"] != "none"]
    assert {row["design_label"] for row in candidates} == {
        "equal_per_class",
        "fixed_total_300",
    }
    equal = [row for row in candidates if row["design_label"] == "equal_per_class"]
    fixed = [row for row in candidates if row["design_label"] == "fixed_total_300"]
    assert {(row["K_aux"], row["m_per_aux"], row["total_auxiliary_budget"]) for row in equal} == {
        (1, 100, 100),
        (3, 100, 300),
        (5, 100, 500),
    }
    assert {(row["K_aux"], row["m_per_aux"], row["total_auxiliary_budget"]) for row in fixed} == {
        (1, 300, 300),
        (3, 100, 300),
        (5, 60, 300),
    }
    assert len({row["run_id"] for row in rows}) == len(rows)
    far_k5 = [row for row in candidates if row["aux_set"] == "far" and row["K_aux"] == 5]
    assert far_k5
    assert {row["aux_unique_combinations"] for row in far_k5} == {1}
    assert len({row["aux_composition"] for row in far_k5}) == 1


def test_resource_report_includes_total_gpu_checkpoint_sample_and_storage(tmp_path: Path):
    rows = _shape_rows("imagenet64_release_pilot.yaml", tmp_path)
    resolved = load_resolved_config(CONFIG_DIR / "imagenet64_release_pilot.yaml")
    report = job_breakdown(rows, resolved.resolved)
    assert report["total_job_count"] == report["estimated_jobs"] == 12
    assert report["estimated_gpu_hours"] == 24.0
    assert report["estimated_checkpoints"] == 24
    assert report["estimated_generated_samples"] == 12 * 512
    assert report["estimated_checkpoint_storage_bytes"] > 0
    assert report["estimated_sample_storage_bytes"] > 0
    assert report["estimated_total_storage_bytes"] > report["estimated_checkpoint_storage_bytes"]
    assert report["breakdown"]["protocol"] == {
        "natural_compute_matched": 6,
        "target_exposure_matched": 6,
    }
