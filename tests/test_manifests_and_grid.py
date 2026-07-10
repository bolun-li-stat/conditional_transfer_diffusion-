from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_transfer.data.class_sets import draw_aux_synset_combinations, select_aux_synsets
from image_transfer.data.manifests import (
    ManifestInsufficientDataError,
    build_data_manifest,
    equal_total_feasibility,
    load_manifest,
    persist_or_validate_manifest,
    target_training_subset,
)
from image_transfer.scripts.make_job_grid import rows_for_experiment


def _manifest(**overrides):
    arguments = {
        "dataset_name": "unit-images",
        "target_class": "target",
        "data_split_seed": 17,
        "train_pools": {
            "target": list(range(20)),
            "near-a": list(range(100, 120)),
            "near-b": list(range(200, 220)),
        },
        "auxiliary_classes": ["near-a", "near-b"],
        "eval_source": "train_holdout",
        "target_eval_size": 4,
        "target_val_size": 3,
        "auxiliary_eval_size": 2,
        "experiment_family": "A",
        "mode": "strict",
    }
    arguments.update(overrides)
    return build_data_manifest(**arguments)


def test_manifest_is_disjoint_hashed_and_model_invariant():
    first = _manifest()
    paired_model = _manifest()  # model_type, aux_set, and n0 are intentionally not inputs
    assert first["manifest_hash"] == paired_model["manifest_hash"]
    target = first["target"]
    train = set(target["train_candidate_pool"])
    validation = set(target["validation"])
    evaluation = set(target["eval"])
    assert not train & validation
    assert not train & evaluation
    assert not validation & evaluation
    assert len(evaluation) == 4
    assert len(validation) == 3
    assert first["split_sizes"]["target_training_available"] == 13
    for pools in first["auxiliary"].values():
        assert not set(pools["train_candidate_pool"]) & set(pools["eval_candidate_pool"])


def test_nested_target_subsets_and_equal_total_feasibility_after_reservations():
    manifest = _manifest()
    small = target_training_subset(manifest, 3)
    medium = target_training_subset(manifest, 8)
    assert medium[: len(small)] == small
    assert equal_total_feasibility(manifest, n0=3, m_per_aux=2, k_aux=2)["feasible"]
    unavailable = equal_total_feasibility(manifest, n0=8, m_per_aux=3, k_aux=2)
    assert not unavailable["feasible"]
    assert unavailable["available_target_train_after_reservations"] == 13
    assert unavailable["shortfall"] == 1


def test_paper_manifest_never_shrinks_holdouts_silently():
    with pytest.raises(ManifestInsufficientDataError) as error:
        _manifest(target_eval_size=18, target_val_size=3)
    assert error.value.details == {
        "what": "target validation split",
        "requested": 3,
        "available": 2,
        "shortfall": 1,
    }
    debug = _manifest(target_eval_size=18, target_val_size=3, mode="debug")
    assert debug["split_sizes"]["actual_target_eval"] == 18
    assert debug["split_sizes"]["actual_target_validation"] == 2
    assert debug["feasibility_issues"][0]["shortfall"] == 1


def test_manifest_persistence_reuses_canonical_identity(tmp_path: Path):
    path = tmp_path / "manifest.json"
    manifest, written = persist_or_validate_manifest(_manifest(), path)
    second, second_path = persist_or_validate_manifest(_manifest(), path)
    assert written == second_path == path
    assert manifest["manifest_hash"] == second["manifest_hash"] == load_manifest(path)["manifest_hash"]
    serialized = json.loads(path.read_text(encoding="utf-8"))
    assert serialized["schema_version"] == "1.0"


def test_auxiliary_draws_are_reproducible_and_do_not_fake_replicates():
    groups = {"close": ["a", "b", "c"]}
    first, available = draw_aux_synset_combinations(groups, "close", 2, num_draws=3, aux_draw_seed=91)
    second, second_available = draw_aux_synset_combinations(groups, "close", 2, num_draws=3, aux_draw_seed=91)
    assert first == second
    assert available == second_available == 3
    assert len({tuple(draw) for draw in first}) == 3
    assert select_aux_synsets(groups, "close", 2, aux_draw_seed=91, aux_draw_id=1) == first[1]

    only_one, only_available = draw_aux_synset_combinations(
        {"close": ["a", "b"]}, "close", 2, num_draws=9, aux_draw_seed=3
    )
    assert only_available == 1
    assert only_one == [["a", "b"]]


def _grid_config() -> dict:
    return {
        "dataset": "unit-images",
        "output_root": "results",
        "targets": [{
            "name": "target",
            "synset": "target",
            "auxiliary_sets": {
                "close": ["a", "b", "c"],
                "medium": ["d", "e", "f"],
                "far": ["g", "h", "i"],
            },
        }],
        "n0_values": [4],
        "seeds": [7],
        "training": {"protocols": ["natural_compute_matched", "target_exposure_matched"]},
        "sampling": {"sampler": "ddim", "seed": 31},
        "evaluation": {"seed": 41},
        "data_split": {"data_split_seed": 11},
        "experiments": {
            "A": {
                "n0_values": [4],
                "aux_sets": ["close"],
                "m_per_aux_values": [2, 3],
                "K_aux_values": [1, 2],
                "num_aux_set_draws": 2,
                "aux_draw_seed": 101,
            },
            "B": {
                "n0_values": [4],
                "aux_sets": ["close"],
                "total_auxiliary_budget_values": [6],
                "K_aux_values": [1, 2, 3],
            },
        },
    }


def test_job_grid_has_full_ids_sweeps_and_no_aux_set_baseline_duplicates():
    cfg = _grid_config()
    rows = rows_for_experiment("A", cfg, "unit.yaml")
    # Four target-size settings each have two auxiliary draws per protocol, but
    # each target-only baseline is trained just once per protocol.
    assert len(rows) == 20
    assert len({row["run_id"] for row in rows}) == len(rows)
    candidates = [row for row in rows if row["aux_set"] != "none"]
    assert {(row["m_per_aux"], row["K_aux"]) for row in candidates} == {
        (2, 1), (2, 2), (3, 1), (3, 2)
    }
    baselines = [row for row in rows if row["aux_set"] == "none"]
    assert {(row["m_per_aux"], row["K_aux"]) for row in baselines} == {(0, 0)}
    assert len(baselines) == 4
    for protocol in {"natural_compute_matched", "target_exposure_matched"}:
        subset = [row for row in baselines if row["training_protocol"] == protocol]
        assert sum(row["model_type"] == "unconditional_n0" for row in subset) == 1
        assert sum(row["model_type"] == "conditional_target_only_n0" for row in subset) == 1
    for m_per_aux, k_aux in {(2, 1), (2, 2), (3, 1), (3, 2)}:
        for protocol in {"natural_compute_matched", "target_exposure_matched"}:
            subset = [
                row for row in candidates
                if row["m_per_aux"] == m_per_aux and row["K_aux"] == k_aux and row["training_protocol"] == protocol
            ]
            assert len(subset) == 2
    assert all(row["data_split_seed"] == 11 for row in rows)
    assert all(row["model_initialization_seed"] == 7 for row in rows)
    assert all(row["sampling_seed"] == 31 for row in rows)
    assert all(row["evaluation_seed"] == 41 for row in rows)
    assert all(row["training_protocol"] in row["run_id"] for row in rows)
    assert all(row["config_hash"][:12] in row["run_id"] for row in rows)


def test_equal_total_grid_supports_fixed_budget_varying_k():
    rows = rows_for_experiment("B", _grid_config(), "unit.yaml")
    candidates = [row for row in rows if row["aux_set"] != "none"]
    assert {(row["m_per_aux"], row["K_aux"], row["total_auxiliary_budget"]) for row in candidates} == {
        (6, 1, 6),
        (3, 2, 6),
        (2, 3, 6),
    }
    baselines = [row for row in rows if row["aux_set"] == "none"]
    assert len(baselines) == 4
    assert {(row["m_per_aux"], row["K_aux"], row["total_auxiliary_budget"]) for row in baselines} == {
        (6, 1, 6)
    }
    assert all(row["baseline_target_count"] == 10 for row in rows)
    for protocol in {"natural_compute_matched", "target_exposure_matched"}:
        subset = [row for row in baselines if row["training_protocol"] == protocol]
        assert sum(row["model_type"] == "unconditional_equal_total" for row in subset) == 1
        assert sum(row["model_type"] == "conditional_target_only_equal_total" for row in subset) == 1
    for m_per_aux, k_aux in {(6, 1), (3, 2), (2, 3)}:
        for protocol in {"natural_compute_matched", "target_exposure_matched"}:
            subset = [
                row for row in candidates
                if row["m_per_aux"] == m_per_aux and row["K_aux"] == k_aux and row["training_protocol"] == protocol
            ]
            assert len(subset) == 1

    invalid = _grid_config()
    invalid["experiments"]["B"]["total_auxiliary_budget_values"] = [5]
    invalid["experiments"]["B"]["K_aux_values"] = [2]
    with pytest.raises(ValueError, match="must be divisible"):
        rows_for_experiment("B", invalid, "unit.yaml")


def test_training_eval_views_reuse_manifest_indices_and_are_deterministic(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from image_transfer.data.builders import build_datasets_for_job

    cfg = {
        "dataset": "cifar10",
        "use_fake_data": True,
        "fake_data_size": 32,
        "image_size": 8,
        "output_root": str(tmp_path),
        "targets": [{
            "synset": "dog",
            "auxiliary_sets": {"close": ["cat"]},
        }],
        "data_split": {
            "manifest_root": str(tmp_path / "manifests"),
            "target_eval_size": 4,
            "target_val_size": 2,
            "auxiliary_eval_size": 2,
            "eval_source": "train_holdout",
        },
        "evaluation": {"mode": "debug"},
    }
    job = {
        "experiment": "A",
        "target_synset": "dog",
        "aux_composition": json.dumps(["cat"]),
        "data_split_seed": 9,
    }
    bundle = build_datasets_for_job(
        cfg,
        job,
        n0=3,
        m_per_aux=2,
        k_aux=1,
        seed=99,
        model_type="conditional_close",
    )

    assert bundle.target_train.dataset.indices == bundle.target_train_eval.dataset.indices
    assert set(bundle.auxiliary_train_datasets) == set(bundle.auxiliary_train_eval_by_class) == {"cat"}
    assert (
        bundle.auxiliary_train_datasets["cat"].dataset.indices
        == bundle.auxiliary_train_eval_by_class["cat"].dataset.indices
    )
    first_image, first_label = bundle.target_train_eval[0]
    second_image, second_label = bundle.target_train_eval[0]
    assert first_label == second_label == 0
    assert torch.equal(first_image, second_image)
