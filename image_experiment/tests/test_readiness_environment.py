from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket

import pytest
import torch
import yaml

from image_transfer.config.config_schema import load_resolved_config, resolve_config
from image_transfer.evaluation.corruption_bank import create_corruption_bank, save_corruption_bank
from image_transfer.environment_lock import build_exact_environment_lock
from image_transfer.readiness import enforce_readiness_gate
from image_transfer.scripts.inspect_environment import inspect_environment
from image_transfer.utils.io import canonical_json_hash
from image_transfer.scripts.make_job_grid import JOB_FIELDS, compute_resolved_run_spec_hash
from image_transfer.scripts import prepare_metric_assets as asset_tools
from image_transfer.scripts.prepare_metric_assets import (
    REQUIRED_ASSET_PATHS,
    build_manifest,
    initialize_backends_offline,
    verify_manifest,
)
from image_transfer.scripts.validate_release_pilot import (
    _read_and_compare_grid,
    rebuild_expected_jobs,
    validate_pilot,
)
from image_transfer.utils.io import write_run_result


def _write_analysis_plan(path: Path) -> Path:
    payload = {
        "analysis_plan_id": "unit",
        "primary": {
            "protocol": "natural_compute_matched",
            "comparisons": [
                "conditional_close_vs_conditional_target_only",
                "conditional_far_vs_conditional_target_only",
            ],
            "endpoints": ["test_epsilon_mse_target", "kid_target_mean"],
        },
        "secondary": {"comparisons": [], "endpoints": [], "studies": []},
        "transfer_gap_convention": {
            "lower_is_better": "baseline_minus_model",
            "higher_is_better": "model_minus_baseline",
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _minimal_config(tmp_path: Path) -> dict:
    lock = tmp_path / "lock.txt"
    lock.write_text("example==1.0\n", encoding="utf-8")
    analysis_plan = _write_analysis_plan(tmp_path / "analysis.yaml")
    return {
        "dataset": "cifar10",
        "use_fake_data": True,
        "image_size": 16,
        "study_stage": "smoke",
        "targets": [{"name": "dog", "synset": "dog", "auxiliary_sets": {"close": ["cat"]}}],
        "data_split": {"holdout_seed": 1, "training_subset_seed": 2},
        "model": {
            "architecture": "adm_unet",
            "profile": "smoke_tiny",
        },
        "diffusion": {"timesteps": 4, "schedule": "linear"},
        "optimizer": {"name": "adamw", "lr": 0.001},
        "training": {"steps": 2, "batch_size": 2},
        "sampling": {"sampler": "ddpm", "steps": 4},
        "evaluation": {"mode": "debug"},
        "experiments": {"A": {"enabled": True, "aux_sets": ["close"]}},
        "environment_lock_path": str(lock),
        "analysis_plan_path": str(analysis_plan),
    }


def test_legacy_mode_alias_resolves_to_strict_with_warning(tmp_path):
    raw = _minimal_config(tmp_path)
    raw["evaluation"]["mode"] = "paper"
    with pytest.warns(DeprecationWarning):
        resolved = resolve_config(raw)
    assert resolved.resolved["evaluation"]["mode"] == "strict"


def test_conflicting_alias_and_unknown_fields_fail(tmp_path):
    raw = _minimal_config(tmp_path)
    raw["sampling_steps"] = 3
    with pytest.raises(ValueError, match="Conflicting sampling_steps"):
        resolve_config(raw)
    raw = _minimal_config(tmp_path)
    raw["mystery"] = True
    with pytest.raises(ValueError, match="Unknown top-level"):
        resolve_config(raw)


def test_resolved_hash_stable_and_contains_provenance(tmp_path):
    first = resolve_config(_minimal_config(tmp_path))
    second = resolve_config(_minimal_config(tmp_path))
    assert first.resolved_hash == second.resolved_hash
    assert first.model_hash == second.model_hash
    assert first.environment_lock_hash == hashlib.sha256(b"example==1.0\n").hexdigest()


def test_main_stage_requires_fixed_adm_profile_and_frozen_multi_target_set(tmp_path):
    analysis = tmp_path / "analysis.yaml"
    analysis.write_text(
        Path("image_transfer/configs/analysis_plan.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    target_path = tmp_path / "targets.yaml"
    target_payload = {
        "target_set_id": "unit-main",
        "target_set_version": "1",
        "reviewed": True,
        "frozen": True,
        "targets": [
            {
                "name": f"target-{index}",
                "synset": f"target-{index}",
                "supercategory": "group-a" if index < 2 else "group-b",
                "selection_rationale": "predeclared unit fixture",
                "auxiliary_sets": {"close": ["a"], "medium": ["b"], "far": ["c"]},
            }
            for index in range(4)
        ],
    }
    target_path.write_text(yaml.safe_dump(target_payload), encoding="utf-8")
    raw = _minimal_config(tmp_path)
    raw.pop("targets")
    raw.update(
        {
            "study_stage": "main",
            "target_set_path": target_path.name,
            "analysis_plan_path": analysis.name,
            "model": {"architecture": "adm_unet", "profile": "main_default"},
        }
    )
    resolved = resolve_config(raw, source_path=tmp_path / "main.yaml")
    assert resolved.resolved["model"]["architecture"] == "adm_unet"
    assert resolved.resolved["model"]["profile"] == "main_default"

    missing_architecture = dict(raw, model={"profile": "main_default"})
    with pytest.raises(ValueError, match="explicit model.architecture"):
        resolve_config(missing_architecture, source_path=tmp_path / "main.yaml")
    overridden = dict(raw, model={"architecture": "adm_unet", "profile": "main_default", "dropout": 0.2})
    with pytest.raises(ValueError, match="profiles are fixed"):
        resolve_config(overridden, source_path=tmp_path / "main.yaml")
    target_payload["frozen"] = False
    target_path.write_text(yaml.safe_dump(target_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed and frozen"):
        resolve_config(raw, source_path=tmp_path / "main.yaml")


def test_analysis_plan_validation_rejects_missing_primary_endpoint(tmp_path):
    raw = _minimal_config(tmp_path)
    invalid_plan = tmp_path / "analysis.yaml"
    invalid_plan.write_text(
        yaml.safe_dump(
            {
                "analysis_plan_id": "invalid",
                "primary": {
                    "protocol": "natural_compute_matched",
                    "comparisons": [
                        "conditional_close_vs_conditional_target_only",
                        "conditional_far_vs_conditional_target_only",
                    ],
                    "endpoints": ["test_epsilon_mse_target"],
                },
                "secondary": {},
                "transfer_gap_convention": {
                    "lower_is_better": "baseline_minus_model",
                    "higher_is_better": "model_minus_baseline",
                },
            }
        ),
        encoding="utf-8",
    )
    raw["evaluation"]["mode"] = "strict"
    raw["analysis_plan_path"] = invalid_plan.name
    with pytest.raises(ValueError, match="primary endpoint"):
        resolve_config(raw, source_path=tmp_path / "config.yaml")


def test_case_study_records_single_target_scope(tmp_path):
    raw = _minimal_config(tmp_path)
    raw["study_stage"] = "case_study"
    resolved = resolve_config(raw, source_path=tmp_path / "case.yaml")
    assert resolved.resolved["single_target_scope"] is True


def test_relative_input_paths_do_not_make_config_hash_checkout_dependent(tmp_path):
    raw = _minimal_config(tmp_path)
    raw.pop("targets")
    raw["study_stage"] = "pilot"
    raw["model"] = {"architecture": "adm_unet", "profile": "smoke_tiny"}
    raw["target_set_path"] = "targets/set.yaml"
    raw["analysis_plan_path"] = "analysis.yaml"
    raw["environment_lock_path"] = "lock.txt"
    target_set = {
        "target_set_id": "unit",
        "target_set_version": "1",
        "reviewed": False,
        "frozen": False,
        "targets": [{
            "name": "target",
            "synset": "target",
            "supercategory": "unit",
            "selection_rationale": "deterministic test fixture",
            "auxiliary_sets": {"close": ["a"], "medium": ["b"], "far": ["c"]},
        }],
    }
    resolved = []
    for directory_name in ("first", "second"):
        directory = tmp_path / directory_name
        (directory / "targets").mkdir(parents=True)
        (directory / "targets" / "set.yaml").write_text(json.dumps(target_set), encoding="utf-8")
        _write_analysis_plan(directory / "analysis.yaml")
        (directory / "lock.txt").write_text("example==1.0\n", encoding="utf-8")
        resolved.append(resolve_config(raw, source_path=directory / "config.yaml"))
    assert resolved[0].resolved_hash == resolved[1].resolved_hash
    assert resolved[0].resolved["target_set_path"] == "targets/set.yaml"


def test_environment_report_hashes_lock(tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_text("x==1\n", encoding="utf-8")
    report = inspect_environment(lock)
    assert report["environment_lock_hash"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert "torch" in report["packages"]
    assert report["lock_matches_runtime"] is False
    assert report["lock_mismatches"]
    assert report["environment_runtime_hash"] == inspect_environment(lock)["environment_runtime_hash"]


def test_offline_asset_check_rejects_missing_and_modified_files(tmp_path):
    root = tmp_path / "assets"
    path = root / "custom-assets.json"
    for index, relative in enumerate(REQUIRED_ASSET_PATHS.values()):
        asset = root / relative
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(f"weights-{index}".encode())
    manifest = build_manifest(root, path)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_manifest(root, path)["files"]
    assert all(entry["path"] != path.name for entry in manifest["assets"])
    asset = root / next(iter(REQUIRED_ASSET_PATHS.values()))
    asset.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="verification failed"):
        verify_manifest(root, path)


def test_asset_manifest_rejects_tampering_unsafe_paths_and_bad_roots(tmp_path):
    root = tmp_path / "assets"
    for relative in REQUIRED_ASSET_PATHS.values():
        asset = root / relative
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(b"weights")
    manifest_path = root / "manifest.json"
    manifest = build_manifest(root, manifest_path)
    manifest["assets"][0]["path"] = "../outside.bin"
    manifest["files"] = manifest["assets"]
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsafe asset manifest path"):
        verify_manifest(root, manifest_path)
    with pytest.raises(ValueError, match="dedicated directory"):
        build_manifest(Path.cwd(), tmp_path / "elsewhere.json")


def test_asset_runtime_verification_and_offline_initialization(tmp_path, monkeypatch):
    root = tmp_path / "assets"
    for relative in REQUIRED_ASSET_PATHS.values():
        asset = root / relative
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(b"weights")
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(build_manifest(root, manifest_path)), encoding="utf-8")
    monkeypatch.setattr(
        asset_tools,
        "_versions",
        lambda: {name: "different" for name in asset_tools.RUNTIME_PACKAGES},
    )
    with pytest.raises(RuntimeError, match="runtime-version"):
        verify_manifest(root, manifest_path, verify_runtime=True)

    calls = []

    def fake_feature(**kwargs):
        calls.append(("feature", kwargs))
        connection = socket.socket()
        try:
            with pytest.raises(RuntimeError, match="network access is disabled"):
                connection.connect(("127.0.0.1", 9))
        finally:
            connection.close()
        return {"status": "ok"}

    def fake_classifier(*args, **kwargs):
        calls.append(("classifier", (args, kwargs)))
        return {"status": "ok"}

    monkeypatch.setattr(asset_tools, "preflight_feature_metric_backend", fake_feature)
    monkeypatch.setattr(asset_tools, "preflight_classifier_fidelity", fake_classifier)
    report = initialize_backends_offline(root)
    assert report["feature_backend"]["status"] == "ok"
    assert [name for name, _ in calls] == ["feature", "classifier"]


def test_environment_pin_matching_and_gpu_config_identity(tmp_path):
    version = importlib.metadata.version("torch")
    lock = tmp_path / "requirements.txt"
    lock.write_text(f"torch=={version}\n", encoding="utf-8")
    report = inspect_environment(lock)
    assert report["lock_matches_runtime"] is True
    assert report["lock_expected_versions"] == {"torch": version}

    root = Path(__file__).resolve().parents[1]
    cuda_definition = root / "environment" / "image-transfer-cuda.yml"
    for name in ("imagenet64_gpu_smoke.yaml", "imagenet64_release_pilot.yaml", "imagenet64_main_template.yaml"):
        config = yaml.safe_load((root / "image_transfer" / "configs" / name).read_text(encoding="utf-8"))
        assert config["environment_lock_path"].endswith("image-transfer-cuda.yml")
        assert config["exact_environment_lock_path"] == "${EXACT_ENVIRONMENT_LOCK_PATH:-}"
        assert config["environment_runtime_report_path"] == "${ENVIRONMENT_RUNTIME_REPORT_PATH:-}"
        assert config["gpu_runtime_probe_path"] == "${GPU_RUNTIME_PROBE_PATH:-}"
        assert config["resume_probe_path"] == "${RESUME_PROBE_PATH:-}"
        assert config["metric_assets_manifest_path"] == "${METRIC_ASSETS_MANIFEST:-}"
    assert cuda_definition.is_file()
    cuda_report = inspect_environment(cuda_definition)
    assert not any(mismatch.startswith("python:") for mismatch in cuda_report["lock_mismatches"])


def test_not_run_status_blocks_main_and_override_is_recorded(tmp_path):
    pilot_path = tmp_path / "pilot.yaml"
    pilot_config = _minimal_config(tmp_path)
    pilot_config["study_stage"] = "pilot"
    pilot_config["model"] = {"architecture": "adm_unet", "profile": "smoke_tiny"}
    pilot_path.write_text(json.dumps(pilot_config), encoding="utf-8")
    pilot = load_resolved_config(pilot_path)
    status = tmp_path / "pilot.json"
    status.write_text(json.dumps({"status": "not_run"}), encoding="utf-8")
    config = {
        "study_stage": "main",
        "readiness_status_path": str(status),
        "readiness_pilot_config_path": str(pilot_path),
        "model": pilot.resolved["model"],
        "environment_lock_hash": pilot.environment_lock_hash,
        "study_plan_hash": pilot.study_plan_hash,
        # A main target set is intentionally different from the single-target
        # engineering pilot and is not compared directly by this gate.
        "target_set_hash": "different-main-targets",
    }
    with pytest.raises(RuntimeError, match="readiness gate failed"):
        enforce_readiness_gate(config, current_git_sha="a" * 40)
    outcome = enforce_readiness_gate(config, override=True, current_git_sha="a" * 40)
    assert outcome["override"] is True
    assert outcome["status"] == "not_run"


def test_passed_gate_links_exact_pilot_and_runtime_identity(tmp_path):
    pilot_path = tmp_path / "pilot.yaml"
    pilot_config = _minimal_config(tmp_path)
    pilot_config["study_stage"] = "pilot"
    pilot_config["model"] = {"architecture": "adm_unet", "profile": "smoke_tiny"}
    pilot_path.write_text(json.dumps(pilot_config), encoding="utf-8")
    pilot = load_resolved_config(pilot_path)
    git_sha = "b" * 40
    status_path = tmp_path / "status.json"
    status = {
        "schema_version": "3.0",
        "status": "passed",
        "git_sha": git_sha,
        "model_config_hash": pilot.model_hash,
        "target_set_hash": pilot.target_set_hash,
        "environment_lock_hash": pilot.environment_lock_hash,
        "study_plan_hash": pilot.study_plan_hash,
        "pilot_config_hash": pilot.resolved_hash,
        "expected_jobs": 1,
        "validated_jobs": 1,
        "resume_state_validated_jobs": 1,
        "checkpoint_artifacts_validated_jobs": 1,
        "sample_artifacts_validated_jobs": 1,
        "metric_artifacts_validated_jobs": 1,
        "nearest_neighbor_artifacts_validated_jobs": 1,
        "figure_artifacts_validated_jobs": 1,
        "provenance_artifacts_validated_jobs": 1,
        "validated_pairs": 1,
        "validated_result_hashes": {"run": "1" * 64},
        "expected_job_grid_hash": "2" * 64,
        "jobs_csv_hash": "3" * 64,
        "environment_runtime_hash": "4" * 64,
        "environment_report_hash": "5" * 64,
        "exact_environment_lock_hash": "8" * 64,
        "gpu_load_probe_hash": "6" * 64,
        "resume_probe_hash": "7" * 64,
        "failures": [],
    }
    status_path.write_text(json.dumps(status), encoding="utf-8")
    main = {
        "study_stage": "main",
        "readiness_status_path": status_path.name,
        "readiness_pilot_config_path": pilot_path.name,
        "model": pilot.resolved["model"],
        "environment_lock_hash": pilot.environment_lock_hash,
        "study_plan_hash": pilot.study_plan_hash,
        "target_set_hash": "intentionally-different",
    }
    outcome = enforce_readiness_gate(main, current_git_sha=git_sha, config_source_path=tmp_path / "main.yaml")
    assert outcome["passed"] is True
    assert outcome["mismatches"] == []
    with pytest.raises(RuntimeError, match="git_sha"):
        enforce_readiness_gate(main, current_git_sha="c" * 40, config_source_path=tmp_path / "main.yaml")


def test_release_validator_rebuilds_grid_and_rejects_altered_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path / "results"))
    config_path = Path("image_transfer/configs/imagenet64_release_pilot.yaml")
    resolved = load_resolved_config(config_path)
    expected = rebuild_expected_jobs(config_path, resolved)
    assert len(expected) == 12

    jobs_path = tmp_path / "jobs.csv"

    def write_rows(rows):
        with jobs_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=JOB_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    write_rows(expected)
    _, failures = _read_and_compare_grid(jobs_path, expected)
    assert failures == []
    report = validate_pilot(config_path, jobs_path, tmp_path / "results")
    assert report["status"] == "failed"
    assert report["expected_jobs"] == report["supplied_jobs"] == 12
    assert sum(item.startswith("missing_result:") for item in report["failures"]) == 12

    altered = [dict(row) for row in expected]
    altered[0]["sampling_steps"] = int(altered[0]["sampling_steps"]) + 1
    altered.append(dict(expected[1]))
    write_rows(altered)
    _, failures = _read_and_compare_grid(jobs_path, expected)
    assert any(item.startswith("job_grid_altered:") for item in failures)
    assert any(item.startswith("job_grid_duplicate:") for item in failures)


def test_readiness_audit_fields_change_run_identity():
    base = {field: "value" for field in (
        "dataset", "experiment", "target_synset", "model_type", "aux_set", "aux_composition",
        "aux_draw_id", "training_protocol", "sampler", "architecture", "architecture_profile",
        "model_config_hash", "resolved_config_hash", "study_plan_hash", "target_set_hash",
        "environment_lock_hash", "split_manifest_key", "subset_manifest_key",
    )}
    base.update({
        "n0": 1, "m_per_aux": 0, "K_aux": 0, "total_auxiliary_budget": 0,
        "holdout_seed": 1, "training_subset_seed": 2, "model_initialization_seed": 3,
        "training_seed": 4, "sampling_seed": 5, "evaluation_seed": 6, "sampling_steps": 7,
        "image_size": 8, "training_steps": 9, "num_generated": 10, "ddim_eta": 0.0,
        "readiness_gate_required": True, "readiness_gate_override": False,
        "readiness_gate_status": "passed", "readiness_gate_passed": True,
        "readiness_gate_mismatches": "[]", "readiness_status_file_hash": "status",
        "readiness_pilot_config_hash": "pilot", "readiness_validated_git_sha": "a" * 40,
        "readiness_current_git_sha": "a" * 40,
    })
    changed = dict(base, readiness_gate_override=True)
    assert compute_resolved_run_spec_hash(base) != compute_resolved_run_spec_hash(changed)


def test_release_validator_accepts_exact_complete_fixture(tmp_path, monkeypatch):
    lock = tmp_path / "lock.txt"
    lock.write_text("example==1.0\n", encoding="utf-8")
    exact_lock = tmp_path / "exact-lock.json"
    exact_lock.write_text(
        json.dumps(
            build_exact_environment_lock(
                source_spec_path=lock,
                source_spec_hash=hashlib.sha256(lock.read_bytes()).hexdigest(),
            )
        ),
        encoding="utf-8",
    )
    analysis_plan = _write_analysis_plan(tmp_path / "analysis.yaml")
    results_root = tmp_path / "results"
    analysis_path = tmp_path / "analysis.yaml"
    analysis_path.write_text(
        Path("image_transfer/configs/analysis_plan.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = {
        "dataset": "unit-images",
        "study_stage": "pilot",
        "use_fake_data": False,
        "data_root": str(tmp_path / "data"),
        "output_root": str(results_root),
        "image_size": 8,
        "targets": [{
            "name": "target",
            "synset": "target",
            "auxiliary_sets": {"close": ["a"], "medium": ["m"], "far": ["b"]},
        }],
        "environment_lock_path": str(lock),
        "exact_environment_lock_path": str(exact_lock),
        "dataset_identity_path": str(tmp_path / "identity.json"),
        "analysis_plan_path": str(analysis_path),
        "analysis_plan_path": str(analysis_plan),
        "data_split": {
            "holdout_seed": 1,
            "training_subset_seed": 2,
            "target_eval_size": 2,
            "target_val_size": 1,
            "auxiliary_eval_size": 1,
        },
        "seeds": [3],
        "model": {"architecture": "adm_unet", "profile": "smoke_tiny"},
        "diffusion": {"timesteps": 4, "schedule": "linear"},
        "training": {"steps": 2, "batch_size": 2, "protocol": "natural_compute_matched"},
        "sampling": {"sampler": "ddpm", "steps": 4},
        "num_generated": 2,
        "n0_values": [2],
        "K_aux_values": [1],
        "auxiliary_ratio_values": [1.0],
        "include_legacy_unconditional": False,
        "experiments": {"A": {"enabled": True, "aux_sets": ["close", "far"]}},
        "evaluation": {
            "mode": "strict",
            "validation_corruptions_per_image": 1,
            "test_corruptions_per_image": 1,
            "train_diagnostic_corruptions_per_image": 1,
            "train_diagnostic_max_images": 2,
            "compute_fid": False,
            "compute_kid": False,
            "compute_prdc": False,
            "compute_classifier_fidelity": False,
            "compute_feature_similarity": False,
            "make_nearest_neighbors": False,
        },
    }
    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    resolved = load_resolved_config(config_path)
    dataset_identity = {
        "dataset_identity_hash": "8" * 64,
        "dataset_content_hash": "9" * 64,
    }
    monkeypatch.setattr(
        "image_transfer.scripts.validate_release_pilot.verify_dataset_identity_file",
        lambda *args, **kwargs: dataset_identity,
    )
    environment_report = inspect_environment(exact_lock, source_spec_path=lock)
    environment_report["lock_matches_runtime"] = True
    environment_report["lock_mismatches"] = []
    environment_report["environment_report_hash"] = canonical_json_hash(
        {key: value for key, value in environment_report.items() if key != "environment_report_hash"}
    )
    monkeypatch.setattr(
        "image_transfer.scripts.validate_release_pilot._validate_external_runtime_evidence",
        lambda *args, **kwargs: {
            "environment_runtime_hash": environment_report["environment_runtime_hash"],
            "environment_report_hash": environment_report["environment_report_hash"],
            "exact_environment_lock_hash": environment_report["exact_environment_lock_hash"],
            "gpu_load_probe_hash": "a" * 64,
            "resume_probe_hash": "b" * 64,
        },
    )
    rows = rebuild_expected_jobs(config_path, resolved)
    assert len(rows) == 3
    jobs_path = tmp_path / "jobs.csv"
    with jobs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=JOB_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    git_sha = "d" * 40
    shared_hashes = {
        "manifest_hash": "combined-manifest",
        "split_manifest_hash": "1" * 64,
        "subset_manifest_hash": "2" * 64,
        "target_training_subset_hash": "3" * 64,
        "paired_target_prefix_hash": "3" * 64,
        "target_eval_indices_hash": "4" * 64,
        "target_validation_indices_hash": "5" * 64,
    }
    split_path = tmp_path / f"target__{'1' * 16}.split.json"
    subset_path = tmp_path / f"target__{'2' * 16}.subset.json"
    split_path.write_text(json.dumps({"split_manifest_hash": "1" * 64}), encoding="utf-8")
    subset_path.write_text(json.dumps({"subset_manifest_hash": "2" * 64}), encoding="utf-8")
    banks = {}
    for split_name, count, manifest_hash in (
        ("validation", 1, "1" * 64),
        ("test", 2, "1" * 64),
        ("train", 2, "train-diagnostic"),
    ):
        bank = create_corruption_bank(
            manifest_hash=manifest_hash,
            evaluation_seed=3,
            timesteps=4,
            corruptions_per_image=1,
            num_images=count,
            split=split_name,
        )
        bank_path = tmp_path / f"{split_name}-{bank.bank_hash}.json"
        save_corruption_bank(bank, bank_path)
        banks[split_name] = (bank, bank_path)

    for row in rows:
        run_id = row["run_id"]
        output_dir = Path(row["output_dir"])
        paths = {
            "last_checkpoint_path": output_dir / "checkpoints" / f"{run_id}_last.pt",
            "best_checkpoint_path": output_dir / "checkpoints" / f"{run_id}_best.pt",
            "sample_path": output_dir / "samples" / f"{run_id}_samples.pt",
            "run_config_path": output_dir / "configs" / f"{run_id}.yaml",
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "checkpoint_schema_version": 3,
            "step": 2,
            "config_hash": row["resolved_config_hash"],
            "manifest_hash": shared_hashes["manifest_hash"],
            "git_sha": git_sha,
            "training_protocol": row["training_protocol"],
            "model_metadata": {"model_config_hash": row["model_config_hash"]},
            "raw_model_state": {"weight": torch.ones(1)},
            "ema_model_state": {"weight": torch.ones(1)},
            "optimizer_state": {"state": {}, "param_groups": [{}]},
            "grad_scaler_state": {},
            "rng_states": {"python": (), "numpy": (), "torch_cpu": torch.zeros(1), "torch_cuda": []},
            "data_state": {"sampler_seed": 3, "global_step": 2, "num_workers": 0},
            "training_protocol_metadata": {
                "optimizer_steps": 2,
                "target_examples_seen": 4,
                "auxiliary_examples_seen": 0,
                "total_examples_seen": 4,
            },
            "provenance": {
                "raw_config": resolved.raw,
                "resolved_config": resolved.resolved,
                "raw_config_hash": row["raw_config_hash"],
                "resolved_config_hash": row["resolved_config_hash"],
                "model_config_hash": row["model_config_hash"],
                "study_plan_hash": row["study_plan_hash"],
                "target_set_hash": row["target_set_hash"],
                "environment_lock_hash": row["environment_lock_hash"],
                "environment_report": {"environment_lock_hash": row["environment_lock_hash"]},
                "split_manifest": {"split_manifest_hash": shared_hashes["split_manifest_hash"]},
                "subset_manifest": {"subset_manifest_hash": shared_hashes["subset_manifest_hash"]},
                "split_manifest_hash": shared_hashes["split_manifest_hash"],
                "subset_manifest_hash": shared_hashes["subset_manifest_hash"],
                "run_id": run_id,
            },
        }
        torch.save(checkpoint, paths["last_checkpoint_path"])
        torch.save(checkpoint, paths["best_checkpoint_path"])
        torch.save(torch.linspace(-1.0, 1.0, 384).reshape(2, 3, 8, 8), paths["sample_path"])
        paths["run_config_path"].write_text(
            yaml.safe_dump({
                "run_id": run_id,
                "resolved_config_hash": row["resolved_config_hash"],
                "model_config_hash": row["model_config_hash"],
            }),
            encoding="utf-8",
        )
        baseline = row["model_type"] == "conditional_target_only_n0"
        target_seen, auxiliary_seen = ((4, 0) if baseline else (2, 2))
        job = {
            field: row[field]
            for field in JOB_FIELDS
            if field not in {"config_path", "output_dir", "manifest_key"}
        }
        metadata = {
            **shared_hashes,
            **{field: row[field] for field in (
                "raw_config_hash", "resolved_config_hash", "study_plan_hash", "target_set_hash",
                "environment_lock_hash", "model_config_hash", "split_manifest_key", "subset_manifest_key",
                "resolved_run_spec_hash", "config_hash", "architecture", "architecture_profile",
            )},
            "git_sha": git_sha,
            **dataset_identity,
            "target_similarity_reference_hash": "6" * 64,
            "auxiliary_similarity_reference_hashes": {"a": "a" * 64, "m": "b" * 64, "b": "c" * 64},
            "environment_runtime_hash": environment_report["environment_runtime_hash"],
            "environment_report_hash": environment_report["environment_report_hash"],
            "environment_report": environment_report,
            "wallclock_total_seconds": 1.0,
            "peak_gpu_memory_bytes": 1,
            "selected_checkpoint_path": str(paths["best_checkpoint_path"]),
            "last_checkpoint_path": str(paths["last_checkpoint_path"]),
            "best_checkpoint_path": str(paths["best_checkpoint_path"]),
            "sample_path": str(paths["sample_path"]),
            "run_config_path": str(paths["run_config_path"]),
            "split_manifest_path": str(split_path),
            "subset_manifest_path": str(subset_path),
            "validation_corruption_bank_path": str(banks["validation"][1]),
            "test_corruption_bank_path": str(banks["test"][1]),
            "train_corruption_bank_path": str(banks["train"][1]),
        }
        selected_labels = json.loads(str(row.get("aux_composition", "[]")))
        metadata["selected_auxiliary_similarity_reference_hashes"] = {
            label: metadata["auxiliary_similarity_reference_hashes"][label]
            for label in selected_labels
        }
        metadata["similarity_metric_reference_hash"] = canonical_json_hash({
            "split": "dedicated_similarity_reference",
            "target": metadata["target_similarity_reference_hash"],
            "auxiliary": metadata["selected_auxiliary_similarity_reference_hashes"],
        })
        checkpoint["provenance"]["environment_runtime_hash"] = metadata["environment_runtime_hash"]
        checkpoint["provenance"]["environment_report_hash"] = metadata["environment_report_hash"]
        torch.save(checkpoint, paths["last_checkpoint_path"])
        torch.save(checkpoint, paths["best_checkpoint_path"])
        training = {
            "optimizer_steps": 2,
            "target_examples_seen": target_seen,
            "auxiliary_examples_seen": auxiliary_seen,
            "total_examples_seen": 4,
            "final_objective_train_loss": 1.0,
            "final_pooled_train_loss": 1.0,
            "final_target_batch_train_loss": 1.0,
            "final_auxiliary_batch_train_loss": None if baseline else 1.0,
            "rolling_pooled_train_loss": 1.0,
            "rolling_target_train_loss": 1.0,
            "rolling_auxiliary_train_loss": None if baseline else 1.0,
            "actual_target_batch_size": 2 if not baseline else 2,
            "actual_auxiliary_batch_size": 0 if baseline else 1,
            "validation_epsilon_mse_target": 1.0,
            "wallclock_train_seconds": 1.0,
            "images_processed_per_second": 1.0,
        }
        metrics = {
            "evaluation_mode": "strict",
            "metric_backend": "torchmetrics_inception_features",
            "metric_backend_version": "1.0",
            "feature_extractor_name": "inception-v3",
            "metric_implementation": "tested",
            "num_generated": 2,
            "num_real_eval": 2,
            "validation_epsilon_mse_target": 1.0,
            "test_epsilon_mse_target": 1.0,
            "num_validation_images": 1,
            "num_test_images": 2,
            "num_train_images": 2,
            "num_validation_corruptions": 1,
            "num_test_corruptions": 2,
            "num_train_corruptions": 2,
            "validation_corruption_bank_hash": banks["validation"][0].bank_hash,
            "test_corruption_bank_hash": banks["test"][0].bank_hash,
            "train_corruption_bank_hash": banks["train"][0].bank_hash,
        }
        write_run_result(
            results_root,
            run_id,
            {"status": "completed", "job": job, "metadata": metadata, "training": training, "metrics": metrics},
        )

    report = validate_pilot(config_path, jobs_path, results_root)
    assert report["status"] == "passed", report["failures"]
    assert report["git_sha"] == git_sha
    assert report["validated_jobs"] == 3
    assert report["validated_pairs"] == 2
