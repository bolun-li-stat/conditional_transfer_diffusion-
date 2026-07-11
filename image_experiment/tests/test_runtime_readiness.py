from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from image_transfer.config import load_resolved_config
from image_transfer.environment_lock import (
    build_exact_environment_lock,
    compare_exact_environment_lock,
    installed_pip_packages,
    validate_exact_environment_lock,
)
from image_transfer.scripts.inspect_environment import inspect_environment, validate_environment_report
from image_transfer.training import trainer
from image_transfer.training.checkpointing import _torch_load, _atomic_torch_save
from image_transfer.training.runtime_probe import (
    maximum_configured_num_classes,
    run_load_probe,
    run_resume_roundtrip,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, x, t, y=None):
        return self.weight * x


class _TinyDiffusion:
    def __init__(self, **kwargs) -> None:
        pass

    def loss(self, model, x0, y=None, *, reduction="mean"):
        values = (model(x0, torch.zeros(len(x0), dtype=torch.long), y) - 0.5).square()
        return values if reduction == "none" else values.mean()


def _validation(*args, **kwargs):
    return {"all": 1.0, "low": 1.0, "mid": 1.0, "high": 1.0}


def _train(tmp_path, monkeypatch, *, resume: bool = False):
    monkeypatch.setattr(trainer, "ImageUNet", lambda **kwargs: _TinyModel())
    monkeypatch.setattr(trainer, "ImageDDPM", _TinyDiffusion)
    data = TensorDataset(torch.ones(4, 1, 2, 2), torch.tensor([0, 0, 1, 1]))
    return trainer.train_image_model(
        data,
        data,
        conditional=True,
        num_classes=2,
        image_size=2,
        base_channels=2,
        channel_mults=[1],
        timesteps=2,
        schedule="linear",
        steps=1,
        batch_size=4,
        lr=1e-3,
        device="cpu",
        checkpoint_path=tmp_path / "run.pt",
        validation_interval=1,
        model_initialization_seed=1,
        training_seed=2,
        validation_evaluator=_validation,
        deterministic_cpu=True,
        config_hash="config",
        manifest_hash="manifest",
        resume=resume,
    )


def test_exact_lock_rejects_unexpected_packages_and_payload_tampering():
    lock = build_exact_environment_lock(
        source_spec_hash="a" * 64,
        conda_packages=[],
        pip_packages=[{"name": "example", "version": "1.0"}],
    )
    assert compare_exact_environment_lock(
        lock,
        pip_packages=[
            {"name": "example", "version": "1.0"},
            {"name": "unexpected", "version": "2.0"},
        ],
        conda_packages=[],
    ) == ["unexpected pip package unexpected==2.0"]
    lock["pip_packages"][0]["version"] = "changed"
    with pytest.raises(ValueError, match="payload hash"):
        validate_exact_environment_lock(lock)


def test_installed_package_inventory_preserves_sys_path_precedence(monkeypatch):
    class Distribution:
        def __init__(self, name: str, version: str) -> None:
            self.metadata = {"Name": name}
            self.version = version

    monkeypatch.setattr(
        "image_transfer.environment_lock.importlib.metadata.distributions",
        lambda: [Distribution("Example_Pkg", "2.0"), Distribution("example-pkg", "1.0")],
    )
    assert installed_pip_packages() == [{"name": "example-pkg", "version": "2.0"}]


@pytest.mark.parametrize(
    "protocol", ["natural_compute_matched", "target_exposure_matched"]
)
def test_resume_roundtrip_explicitly_validates_all_state(protocol, tmp_path):
    config = load_resolved_config(
        "image_transfer/configs/cifar10_fake_smoke.yaml"
    ).resolved
    result = run_resume_roundtrip(
        config,
        device="cpu",
        protocol=protocol,
        work_dir=tmp_path / protocol,
    )
    for field in (
        "raw_model_state_restored",
        "ema_model_state_restored",
        "optimizer_state_restored",
        "grad_scaler_state_restored",
        "raw_not_loaded_from_ema",
        "global_step_continuous",
        "sampler_position_continuous",
        "exposure_counters_restored",
        "exposure_counters_continuous",
        "loss_sequence_matches",
        "passed",
    ):
        assert result[field] is True
    assert (result["global_step_restored"], result["global_step_after_continue"]) == (1, 2)
    assert (result["sampler_position_restored"], result["sampler_position_after_continue"]) == (1, 2)


@pytest.mark.parametrize(
    "protocol,expected_auxiliary",
    [("natural_compute_matched", 0), ("target_exposure_matched", 4)],
)
def test_configured_load_probe_records_finite_training_step(protocol, expected_auxiliary):
    config = load_resolved_config(
        "image_transfer/configs/cifar10_fake_smoke.yaml"
    ).resolved
    result = run_load_probe(config, device="cpu", protocol=protocol)
    assert result["target_batch_size"] == 4
    assert result["auxiliary_batch_size"] == expected_auxiliary
    assert result["optimizer"] == "adamw"
    assert result["ema_updated"] is True
    assert result["gradient_clipping_configured"] is True
    assert result["step_finite"] is True


def test_environment_report_double_hash_detects_tampering(tmp_path):
    import importlib.metadata

    lock = tmp_path / "requirements.txt"
    lock.write_text(f"torch=={importlib.metadata.version('torch')}\n", encoding="utf-8")
    report = inspect_environment(lock)
    validate_environment_report(report)
    report["gpu_names"] = ["changed"]
    with pytest.raises(ValueError, match="runtime hash"):
        validate_environment_report(report)


def test_terminal_checkpoint_can_finish_result_without_another_step(tmp_path, monkeypatch):
    _, _, first = _train(tmp_path, monkeypatch)
    _, _, resumed = _train(tmp_path, monkeypatch, resume=True)
    for field in (
        "final_objective_train_loss",
        "final_pooled_train_loss",
        "final_target_batch_train_loss",
        "validation_epsilon_mse_target",
        "optimizer_steps",
    ):
        assert resumed[field] == first[field]


def test_terminal_checkpoint_without_terminal_state_fails_loudly(tmp_path, monkeypatch):
    _train(tmp_path, monkeypatch)
    path = tmp_path / "run_last.pt"
    checkpoint = _torch_load(path)
    checkpoint["training_protocol_metadata"].pop("terminal_training_state")
    _atomic_torch_save(checkpoint, path)
    with pytest.raises(RuntimeError, match="lacks terminal training state"):
        _train(tmp_path, monkeypatch, resume=True)


def test_runtime_probe_class_count_covers_auxiliary_size_settings():
    config = {
        "K_aux_values": [2],
        "auxiliary_size_settings": [{"K_aux": 8}],
        "targets": [{"auxiliary_sets": {"close": ["a", "b", "c"]}}],
    }
    assert maximum_configured_num_classes(config) == 9


def test_runtime_probe_class_count_uses_declared_k_not_candidate_pool_size():
    config = {
        "K_aux_values": [5],
        "targets": [
            {
                "auxiliary_sets": {
                    "close": ["a", "b", "c", "d", "e", "f"],
                    "far": ["g", "h", "i", "j", "k", "l"],
                }
            }
        ],
    }
    assert maximum_configured_num_classes(config) == 6


def test_runtime_probe_class_count_falls_back_to_candidate_pool_for_legacy_config():
    config = {
        "targets": [
            {"auxiliary_sets": {"close": ["a", "b", "c"], "far": ["d", "e"]}}
        ]
    }
    assert maximum_configured_num_classes(config) == 4
