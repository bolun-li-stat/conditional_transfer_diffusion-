from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import TensorDataset

from image_transfer.diffusion import ImageDDIM, ImageDDPM
from image_transfer.evaluation.classifier_fidelity import (
    evaluate_classifier_fidelity,
    imagenet_synset_to_index,
)
from image_transfer.evaluation.corruption_bank import (
    CorruptionBank,
    CorruptionRecord,
    create_corruption_bank,
    evaluate_corruption_bank,
    load_corruption_bank,
    save_corruption_bank,
    timestep_bin_name,
)
from image_transfer.evaluation.feature_metrics import (
    MetricBackendError,
    compute_feature_metrics,
    real_feature_cache_key,
)
from image_transfer.evaluation.nearest_neighbors import (
    compute_memorization_diagnostics,
    nearest_neighbor_search_from_features,
)
from image_transfer.models.unet import ImageUNet


class _ZeroEpsilon(nn.Module):
    def forward(self, x, t, y=None):
        return torch.zeros_like(x)


class _TinyUint8Extractor(nn.Module):
    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(x.float(), (2, 2)).flatten(1)


def test_conditional_and_unconditional_shared_initialization_matches():
    kwargs = dict(image_size=8, base_channels=8, channel_mults=(1,))
    torch.manual_seed(2026)
    unconditional = ImageUNet(**kwargs, num_classes=None)
    torch.manual_seed(2026)
    conditional = ImageUNet(**kwargs, num_classes=7)
    unconditional_state = unconditional.state_dict()
    conditional_state = conditional.state_dict()
    common = set(unconditional_state).intersection(conditional_state)
    assert common
    assert all(torch.equal(unconditional_state[key], conditional_state[key]) for key in common)
    assert set(conditional_state) - set(unconditional_state) == {"class_emb.weight"}


def test_ddpm_requires_full_steps_and_uses_explicit_generator():
    diffusion = ImageDDPM(timesteps=4, device="cpu")
    model = _ZeroEpsilon()
    with pytest.raises(ValueError, match="sampling steps == diffusion timesteps"):
        diffusion.sample(model, (2, 1, 4, 4), steps=2)
    first = diffusion.sample(
        model, (2, 1, 4, 4), steps=4, generator=torch.Generator().manual_seed(91)
    )
    # Consume the process-global RNG; an explicit generator must remain isolated.
    _ = torch.randn(100)
    second = diffusion.sample(
        model, (2, 1, 4, 4), steps=4, generator=torch.Generator().manual_seed(91)
    )
    assert torch.equal(first, second)


def test_ddim_respacing_eta_and_determinism():
    diffusion = ImageDDIM(timesteps=6, eta=0.0, device="cpu")
    model = _ZeroEpsilon()
    with pytest.raises(ValueError, match=r"\[2, 6\]"):
        diffusion.sample(model, (2, 1, 4, 4), steps=1)
    with pytest.raises(ValueError):
        diffusion.sample(model, (2, 1, 4, 4), steps=7)
    assert diffusion._sampling_sequence(3).tolist() == [5, 2, 0]
    first = diffusion.sample(
        model, (2, 1, 4, 4), steps=3, eta=0.5, generator=torch.Generator().manual_seed(22)
    )
    second = diffusion.sample(
        model, (2, 1, 4, 4), steps=3, eta=0.5, generator=torch.Generator().manual_seed(22)
    )
    assert torch.equal(first, second)


def test_corruption_bank_hash_roundtrip_and_manifest_key(tmp_path):
    kwargs = dict(
        evaluation_seed=17,
        timesteps=10,
        corruptions_per_image=4,
        num_images=5,
    )
    first = create_corruption_bank(manifest_hash="manifest-a", **kwargs)
    second = create_corruption_bank(manifest_hash="manifest-a", **kwargs)
    changed = create_corruption_bank(manifest_hash="manifest-b", **kwargs)
    assert first == second
    assert first.bank_hash == second.bank_hash
    assert first.bank_hash != changed.bank_hash
    path = save_corruption_bank(first, tmp_path / "bank.json")
    assert load_corruption_bank(path) == first


def _manual_bank() -> CorruptionBank:
    records = []
    timesteps = (0, 1, 4, 9)
    for image_index in range(4):
        for corruption_index, timestep in enumerate(timesteps):
            records.append(
                CorruptionRecord(
                    image_index=image_index,
                    timestep=timestep,
                    noise_seed=1000 + image_index * len(timesteps) + corruption_index,
                )
            )
    return CorruptionBank(
        manifest_hash="fixed-validation-manifest",
        evaluation_seed=5,
        timesteps=10,
        corruptions_per_image=4,
        noise_bins=(("low", 0.0, 0.2), ("mid", 0.2, 0.7), ("high", 0.7, 1.0)),
        records=tuple(records),
    )


def test_corruption_evaluation_is_batch_invariant_and_overall_is_uniform_weighted():
    bank = _manual_bank()
    dataset = TensorDataset(torch.zeros(4, 1, 3, 3), torch.zeros(4, dtype=torch.long))
    diffusion = ImageDDPM(timesteps=10, device="cpu")
    one = evaluate_corruption_bank(_ZeroEpsilon(), diffusion, dataset, bank, "cpu", batch_size=1)
    seven = evaluate_corruption_bank(_ZeroEpsilon(), diffusion, dataset, bank, "cpu", batch_size=7)
    for key in (
        "validation_epsilon_mse_target",
        "validation_epsilon_mse_low_noise",
        "validation_epsilon_mse_mid_noise",
        "validation_epsilon_mse_high_noise",
        "validation_epsilon_mse_standard_error",
    ):
        assert one[key] == pytest.approx(seven[key], rel=0, abs=0)
    counts = {"low": 0, "mid": 0, "high": 0}
    for record in bank.records:
        counts[timestep_bin_name(bank, record.timestep)] += 1
    weighted = sum(
        one[f"validation_epsilon_mse_{name}_noise"] * count for name, count in counts.items()
    ) / bank.num_corruptions
    equal_bin_average = sum(one[f"validation_epsilon_mse_{name}_noise"] for name in counts) / 3
    assert one["validation_epsilon_mse_target"] == pytest.approx(weighted)
    # Low has twice the finite-bank mass of each other bin, so this guards the
    # former incorrect equal-bin averaging path.
    assert not math.isclose(weighted, equal_bin_average, rel_tol=0, abs_tol=1e-12)
    assert one["num_validation_images"] == 4
    assert one["num_corruptions"] == 16


def test_debug_metric_never_writes_fid_or_kid(tmp_path):
    generated = torch.zeros(4, 3, 8, 8)
    real = torch.ones(4, 3, 8, 8)
    result = compute_feature_metrics(
        generated,
        real,
        mode="debug",
        real_manifest_hash="manifest-debug",
        cache_dir=tmp_path,
    )
    assert "debug_pooled_pixel_distance" in result
    assert "fid_target" not in result
    assert "kid_target_mean" not in result
    cached = torch.load(result["real_feature_cache_path"], map_location="cpu")
    assert cached["features"].shape[0] == len(real)


def test_paper_mode_backend_failure_raises_without_fallback(monkeypatch):
    import image_transfer.evaluation.feature_metrics as feature_metrics

    def fail(*args, **kwargs):
        raise MetricBackendError("backend missing")

    monkeypatch.setattr(feature_metrics, "_build_torchmetrics_extractor", fail)
    with pytest.raises(MetricBackendError, match="backend missing"):
        compute_feature_metrics(
            torch.zeros(4, 3, 8, 8),
            torch.zeros(4, 3, 8, 8),
            mode="paper",
            real_manifest_hash="manifest-paper",
        )


def test_cache_key_changes_with_manifest():
    kwargs = dict(
        feature_extractor_id="extractor-v1",
        preprocessing_config={"range": [-1, 1]},
        image_size=32,
    )
    assert real_feature_cache_key(manifest_hash="a", **kwargs) != real_feature_cache_key(
        manifest_hash="b", **kwargs
    )


def test_unified_paper_metrics_with_injected_offline_extractor(tmp_path):
    real = torch.linspace(-1, 1, 6 * 3 * 4 * 4).view(6, 3, 4, 4)
    generated = real.clone()
    result = compute_feature_metrics(
        generated,
        real,
        mode="paper",
        real_manifest_hash="manifest-paper",
        cache_dir=tmp_path,
        feature_extractor=_TinyUint8Extractor(),
        feature_extractor_id="tiny-offline-test-v1",
        kid_subset_size=4,
        kid_num_subsets=3,
        prdc_k=2,
        feature_batch_size=2,
        distance_batch_size=2,
    )
    assert result["fid_target"] == pytest.approx(0.0, abs=1e-7)
    assert result["kid_target_status"] == "ok"
    assert result["prdc_status"] == "ok"
    assert result["precision_target"] == pytest.approx(1.0)
    cached = torch.load(result["real_feature_cache_path"], map_location="cpu")
    assert {"features", "mean", "cov", "cache_key"}.issubset(cached)


def test_classifier_mapping_is_exact_and_cifar_is_explicitly_unavailable():
    assert imagenet_synset_to_index("n02108915") == 245
    assert imagenet_synset_to_index("French bulldog") is None
    result = evaluate_classifier_fidelity(
        torch.zeros(2, 3, 8, 8), "dog", ["cat"], dataset_name="cifar10"
    )
    assert math.isnan(result["classifier_target_top1_acc"])
    assert "CIFAR-10 classifier" in result["classifier_unavailable_reason"]

    class FixedClassifier(torch.nn.Module):
        def forward(self, images):
            logits = torch.zeros(images.shape[0], 1000, device=images.device)
            logits[:, 245] = 1.0
            return logits

    imagenet64 = evaluate_classifier_fidelity(
        torch.zeros(2, 3, 8, 8),
        "n02108915",
        [],
        dataset_name="imagenet64",
        classifier=FixedClassifier(),
        preprocess=torch.nn.Identity(),
        strict=True,
    )
    assert imagenet64["classifier_dataset"] == "imagenet"
    assert imagenet64["classifier_target_top1_acc"] == 1.0


def test_nearest_neighbor_search_is_batched_and_four_reference_stats_are_reported():
    queries = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    references = torch.tensor([[0.0, 1.0], [3.0, 0.0], [9.0, 9.0]])
    distances, indices = nearest_neighbor_search_from_features(
        queries,
        references,
        k=1,
        query_batch_size=1,
        reference_batch_size=1,
    )
    assert indices[:, 0].tolist() == [0, 1]
    assert distances[:, 0].tolist() == pytest.approx([1.0, 1.0])

    generated = torch.tensor([[[[0.0, 0.0]]], [[[1.0, 1.0]]]])
    references_by_name = {
        "target_train": generated.clone(),
        "target_eval": generated + 0.25,
        "auxiliary_train": generated + 0.5,
        "auxiliary_eval": generated + 0.75,
    }
    diagnostics = compute_memorization_diagnostics(
        generated,
        references_by_name,
        near_duplicate_threshold=0.01,
        extractor=nn.Flatten(),
        feature_batch_size=1,
        distance_batch_size=1,
        reference_batch_size=1,
    )
    for name in references_by_name:
        assert diagnostics[f"nearest_neighbor_{name}_status"] == "ok"
        assert f"nearest_neighbor_{name}_q05" in diagnostics
    assert diagnostics["nearest_neighbor_target_train_near_duplicate_rate"] == 1.0
    assert "nearest_neighbor_target_train_minus_eval_mean" in diagnostics
