from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from image_transfer.models import (
    ADMAttentionBlock,
    ADMResBlock,
    build_image_model,
    model_config_hash,
    model_parameter_metadata,
    resolve_model_config,
)
from image_transfer.training.checkpointing import load_checkpoint, save_checkpoint


def _tiny(*, image_size=32, conditional=False, num_classes=3, seed=7):
    return build_image_model(
        {"architecture": "adm_unet", "profile": "smoke_tiny"},
        image_size=image_size,
        conditional=conditional,
        num_classes=num_classes,
        model_seed=seed,
    )


@pytest.mark.parametrize("image_size", [32, 64])
def test_adm_unet_output_shape_32_and_64(image_size):
    model = _tiny(image_size=image_size)
    output = model(torch.randn(1, 3, image_size, image_size), torch.tensor([1]))
    assert output.shape == (1, 3, image_size, image_size)


def test_conditional_requires_labels_and_one_multi_class_forward():
    x, t = torch.randn(2, 3, 32, 32), torch.tensor([0, 1])
    with pytest.raises(ValueError, match="requires labels"):
        _tiny(conditional=True, num_classes=1)(x, t)
    assert _tiny(conditional=True, num_classes=1)(x, t, torch.zeros(2, dtype=torch.long)).shape == x.shape
    assert _tiny(conditional=True, num_classes=4)(x, t, torch.tensor([1, 3])).shape == x.shape


def test_unconditional_accepts_no_labels_and_rejects_labels():
    model = _tiny()
    x, t = torch.randn(1, 3, 32, 32), torch.tensor([0])
    assert model(x, t).shape == x.shape
    with pytest.raises(ValueError, match="does not accept labels"):
        model(x, t, torch.tensor([0]))


def test_shared_initialization_exact_match_and_separate_class_rng():
    models = [
        _tiny(conditional=False, seed=19),
        _tiny(conditional=True, num_classes=1, seed=19),
        _tiny(conditional=True, num_classes=5, seed=19),
    ]
    states = [model.state_dict() for model in models]
    shared = set(states[0]) & set(states[1]) & set(states[2])
    assert shared and all(not key.startswith("class_embedding.") for key in shared)
    for key in shared:
        assert torch.equal(states[0][key], states[1][key])
        assert torch.equal(states[0][key], states[2][key])

    _tiny(conditional=False, seed=23)
    after_unconditional = torch.rand(5)
    _tiny(conditional=True, num_classes=9, seed=23)
    after_conditional = torch.rand(5)
    assert torch.equal(after_unconditional, after_conditional)


@pytest.mark.parametrize(
    "profile,lower,upper",
    [
        ("smoke_tiny", 0, 2_000_000),
        ("pilot_small", 8_000_000, 15_000_000),
        ("main_default", 25_000_000, 40_000_000),
        ("capacity_large", 50_000_000, 80_000_000),
    ],
)
def test_parameter_count_profiles(profile, lower, upper):
    size = 32 if profile == "smoke_tiny" else 64
    count = model_parameter_metadata(
        build_image_model(
            {"architecture": "adm_unet", "profile": profile},
            image_size=size,
            conditional=False,
            num_classes=1,
            model_seed=0,
        )
    )["model_parameter_count"]
    assert lower < count < upper


def test_attention_head_and_resolution_validation():
    with pytest.raises(ValueError, match="divisible"):
        ADMAttentionBlock(48, num_head_channels=32)
    with pytest.raises(ValueError, match="do not occur"):
        resolve_model_config(
            {"architecture": "adm_unet", "profile": "smoke_tiny", "attention_resolutions": [7]},
            image_size=32,
        )


def test_zero_initialized_output_and_skip_shapes_match_without_repair():
    model = _tiny(image_size=32)
    x = torch.randn(2, 3, 32, 32)
    output = model(x, torch.tensor([1, 2]))
    assert torch.count_nonzero(output) == 0
    assert not hasattr(model, "interpolate")


def test_resblock_scale_shift_up_down_and_backward_finite():
    embedding = torch.randn(2, 32)
    x = torch.randn(2, 8, 8, 8, requires_grad=True)
    scale_shift = ADMResBlock(8, 32, out_channels=16, zero_init_residual=False)
    output = scale_shift(x, embedding)
    assert output.shape == (2, 16, 8, 8)
    assert scale_shift.embedding_projection[-1].out_features == 32
    up = ADMResBlock(8, 32, up=True)(torch.randn(2, 8, 8, 8), embedding)
    down = ADMResBlock(8, 32, down=True)(torch.randn(2, 8, 8, 8), embedding)
    assert up.shape[-2:] == (16, 16)
    assert down.shape[-2:] == (4, 4)
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_checkpoint_roundtrip_and_architecture_mismatch_fails(tmp_path):
    model = _tiny(conditional=True, num_classes=2)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, model_metadata=model_parameter_metadata(model))
    clone = _tiny(conditional=True, num_classes=2)
    load_checkpoint(path, clone)
    for key, value in model.state_dict().items():
        assert torch.equal(value, clone.state_dict()[key])
    legacy = build_image_model(
        {"architecture": "legacy_simple_unet", "base_channels": 8, "channel_mults": [1]},
        image_size=32,
        conditional=True,
        num_classes=2,
        model_seed=7,
    )
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(path, legacy)


def test_legacy_model_and_config_mapping_and_stable_hash():
    with pytest.warns(DeprecationWarning):
        resolved = resolve_model_config({"base_channels": 8, "channel_mults": [1]}, image_size=32)
    assert resolved["architecture"] == "legacy_simple_unet"
    legacy = build_image_model(resolved, image_size=32, conditional=False, num_classes=1, model_seed=0)
    assert legacy(torch.randn(1, 3, 32, 32), torch.tensor([1])).shape == (1, 3, 32, 32)
    first = resolve_model_config({"architecture": "adm_unet", "profile": "main_default"}, image_size=64)
    second = copy.deepcopy(first)
    assert model_config_hash(first) == model_config_hash(second)


def test_unknown_field_fails_and_profile_is_not_data_dependent():
    with pytest.raises(ValueError, match="unknown ADM"):
        resolve_model_config({"architecture": "adm_unet", "profile": "smoke_tiny", "n0": 50}, image_size=32)
    first = resolve_model_config({"architecture": "adm_unet", "profile": "main_default"}, image_size=64)
    # n0 and aux_set are intentionally absent from the factory interface/config.
    assert "n0" not in first and "aux_set" not in first


def test_tiny_overfit_loss_decreases():
    torch.manual_seed(5)
    model = build_image_model(
        {
            "architecture": "adm_unet",
            "profile": "smoke_tiny",
            "model_channels": 8,
            "channel_mults": [1],
            "num_head_channels": 8,
        },
        image_size=16,
        conditional=False,
        num_classes=1,
        model_seed=4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    x = torch.randn(4, 3, 16, 16)
    target = torch.tanh(x)
    losses = []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x, torch.zeros(4, dtype=torch.long)), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < 0.8 * losses[0]
