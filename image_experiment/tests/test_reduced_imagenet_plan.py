from __future__ import annotations

import hashlib
import copy
from collections import Counter
from pathlib import Path

import yaml

from image_transfer.config import load_resolved_config
from image_transfer.data.class_sets import IMAGENET_CLASSES
from image_transfer.data.manifests import canonical_sha256
from image_transfer.scripts.make_job_grid import rows_for_experiment


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "image_transfer" / "configs"
TARGET_DIR = CONFIG_DIR / "targets"

CONFIGS = {
    "release": "imagenet64_release_pilot.yaml",
    "main": "imagenet64_reduced_main.yaml",
    "primary": "imagenet64_reduced_primary_protocol.yaml",
    "fixed_k": "imagenet64_reduced_fixed_budget_k.yaml",
}

TARGET_HASHES = {
    "reduced_main_4targets_v1.yaml": "810687600fe5f38335cba3f2efcde2a8ce6ad1cbcb44d6f9f77c85cc79b50dd8",
    "french_bulldog_primary_fixed3_v1.yaml": "e8d9c6065efc43fe3a33fc33f3e6d1e21bc3787b83725c0d241c9605e5307204",
    "french_bulldog_k_sensitivity_v1.yaml": "c27837d50db6180b29a5cb47e10098dd3d56e45f81380a5d3b94ebd61b837e8f",
}


def _payload(name: str) -> dict:
    return yaml.safe_load((TARGET_DIR / name).read_text(encoding="utf-8"))


def _rows(name: str) -> list[dict]:
    source = CONFIG_DIR / CONFIGS[name]
    resolved = load_resolved_config(source)
    return rows_for_experiment(
        "A",
        resolved.raw,
        str(source),
        resolved_info=resolved,
        override_readiness_gate=True,
    )


def test_reduced_main_target_metadata_is_frozen_disjoint_and_complete():
    payload = _payload("reduced_main_4targets_v1.yaml")
    targets = payload["targets"]
    expected = {
        "n02108915": {
            "close": ["n02096585", "n02110958", "n02108089"],
            "medium": ["n02123045", "n02119022", "n02114367"],
            "far": ["n04146614", "n03028079", "n04398044"],
        },
        "n01580077": {
            "close": ["n01537544", "n01582220", "n01592084"],
            "medium": ["n01518878", "n01614925", "n01860187"],
            "far": ["n04146614", "n04398044", "n07734744"],
        },
        "n03594945": {
            "close": ["n02814533", "n03770679", "n03930630"],
            "medium": ["n02690373", "n02951358", "n04347754"],
            "far": ["n01443537", "n04398044", "n03028079"],
        },
        "n02676566": {
            "close": ["n03272010", "n02787622", "n02992211"],
            "medium": ["n02672831", "n03394916", "n03249569"],
            "far": ["n01443537", "n04146614", "n07734744"],
        },
    }

    assert payload["target_set_id"] == "reduced_main_4targets_v1"
    assert payload["target_set_version"] == "1.0"
    assert payload["reviewed"] is True
    assert payload["frozen"] is True
    assert payload["reviewer"] == "Bolun Li"
    assert payload["review_date"] == "2026-08-16"
    assert len(targets) == 4
    assert len({target["synset"] for target in targets}) == 4
    assert len({target["supercategory"] for target in targets}) == 4

    target_synsets = {target["synset"] for target in targets}
    all_auxiliary = set()
    for target in targets:
        groups = target["auxiliary_sets"]
        assert groups == expected[target["synset"]]
        assert set(groups) == {"close", "medium", "far"}
        for values in groups.values():
            assert len(values) == 3
            assert len(set(values)) == 3
            all_auxiliary.update(values)
        assert set(groups["close"]).isdisjoint(groups["medium"])
        assert set(groups["close"]).isdisjoint(groups["far"])
        assert set(groups["medium"]).isdisjoint(groups["far"])

    assert target_synsets.isdisjoint(all_auxiliary)


def test_single_target_files_are_frozen_and_have_declared_pool_sizes():
    primary = _payload("french_bulldog_primary_fixed3_v1.yaml")
    fixed_k = _payload("french_bulldog_k_sensitivity_v1.yaml")

    for payload in (primary, fixed_k):
        assert payload["reviewed"] is True
        assert payload["frozen"] is True
        assert payload["reviewer"] == "Bolun Li"
        assert payload["review_date"] == "2026-08-16"
        assert len(payload["targets"]) == 1

    primary_groups = primary["targets"][0]["auxiliary_sets"]
    assert {name: len(values) for name, values in primary_groups.items()} == {
        "close": 3,
        "medium": 3,
        "far": 3,
    }
    assert primary_groups == {
        "close": ["n02096585", "n02110958", "n02108089"],
        "medium": ["n02123045", "n02119022", "n02114367"],
        "far": ["n04146614", "n03028079", "n04398044"],
    }

    fixed_groups = fixed_k["targets"][0]["auxiliary_sets"]
    assert {name: len(values) for name, values in fixed_groups.items()} == {
        "close": 6,
        "medium": 5,
        "far": 5,
    }
    assert fixed_groups == {
        "close": [
            "n02096585", "n02110958", "n02108089", "n02085620", "n02091032", "n02099601"
        ],
        "medium": ["n02123045", "n02123159", "n02124075", "n02119022", "n02114367"],
        "far": ["n03594945", "n04146614", "n03028079", "n03457902", "n03710193"],
    }
    assert all(len(values) == len(set(values)) for values in fixed_groups.values())
    assert set(fixed_groups["close"]).isdisjoint(fixed_groups["medium"])
    assert set(fixed_groups["close"]).isdisjoint(fixed_groups["far"])
    assert set(fixed_groups["medium"]).isdisjoint(fixed_groups["far"])


def test_target_set_hashes_are_stable_and_order_sensitive():
    for filename, expected_hash in TARGET_HASHES.items():
        source = next(
            CONFIG_DIR / config
            for config in CONFIGS.values()
            if load_resolved_config(CONFIG_DIR / config).raw.get("target_set_path")
            == f"targets/{filename}"
        )
        assert load_resolved_config(source).target_set_hash == expected_hash

    payload = _payload("reduced_main_4targets_v1.yaml")
    reordered = copy.deepcopy(payload)
    reordered["targets"][0]["auxiliary_sets"]["close"].reverse()
    assert canonical_sha256(payload) != canonical_sha256(reordered)


def test_reduced_grid_counts_protocols_and_medium_not_run():
    grids = {name: _rows(name) for name in CONFIGS}
    assert {name: len(rows) for name, rows in grids.items()} == {
        "release": 12,
        "main": 180,
        "primary": 120,
        "fixed_k": 51,
    }
    assert sum(map(len, grids.values())) == 363

    expected_protocols = {
        "release": {"natural_compute_matched": 6, "target_exposure_matched": 6},
        "main": {"natural_compute_matched": 180},
        "primary": {"natural_compute_matched": 60, "target_exposure_matched": 60},
        "fixed_k": {"natural_compute_matched": 51},
    }
    assert {
        name: dict(Counter(row["training_protocol"] for row in rows))
        for name, rows in grids.items()
    } == expected_protocols
    assert Counter(
        row["training_protocol"] for rows in grids.values() for row in rows
    ) == {"natural_compute_matched": 297, "target_exposure_matched": 66}

    for name in ("main", "primary", "fixed_k"):
        assert {row["aux_set"] for row in grids[name]} == {"none", "close", "far"}
        assert all(row["aux_set"] != "medium" for row in grids[name])


def test_reduced_configs_preserve_declared_training_and_sampling_settings():
    expected_paths = {
        "main": "targets/reduced_main_4targets_v1.yaml",
        "primary": "targets/french_bulldog_primary_fixed3_v1.yaml",
        "fixed_k": "targets/french_bulldog_k_sensitivity_v1.yaml",
    }
    for name, target_path in expected_paths.items():
        resolved = load_resolved_config(CONFIG_DIR / CONFIGS[name])
        raw = resolved.raw
        assert raw["target_set_path"] == target_path
        assert raw["training"]["steps"] == 20000
        assert raw["sampling"]["sampler"] == "ddim"
        assert raw["sampling"]["steps"] == 50
        assert raw["num_generated"] == 512
        assert raw["experiments"]["A"]["aux_sets"] == ["close", "far"]
        assert raw["experiments"]["B"]["enabled"] is False
        assert raw["experiments"]["C"]["enabled"] is False
        assert raw["evaluation"]["mode"] == "strict"


def test_fixed_budget_k_grid_uses_only_real_unique_combinations():
    rows = _rows("fixed_k")
    baselines = [row for row in rows if row["aux_set"] == "none"]
    candidates = [row for row in rows if row["aux_set"] != "none"]

    assert len(baselines) == 3
    assert {row["training_subset_seed"] for row in baselines} == {0, 1, 2}
    assert {(row["K_aux"], row["m_per_aux"]) for row in candidates} == {
        (1, 300),
        (3, 100),
        (5, 60),
    }
    assert {row["total_auxiliary_budget"] for row in candidates} == {300}
    assert {row["design_label"] for row in candidates} == {"fixed_total_300"}
    assert len(
        {
            (row["aux_set"], row["K_aux"], row["aux_draw_id"])
            for row in candidates
        }
    ) == 16

    close_k5 = [row for row in candidates if row["aux_set"] == "close" and row["K_aux"] == 5]
    far_k5 = [row for row in candidates if row["aux_set"] == "far" and row["K_aux"] == 5]
    assert len({row["aux_composition"] for row in close_k5}) == 3
    assert len({row["aux_composition"] for row in far_k5}) == 1
    assert {row["aux_unique_combinations"] for row in far_k5} == {1}
    assert all(row["aux_set"] != "medium" for row in candidates)


def test_release_pilot_files_and_grid_remain_unchanged():
    release = CONFIG_DIR / "imagenet64_release_pilot.yaml"
    pilot_target = TARGET_DIR / "french_bulldog_pilot.yaml"
    assert hashlib.sha256(release.read_bytes()).hexdigest() == (
        "ce25e0ea883d584e1a99212e2b9af91512b8803d8899412ba61e3ba1a390eb7e"
    )
    assert hashlib.sha256(pilot_target.read_bytes()).hexdigest() == (
        "4f6a1918f9b09a9959f29a2b1cacd916de92aa37d819c66477e585af8f683f28"
    )
    assert len(_rows("release")) == 12


def test_all_declared_reduced_classes_have_readable_names():
    required = set()
    for filename in TARGET_HASHES:
        payload = _payload(filename)
        for target in payload["targets"]:
            required.add(target["synset"])
            for values in target["auxiliary_sets"].values():
                required.update(values)
    assert required <= set(IMAGENET_CLASSES)
    assert all(IMAGENET_CLASSES[synset] != synset for synset in required)
    assert {synset: IMAGENET_CLASSES[synset] for synset in required} == {
        "n01443537": "goldfish",
        "n01518878": "ostrich",
        "n01537544": "indigo bunting",
        "n01580077": "jay",
        "n01582220": "magpie",
        "n01592084": "chickadee",
        "n01614925": "bald eagle",
        "n01860187": "black swan",
        "n02085620": "Chihuahua",
        "n02091032": "Italian greyhound",
        "n02096585": "Boston bull",
        "n02099601": "golden retriever",
        "n02108089": "boxer",
        "n02108915": "French bulldog",
        "n02110958": "pug",
        "n02114367": "timber wolf",
        "n02119022": "red fox",
        "n02123045": "tabby cat",
        "n02123159": "tiger cat",
        "n02124075": "Egyptian cat",
        "n02672831": "accordion",
        "n02676566": "acoustic guitar",
        "n02690373": "airliner",
        "n02787622": "banjo",
        "n02814533": "station wagon",
        "n02951358": "canoe",
        "n02992211": "cello",
        "n03028079": "church",
        "n03249569": "drum",
        "n03272010": "electric guitar",
        "n03394916": "French horn",
        "n03457902": "greenhouse",
        "n03594945": "jeep",
        "n03710193": "mailbox",
        "n03770679": "minivan",
        "n03930630": "pickup truck",
        "n04146614": "school bus",
        "n04347754": "submarine",
        "n04398044": "teapot",
        "n07734744": "mushroom",
    }
