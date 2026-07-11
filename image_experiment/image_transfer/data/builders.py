from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from torch.utils.data import ConcatDataset, Dataset, Subset

from image_transfer.utils.io import resolve_env_path
from .cifar import BlockLabeledDataset, RemappedDataset, build_fake_data, class_id, load_cifar10
from .imagenet_subset import RemappedImageFolder, load_imagefolder, validate_synsets
from .manifests import (
    ManifestInsufficientDataError,
    auxiliary_training_subset,
    build_split_manifest,
    build_subset_manifest,
    canonical_sha256,
    combine_manifests,
    equal_total_feasibility,
    parse_sample_ref,
    persist_or_validate_manifest,
    resolve_manifest_seeds,
    split_manifest_path,
    subset_manifest_path,
    target_training_subset,
    validate_split_manifest,
    validate_subset_manifest,
)


EQUAL_TOTAL_MODEL_TYPES = {"unconditional_equal_total", "conditional_target_only_equal_total"}
TARGET_ONLY_MODEL_TYPES = {
    "unconditional_n0",
    "conditional_target_only_n0",
    "unconditional_equal_total",
    "conditional_target_only_equal_total",
}


@dataclass
class DatasetBundle:
    """Datasets and immutable split provenance for one training job.

    ``train`` and ``val`` retain the original API.  New training code should use
    ``target_train`` and ``auxiliary_train`` for exposure-matched sampling,
    ``target_train_eval``/``auxiliary_train_eval_by_class`` for deterministic
    memorization checks, and ``target_val`` for fixed-bank checkpoint selection.
    """

    train: Dataset
    val: Dataset
    target_eval: Dataset
    class_labels: list[str]
    aux_synsets: list[str]
    total_train_images: int
    num_target_available: int
    aux_eval_datasets: list[Dataset] | None = None
    skipped: bool = False
    skip_reason: str = ""
    target_train: Dataset | None = None
    target_train_eval: Dataset | None = None
    auxiliary_train: Dataset | None = None
    target_val: Dataset | None = None
    auxiliary_train_datasets: dict[str, Dataset] = field(default_factory=dict)
    auxiliary_train_eval_by_class: dict[str, Dataset] = field(default_factory=dict)
    auxiliary_eval_by_class: dict[str, Dataset] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""
    manifest_path: str = ""
    split_manifest: dict[str, Any] = field(default_factory=dict)
    subset_manifest: dict[str, Any] = field(default_factory=dict)
    split_manifest_hash: str = ""
    subset_manifest_hash: str = ""
    split_manifest_path: str = ""
    subset_manifest_path: str = ""
    target_eval_indices_hash: str = ""
    target_validation_indices_hash: str = ""
    target_training_subset_hash: str = ""
    paired_target_prefix_hash: str = ""
    auxiliary_training_subset_hashes: dict[str, str] = field(default_factory=dict)
    feasibility: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_train is None:
            self.target_train = self.train
        if self.target_train_eval is None:
            self.target_train_eval = self.target_train
        if self.target_val is None:
            self.target_val = self.val
        if self.aux_eval_datasets is None:
            self.aux_eval_datasets = list(self.auxiliary_eval_by_class.values())


def _job_value(job: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if not job:
        return default
    value = job.get(key, default)
    return default if value is None or value == "" else value


def _parse_aux(job: Mapping[str, Any] | None) -> list[str]:
    raw = _job_value(job, "aux_composition", "[]")
    if isinstance(raw, list):
        return list(map(str, raw))
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        value = []
    return list(map(str, value)) if isinstance(value, list) else []


def _target_config(cfg: Mapping[str, Any], target_class: str) -> Mapping[str, Any]:
    for target in cfg.get("targets", []):
        if str(target.get("synset") or target.get("name")) == str(target_class):
            return target
    return {}


def _all_auxiliary_classes(cfg: Mapping[str, Any], target_class: str, selected: list[str]) -> list[str]:
    target = _target_config(cfg, target_class)
    groups = target.get("auxiliary_sets") or cfg.get("auxiliary_sets", {})
    values = [str(item) for candidates in groups.values() for item in candidates]
    return list(dict.fromkeys([*values, *map(str, selected)]))


def _uses_auxiliary_data(model_type: str) -> bool:
    return model_type not in TARGET_ONLY_MODEL_TYPES and not model_type.startswith("unconditional")


def _split_settings(cfg: Mapping[str, Any], *, num_auxiliary_classes: int) -> dict[str, Any]:
    split = dict(cfg.get("data_split", {}))
    evaluation = dict(cfg.get("evaluation", {}))
    eval_size = int(split.get("target_eval_size", evaluation.get("real_eval_max", cfg.get("num_real_eval", 1000))))
    val_size = int(split.get("target_val_size", 100))
    legacy_source = evaluation.get("eval_split")
    source = str(split.get("eval_source", legacy_source or "train_holdout"))
    if source in {"val", "validation", "test", "official"}:
        source = "test" if str(cfg.get("dataset", "")).lower().startswith("cifar") else "val"
    aux_eval_default = max(1, eval_size // max(num_auxiliary_classes, 1))
    return {
        "eval_source": source,
        "target_eval_size": eval_size,
        "target_val_size": val_size,
        "auxiliary_eval_size": int(split.get("auxiliary_eval_size", aux_eval_default)),
        "mode": str(evaluation.get("mode", "debug" if cfg.get("use_fake_data", False) else "strict")),
        "nested_training_subsets": bool(split.get("nested_training_subsets", True)),
    }


def _manifest_root(cfg: Mapping[str, Any]) -> Path:
    split = cfg.get("data_split", {})
    configured = split.get("manifest_root")
    if configured:
        return Path(resolve_env_path(str(configured)))
    output_root = Path(resolve_env_path(cfg.get("output_root"), "image_transfer_results"))
    return output_root / "manifests"


def _manifest_seeds(
    cfg: Mapping[str, Any], job: Mapping[str, Any] | None, legacy_seed: int
) -> tuple[int, int]:
    split = cfg.get("data_split", {})
    legacy_value = _job_value(job, "data_split_seed", split.get("data_split_seed"))
    holdout_value = _job_value(job, "holdout_seed", split.get("holdout_seed"))
    subset_value = _job_value(job, "training_subset_seed", split.get("training_subset_seed"))
    if legacy_value is None and holdout_value is None and subset_value is None:
        # The pre-v2 function-level ``seed`` fallback remains deterministic but
        # does not emit a deprecation warning because no deprecated field was
        # explicitly supplied.
        holdout_value = subset_value = int(legacy_seed)
    return resolve_manifest_seeds(
        holdout_seed=None if holdout_value is None else int(holdout_value),
        training_subset_seed=None if subset_value is None else int(subset_value),
        data_split_seed=None if legacy_value is None else int(legacy_value),
    )


def _indices(references: list[str], expected_sources: set[str]) -> list[int]:
    result: list[int] = []
    for reference in references:
        source, value = parse_sample_ref(reference)
        if source not in expected_sources:
            raise ValueError(f"Reference {reference!r} belongs to {source!r}, expected one of {sorted(expected_sources)}")
        if not isinstance(value, int):
            raise ValueError(f"Dataset builder requires integer manifest indices, got {reference!r}")
        result.append(value)
    return result


def _model_target_count(model_type: str, n0: int, m_per_aux: int, k_aux: int) -> int:
    return int(n0) + int(m_per_aux) * int(k_aux) if model_type in EQUAL_TOTAL_MODEL_TYPES else int(n0)


def _imagefolder_fingerprint(dataset: Any, classes: list[str], eval_dataset: Any | None) -> str:
    def selected_paths(current: Any | None) -> dict[str, list[list[str | int]]]:
        if current is None:
            return {}
        root = Path(current.root)
        result: dict[str, list[list[str | int]]] = {}
        for label in classes:
            old = current.class_to_idx.get(label)
            if old is None:
                continue
            result[label] = [
                [str(Path(path).relative_to(root)), int(Path(path).stat().st_size)]
                for path, y in current.samples
                if y == old
            ]
        return result

    return canonical_sha256({"train": selected_paths(dataset), "evaluation": selected_paths(eval_dataset)})


def _create_and_persist_manifests(
    cfg: Mapping[str, Any],
    job: Mapping[str, Any] | None,
    *,
    target_class: str,
    all_auxiliary: list[str],
    holdout_seed: int,
    training_subset_seed: int,
    train_pools: Mapping[str, list[int]],
    eval_pools: Mapping[str, list[int]] | None,
    dataset_fingerprint: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    settings = _split_settings(cfg, num_auxiliary_classes=len(all_auxiliary))
    split = build_split_manifest(
        dataset_name=str(cfg.get("dataset", "cifar10")),
        target_class=target_class,
        holdout_seed=holdout_seed,
        train_pools=train_pools,
        auxiliary_classes=all_auxiliary,
        eval_pools=eval_pools,
        eval_source=settings["eval_source"],
        target_eval_size=settings["target_eval_size"],
        target_val_size=settings["target_val_size"],
        auxiliary_eval_size=settings["auxiliary_eval_size"],
        dataset_fingerprint=dataset_fingerprint,
        mode=settings["mode"],
    )
    split_path = split_manifest_path(
        _manifest_root(cfg),
        dataset_name=str(cfg.get("dataset", "cifar10")),
        target_class=target_class,
        holdout_seed=holdout_seed,
        split_manifest_hash=split["split_manifest_hash"],
    )
    split, split_path = persist_or_validate_manifest(split, split_path)
    subset = build_subset_manifest(
        split,
        training_subset_seed=training_subset_seed,
        nested_training_subsets=settings["nested_training_subsets"],
    )
    subset_path = subset_manifest_path(
        _manifest_root(cfg),
        dataset_name=str(cfg.get("dataset", "cifar10")),
        target_class=target_class,
        holdout_seed=holdout_seed,
        training_subset_seed=training_subset_seed,
        split_manifest_hash=split["split_manifest_hash"],
        subset_manifest_hash=subset["subset_manifest_hash"],
    )
    subset, subset_path = persist_or_validate_manifest(subset, subset_path)
    return split, subset, split_path, subset_path


def count_available_target_images(cfg: dict[str, Any], target_synset: str) -> int:
    """Count target training images available *after* configured reservations."""

    dataset_name = str(cfg.get("dataset", "cifar10")).lower()
    image_size = int(cfg.get("image_size", 32))
    data_root = resolve_env_path(cfg.get("data_root"), "data")
    settings = _split_settings(cfg, num_auxiliary_classes=0)
    reserve = settings["target_eval_size"] + settings["target_val_size"] if settings["eval_source"] == "train_holdout" else 0
    if cfg.get("use_fake_data", False):
        raw = int(cfg.get("fake_data_size", 100000))
    elif dataset_name.startswith("cifar"):
        dataset = load_cifar10(data_root, image_size, train=True, download=bool(cfg.get("download", True)))
        target_id = class_id(target_synset)
        raw = sum(1 for label in dataset.targets if int(label) == target_id)
    else:
        validate_synsets(data_root, "train", [target_synset])
        dataset = load_imagefolder(data_root, "train", image_size, train=False)
        target_idx = dataset.class_to_idx[target_synset]
        raw = sum(1 for _, label in dataset.samples if label == target_idx)
    return max(raw - reserve, 0)


def build_datasets_for_job(
    cfg: dict[str, Any],
    job: dict[str, Any] | None,
    *,
    n0: int,
    m_per_aux: int,
    k_aux: int,
    seed: int,
    model_type: str,
    manifest: Mapping[str, Any] | None = None,
) -> DatasetBundle:
    """Build job datasets from one persisted, model-independent manifest.

    The legacy keyword-only signature remains valid.  Callers may additionally
    provide a preloaded ``manifest``; otherwise it is deterministically created
    from ``data_split_seed`` and persisted beneath ``data_split.manifest_root``.
    """

    dataset_name = str(cfg.get("dataset", "cifar10")).lower()
    image_size = int(cfg.get("image_size", 32))
    data_root = resolve_env_path(cfg.get("data_root"), "data")
    target_class = str(
        _job_value(job, "target_synset", cfg.get("targets", [{"synset": "dog"}])[0].get("synset", "dog"))
    )
    selected_auxiliary = _parse_aux(job) if _uses_auxiliary_data(model_type) else []
    all_auxiliary = _all_auxiliary_classes(cfg, target_class, selected_auxiliary)
    holdout_seed, training_subset_seed = _manifest_seeds(cfg, job, seed)
    target_count = _model_target_count(model_type, n0, m_per_aux, k_aux)

    train_base: Dataset
    train_eval_base: Dataset
    official_eval_base: Dataset | None
    label_ids: dict[str, int]
    train_pools: dict[str, list[int]]
    eval_pools: dict[str, list[int]] | None
    fingerprint: str | None = None
    source = _split_settings(cfg, num_auxiliary_classes=len(all_auxiliary))["eval_source"]

    if cfg.get("use_fake_data", False):
        all_classes = list(dict.fromkeys([target_class, *all_auxiliary]))
        per_class = int(cfg.get("fake_data_size", 100000))
        total = max(per_class * len(all_classes), 1)
        label_ids = {label: index for index, label in enumerate(all_classes)}
        fake_data_seed = int(cfg.get("fake_data_seed", 0))
        train_base = BlockLabeledDataset(
            build_fake_data(total, image_size, max(len(all_classes), 1), fake_data_seed), per_class, len(all_classes)
        )
        train_eval_base = train_base
        official_eval_base = BlockLabeledDataset(
            build_fake_data(total, image_size, max(len(all_classes), 1), fake_data_seed + 1), per_class, len(all_classes)
        )
        train_pools = {
            label: list(range(index * per_class, (index + 1) * per_class)) for label, index in label_ids.items()
        }
        eval_pools = dict(train_pools) if source != "train_holdout" else None
        fingerprint = canonical_sha256(
            {
                "kind": "FakeData",
                "per_class": per_class,
                "classes": all_classes,
                "image_size": image_size,
                "dataset_generation_seed": fake_data_seed,
            }
        )
    elif dataset_name.startswith("cifar"):
        train_base = load_cifar10(data_root, image_size, train=True, download=bool(cfg.get("download", True)))
        train_eval_base = load_cifar10(
            data_root, image_size, train=True, download=bool(cfg.get("download", True)), eval_transform=True
        )
        official_eval_base = load_cifar10(
            data_root, image_size, train=False, download=bool(cfg.get("download", True)), eval_transform=True
        )
        label_ids = {label: class_id(label) for label in [target_class, *all_auxiliary]}
        train_pools = {
            label: [index for index, y in enumerate(train_base.targets) if int(y) == old]
            for label, old in label_ids.items()
        }
        eval_pools = {
            label: [index for index, y in enumerate(official_eval_base.targets) if int(y) == old]
            for label, old in label_ids.items()
        } if source != "train_holdout" else None
    else:
        classes = [target_class, *all_auxiliary]
        validate_synsets(data_root, "train", classes)
        if source != "train_holdout":
            validate_synsets(data_root, "val", classes)
        train_base = load_imagefolder(data_root, "train", image_size, train=True)
        train_eval_base = load_imagefolder(data_root, "train", image_size, train=False)
        official_eval_base = load_imagefolder(data_root, "val", image_size, train=False) if source != "train_holdout" else None
        label_ids = {label: train_base.class_to_idx[label] for label in classes}
        train_pools = {
            label: [index for index, (_, y) in enumerate(train_base.samples) if y == old]
            for label, old in label_ids.items()
        }
        eval_pools = None
        if official_eval_base is not None:
            eval_pools = {
                label: [index for index, (_, y) in enumerate(official_eval_base.samples) if y == official_eval_base.class_to_idx[label]]
                for label in classes
            }
        fingerprint = _imagefolder_fingerprint(train_base, classes, official_eval_base)

    if manifest is None:
        split_manifest, subset_manifest, persisted_split_path, persisted_subset_path = _create_and_persist_manifests(
            cfg,
            job,
            target_class=target_class,
            all_auxiliary=all_auxiliary,
            holdout_seed=holdout_seed,
            training_subset_seed=training_subset_seed,
            train_pools=train_pools,
            eval_pools=eval_pools,
            dataset_fingerprint=fingerprint,
        )
        loaded_manifest = combine_manifests(split_manifest, subset_manifest)
    else:
        loaded_manifest = dict(manifest)
        split_manifest = dict(loaded_manifest.get("split_manifest", {}))
        subset_manifest = dict(loaded_manifest.get("subset_manifest", {}))
        validate_split_manifest(split_manifest)
        validate_subset_manifest(subset_manifest, split_manifest=split_manifest)
        persisted_split_path = split_manifest_path(
            _manifest_root(cfg),
            dataset_name=str(cfg.get("dataset", "cifar10")),
            target_class=target_class,
            holdout_seed=holdout_seed,
            split_manifest_hash=split_manifest["split_manifest_hash"],
        )
        persisted_subset_path = subset_manifest_path(
            _manifest_root(cfg),
            dataset_name=str(cfg.get("dataset", "cifar10")),
            target_class=target_class,
            holdout_seed=holdout_seed,
            training_subset_seed=training_subset_seed,
            split_manifest_hash=split_manifest["split_manifest_hash"],
            subset_manifest_hash=subset_manifest["subset_manifest_hash"],
        )

    feasibility = equal_total_feasibility(loaded_manifest, n0=n0, m_per_aux=m_per_aux, k_aux=k_aux)
    if model_type in EQUAL_TOTAL_MODEL_TYPES and not feasibility["feasible"]:
        raise ManifestInsufficientDataError(
            "Equal-total target-only baseline is unavailable after target validation/evaluation reservations",
            details=feasibility,
        )

    target_refs = target_training_subset(loaded_manifest, target_count)
    target_indices = _indices(target_refs, {"train"})
    aux_refs = {
        auxiliary: auxiliary_training_subset(loaded_manifest, auxiliary, m_per_aux)
        for auxiliary in selected_auxiliary
    }
    aux_indices = {auxiliary: _indices(refs, {"train"}) for auxiliary, refs in aux_refs.items()}
    paired_target_refs = target_training_subset(loaded_manifest, n0)

    selected_labels = [target_class, *selected_auxiliary]
    target_train = RemappedDataset(
        Subset(train_base, target_indices), [label_ids[target_class]], {label_ids[target_class]: 0}
    ) if dataset_name.startswith("cifar") or cfg.get("use_fake_data", False) else RemappedImageFolder(
        Subset(train_base, target_indices), {label_ids[target_class]: 0}
    )
    target_train_eval = RemappedDataset(
        Subset(train_eval_base, target_indices), [label_ids[target_class]], {label_ids[target_class]: 0}
    ) if dataset_name.startswith("cifar") or cfg.get("use_fake_data", False) else RemappedImageFolder(
        Subset(train_eval_base, target_indices), {label_ids[target_class]: 0}
    )

    auxiliary_train_by_class: dict[str, Dataset] = {}
    auxiliary_train_eval_by_class: dict[str, Dataset] = {}
    for position, auxiliary in enumerate(selected_auxiliary, start=1):
        if dataset_name.startswith("cifar") or cfg.get("use_fake_data", False):
            dataset = RemappedDataset(
                Subset(train_base, aux_indices[auxiliary]), [label_ids[auxiliary]], {label_ids[auxiliary]: position}
            )
            eval_view = RemappedDataset(
                Subset(train_eval_base, aux_indices[auxiliary]),
                [label_ids[auxiliary]],
                {label_ids[auxiliary]: position},
            )
        else:
            dataset = RemappedImageFolder(Subset(train_base, aux_indices[auxiliary]), {label_ids[auxiliary]: position})
            eval_view = RemappedImageFolder(
                Subset(train_eval_base, aux_indices[auxiliary]), {label_ids[auxiliary]: position}
            )
        auxiliary_train_by_class[auxiliary] = dataset
        auxiliary_train_eval_by_class[auxiliary] = eval_view
    auxiliary_train_parts = list(auxiliary_train_by_class.values())
    auxiliary_train = ConcatDataset(auxiliary_train_parts) if auxiliary_train_parts else None
    train = ConcatDataset([target_train, *auxiliary_train_parts]) if auxiliary_train_parts else target_train

    def target_holdout_dataset(key: str) -> Dataset:
        references = list(loaded_manifest["target"][key])
        expected_source = "train" if source == "train_holdout" else source
        indices = _indices(references, {expected_source})
        base = train_eval_base if source == "train_holdout" else official_eval_base
        assert base is not None
        old = label_ids[target_class] if source == "train_holdout" or dataset_name.startswith("cifar") or cfg.get("use_fake_data", False) else base.class_to_idx[target_class]
        if dataset_name.startswith("cifar") or cfg.get("use_fake_data", False):
            return RemappedDataset(Subset(base, indices), [old], {old: 0})
        return RemappedImageFolder(Subset(base, indices), {old: 0})

    target_val = target_holdout_dataset("validation")
    target_eval = target_holdout_dataset("eval")

    auxiliary_eval_by_class: dict[str, Dataset] = {}
    for position, auxiliary in enumerate(selected_auxiliary, start=1):
        references = list(loaded_manifest["auxiliary"][auxiliary]["eval_candidate_pool"])
        expected_source = "train" if source == "train_holdout" else source
        indices = _indices(references, {expected_source})
        base = train_eval_base if source == "train_holdout" else official_eval_base
        assert base is not None
        old = label_ids[auxiliary] if source == "train_holdout" or dataset_name.startswith("cifar") or cfg.get("use_fake_data", False) else base.class_to_idx[auxiliary]
        if dataset_name.startswith("cifar") or cfg.get("use_fake_data", False):
            dataset = RemappedDataset(Subset(base, indices), [old], {old: position})
        else:
            dataset = RemappedImageFolder(Subset(base, indices), {old: position})
        auxiliary_eval_by_class[auxiliary] = dataset

    return DatasetBundle(
        train=train,
        val=target_val,
        target_eval=target_eval,
        class_labels=selected_labels,
        aux_synsets=selected_auxiliary,
        total_train_images=len(train),
        num_target_available=len(loaded_manifest["target"]["train_candidate_pool"]),
        aux_eval_datasets=list(auxiliary_eval_by_class.values()),
        target_train=target_train,
        target_train_eval=target_train_eval,
        auxiliary_train=auxiliary_train,
        target_val=target_val,
        auxiliary_train_datasets=auxiliary_train_by_class,
        auxiliary_train_eval_by_class=auxiliary_train_eval_by_class,
        auxiliary_eval_by_class=auxiliary_eval_by_class,
        manifest=loaded_manifest,
        manifest_hash=str(loaded_manifest["manifest_hash"]),
        manifest_path=str(persisted_subset_path),
        split_manifest=split_manifest,
        subset_manifest=subset_manifest,
        split_manifest_hash=str(split_manifest["split_manifest_hash"]),
        subset_manifest_hash=str(subset_manifest["subset_manifest_hash"]),
        split_manifest_path=str(persisted_split_path),
        subset_manifest_path=str(persisted_subset_path),
        target_eval_indices_hash=canonical_sha256(split_manifest["target"]["eval"]),
        target_validation_indices_hash=canonical_sha256(split_manifest["target"]["validation"]),
        target_training_subset_hash=canonical_sha256(target_refs),
        paired_target_prefix_hash=canonical_sha256(paired_target_refs),
        auxiliary_training_subset_hashes={
            auxiliary: canonical_sha256(references) for auxiliary, references in sorted(aux_refs.items())
        },
        feasibility=feasibility,
    )
