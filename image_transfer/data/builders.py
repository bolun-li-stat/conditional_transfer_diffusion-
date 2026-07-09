from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset, Dataset, Subset

from image_transfer.utils.io import resolve_env_path
from .cifar import RemappedDataset, build_fake_data, class_id, indices_for_classes, load_cifar10
from .imagenet_subset import RemappedImageFolder, indices_for_synsets, load_imagefolder, validate_synsets


@dataclass
class DatasetBundle:
    train: Dataset
    val: Dataset
    target_eval: Dataset
    class_labels: list[str]
    aux_synsets: list[str]
    total_train_images: int
    num_target_available: int
    skipped: bool = False
    skip_reason: str = ""


def _job_value(job: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if not job:
        return default
    value = job.get(key, default)
    return default if value in {None, ""} else value


def _parse_aux(job: dict[str, Any] | None) -> list[str]:
    raw = _job_value(job, "aux_composition", "[]")
    if isinstance(raw, list):
        return raw
    try:
        value = json.loads(raw)
    except Exception:
        value = []
    return list(value) if isinstance(value, list) else []


def count_available_target_images(cfg: dict[str, Any], target_synset: str) -> int:
    dataset_name = str(cfg.get("dataset", "cifar10")).lower()
    image_size = int(cfg.get("image_size", 32))
    data_root = resolve_env_path(cfg.get("data_root"), "data")
    if cfg.get("use_fake_data", False):
        return int(cfg.get("fake_data_size", 100000))
    if dataset_name.startswith("cifar"):
        dataset = load_cifar10(data_root, image_size, train=True, download=bool(cfg.get("download", True)))
        target_id = class_id(target_synset)
        return sum(1 for label in dataset.targets if int(label) == target_id)
    validate_synsets(data_root, "train", [target_synset])
    dataset = load_imagefolder(data_root, "train", image_size, train=False)
    target_idx = dataset.class_to_idx[target_synset]
    return sum(1 for _, label in dataset.samples if label == target_idx)


def build_datasets_for_job(cfg: dict[str, Any], job: dict[str, Any] | None, *, n0: int, m_per_aux: int, k_aux: int, seed: int, model_type: str) -> DatasetBundle:
    dataset_name = str(cfg.get("dataset", "cifar10")).lower()
    image_size = int(cfg.get("image_size", 32))
    data_root = resolve_env_path(cfg.get("data_root"), "data")
    target_synset = str(_job_value(job, "target_synset", cfg.get("targets", [{"synset": "dog"}])[0].get("synset", "dog")))
    aux_synsets = [] if model_type.startswith("unconditional") else _parse_aux(job)
    conditional = not model_type.startswith("unconditional")
    target_need = n0
    if model_type == "unconditional_equal_total":
        target_need = n0 + k_aux * m_per_aux
    class_labels = [target_synset] + aux_synsets

    if cfg.get("use_fake_data", False):
        total = target_need if not conditional else n0 + len(aux_synsets) * m_per_aux
        train = build_fake_data(max(total, 1), image_size, max(len(class_labels), 1), seed)
        val = build_fake_data(max(n0, 1), image_size, 1, seed + 1000)
        return DatasetBundle(train=train, val=val, target_eval=val, class_labels=class_labels, aux_synsets=aux_synsets, total_train_images=total, num_target_available=int(cfg.get("fake_data_size", 100000)))

    if dataset_name.startswith("cifar"):
        train_base = load_cifar10(data_root, image_size, train=True, download=bool(cfg.get("download", True)))
        val_base = load_cifar10(data_root, image_size, train=False, download=bool(cfg.get("download", True)))
        target_id = class_id(target_synset)
        aux_ids = [class_id(aux) for aux in aux_synsets]
        counts = {target_id: target_need if not conditional else n0}
        if conditional:
            counts.update({aux_id: m_per_aux for aux_id in aux_ids})
        train_indices = indices_for_classes(train_base, [target_id] + aux_ids, counts, seed)
        val_count = min(max(n0, 1), sum(1 for y in val_base.targets if int(y) == target_id))
        val_indices = indices_for_classes(val_base, [target_id], {target_id: val_count}, seed + 1)
        label_to_new = {old: new for new, old in enumerate([target_id] + aux_ids)}
        train = RemappedDataset(Subset(train_base, train_indices), [target_id] + aux_ids, label_to_new)
        val = RemappedDataset(Subset(val_base, val_indices), [target_id], {target_id: 0})
        target_available = sum(1 for label in train_base.targets if int(label) == target_id)
        return DatasetBundle(train=train, val=val, target_eval=val, class_labels=class_labels, aux_synsets=aux_synsets, total_train_images=len(train), num_target_available=target_available)

    validate_synsets(data_root, "train", class_labels)
    validate_synsets(data_root, "val", [target_synset])
    train_base = load_imagefolder(data_root, "train", image_size, train=True)
    val_base = load_imagefolder(data_root, "val", image_size, train=False)
    counts = {target_synset: target_need if not conditional else n0}
    if conditional:
        counts.update({aux: m_per_aux for aux in aux_synsets})
    train_indices, old_to_new = indices_for_synsets(train_base, class_labels, counts, seed)
    target_available = sum(1 for _, y in train_base.samples if y == train_base.class_to_idx[target_synset])
    val_available = sum(1 for _, y in val_base.samples if y == val_base.class_to_idx[target_synset])
    val_count = min(max(n0, 1), val_available)
    val_indices, val_map = indices_for_synsets(val_base, [target_synset], {target_synset: val_count}, seed + 1)
    train = RemappedImageFolder(Subset(train_base, train_indices), old_to_new)
    val = RemappedImageFolder(Subset(val_base, val_indices), val_map)
    return DatasetBundle(train=train, val=val, target_eval=val, class_labels=class_labels, aux_synsets=aux_synsets, total_train_images=len(train), num_target_available=target_available)
