from __future__ import annotations

import random
from pathlib import Path
from torch.utils.data import Dataset, Subset

from .class_sets import CIFAR10_CLASSES
from .transforms import build_eval_transform, build_train_transform, require_torchvision

try:
    from torchvision.datasets import CIFAR10, FakeData
except Exception:  # pragma: no cover
    CIFAR10 = None
    FakeData = None


class RemappedDataset(Dataset):
    def __init__(self, dataset: Dataset, labels: list[int], label_to_new: dict[int, int]) -> None:
        self.dataset = dataset
        self.labels = labels
        self.label_to_new = label_to_new

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        return x, self.label_to_new[int(y)]


def class_id(name_or_id: str | int) -> int:
    if isinstance(name_or_id, int):
        return name_or_id
    if str(name_or_id).isdigit():
        return int(name_or_id)
    if name_or_id not in CIFAR10_CLASSES:
        raise ValueError(f"Unknown CIFAR-10 class {name_or_id!r}")
    return CIFAR10_CLASSES[name_or_id]


def load_cifar10(root: str | Path, image_size: int, train: bool, download: bool = True, eval_transform: bool = False):
    require_torchvision()
    if CIFAR10 is None:
        raise RuntimeError("torchvision.datasets.CIFAR10 is unavailable")
    transform = build_eval_transform(image_size) if eval_transform else (build_train_transform(image_size) if train else build_eval_transform(image_size))
    return CIFAR10(root=str(root), train=train, download=download, transform=transform)


def indices_for_classes(dataset, class_ids: list[int], counts: dict[int, int], seed: int) -> list[int]:
    rng = random.Random(seed)
    by_class = {class_id: [] for class_id in class_ids}
    targets = getattr(dataset, "targets", None)
    if targets is None and isinstance(dataset, Subset):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    if targets is None:
        raise ValueError("Dataset does not expose class targets")
    for idx, label in enumerate(targets):
        label = int(label)
        if label in by_class:
            by_class[label].append(idx)
    selected: list[int] = []
    for label, need in counts.items():
        available = by_class.get(label, [])
        if len(available) < need:
            raise ValueError(f"CIFAR-10 class {label} has {len(available)} images, need {need}")
        rng.shuffle(available)
        selected.extend(available[:need])
    return selected


def build_fake_data(size: int, image_size: int, num_classes: int, seed: int):
    if FakeData is not None:
        transform = build_eval_transform(image_size)
        return FakeData(size=size, image_size=(3, image_size, image_size), num_classes=num_classes, transform=transform, random_offset=seed)
    import torch
    from torch.utils.data import TensorDataset
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(size, 3, image_size, image_size, generator=generator) * 2 - 1
    y = torch.arange(size) % max(num_classes, 1)
    return TensorDataset(x, y.long())
