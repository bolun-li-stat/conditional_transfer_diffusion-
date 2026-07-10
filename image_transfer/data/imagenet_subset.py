from __future__ import annotations

import random
from pathlib import Path
from torch.utils.data import Dataset, Subset

from .transforms import build_eval_transform, build_train_transform, require_torchvision

try:
    from torchvision.datasets import ImageFolder
except Exception:  # pragma: no cover
    ImageFolder = None


class RemappedImageFolder(Dataset):
    def __init__(self, dataset: Dataset, old_to_new: dict[int, int]) -> None:
        self.dataset = dataset
        self.old_to_new = old_to_new

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        return x, self.old_to_new[int(y)]


def validate_synsets(data_root: str | Path, split: str, synsets: list[str]) -> None:
    root = Path(data_root) / split
    if not root.is_dir():
        raise FileNotFoundError(f"ImageNet split directory not found: {root}")
    missing = [synset for synset in synsets if not (root / synset).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing ImageNet synset directories under {root}: {missing}")


def load_imagefolder(data_root: str | Path, split: str, image_size: int, train: bool):
    require_torchvision()
    if ImageFolder is None:
        raise RuntimeError("torchvision.datasets.ImageFolder is unavailable")
    transform = build_train_transform(image_size) if train else build_eval_transform(image_size)
    return ImageFolder(str(Path(data_root) / split), transform=transform)


def indices_for_synsets(dataset, synsets: list[str], counts: dict[str, int], seed: int) -> tuple[list[int], dict[int, int]]:
    class_to_idx = dataset.class_to_idx
    missing = [synset for synset in synsets if synset not in class_to_idx]
    if missing:
        raise FileNotFoundError(f"Synsets missing from ImageFolder class index: {missing}")
    old_to_new = {class_to_idx[synset]: new for new, synset in enumerate(synsets)}
    by_class = {class_to_idx[synset]: [] for synset in synsets}
    for index, (_, old_label) in enumerate(dataset.samples):
        if old_label in by_class:
            by_class[old_label].append(index)
    rng = random.Random(seed)
    selected: list[int] = []
    for synset, need in counts.items():
        old_label = class_to_idx[synset]
        available = by_class.get(old_label, [])
        if len(available) < need:
            raise ValueError(f"Synset {synset} has {len(available)} images, need {need}")
        rng.shuffle(available)
        selected.extend(available[:need])
    return selected, old_to_new
