"""Fixed, serializable corruption banks for paired denoising evaluation.

A bank stores only image positions, diffusion timesteps and independent noise
seeds.  Epsilon tensors are regenerated per record, which keeps artifacts small
and makes the result invariant to evaluation batch size.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


CORRUPTION_BANK_SCHEMA_VERSION = "1.0"
DEFAULT_NOISE_BINS: tuple[tuple[str, float, float], ...] = (
    ("low", 0.0, 0.2),
    ("mid", 0.2, 0.7),
    ("high", 0.7, 1.0),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def normalize_noise_bins(
    noise_bins: Sequence[Sequence[Any]] | Mapping[str, Sequence[float]] | None,
) -> tuple[tuple[str, float, float], ...]:
    if noise_bins is None:
        normalized = DEFAULT_NOISE_BINS
    elif isinstance(noise_bins, Mapping):
        normalized = tuple((str(name), float(bounds[0]), float(bounds[1])) for name, bounds in noise_bins.items())
    else:
        normalized = tuple((str(item[0]), float(item[1]), float(item[2])) for item in noise_bins)
    if not normalized:
        raise ValueError("noise_bins cannot be empty")
    names: set[str] = set()
    previous_end = 0.0
    for name, lower, upper in normalized:
        if not name or name in names:
            raise ValueError(f"noise-bin names must be non-empty and unique: {name!r}")
        names.add(name)
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError(f"invalid normalized noise bin {name}: [{lower}, {upper})")
        if not math.isclose(lower, previous_end, abs_tol=1e-12):
            raise ValueError("noise bins must be ordered, non-overlapping, and cover [0, 1]")
        previous_end = upper
    if not math.isclose(previous_end, 1.0, abs_tol=1e-12):
        raise ValueError("noise bins must cover [0, 1]")
    return normalized


@dataclass(frozen=True)
class CorruptionRecord:
    image_index: int
    timestep: int
    noise_seed: int

    def to_dict(self) -> dict[str, int]:
        return {
            "image_index": int(self.image_index),
            "timestep": int(self.timestep),
            "noise_seed": int(self.noise_seed),
        }


@dataclass(frozen=True)
class CorruptionBank:
    manifest_hash: str
    evaluation_seed: int
    timesteps: int
    corruptions_per_image: int
    noise_bins: tuple[tuple[str, float, float], ...]
    records: tuple[CorruptionRecord, ...]
    split: str = "validation"
    timestep_distribution: str = "uniform"
    schema_version: str = CORRUPTION_BANK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.manifest_hash:
            raise ValueError("manifest_hash is required")
        if self.timesteps < 1:
            raise ValueError("timesteps must be positive")
        if self.corruptions_per_image < 1:
            raise ValueError("corruptions_per_image must be positive")
        if self.timestep_distribution != "uniform":
            raise ValueError("only a direct uniform-t corruption bank is currently supported")
        normalized = normalize_noise_bins(self.noise_bins)
        object.__setattr__(self, "noise_bins", normalized)
        if any(record.timestep < 0 or record.timestep >= self.timesteps for record in self.records):
            raise ValueError("corruption record contains an invalid timestep")
        if any(record.image_index < 0 for record in self.records):
            raise ValueError("corruption record contains a negative image index")
        if any(record.noise_seed < 0 for record in self.records):
            raise ValueError("corruption record contains a negative noise seed")
        counts: dict[int, int] = {}
        for record in self.records:
            counts[record.image_index] = counts.get(record.image_index, 0) + 1
        if counts and set(counts.values()) != {self.corruptions_per_image}:
            raise ValueError("every image must have exactly corruptions_per_image records")

    @property
    def num_images(self) -> int:
        return len({record.image_index for record in self.records})

    @property
    def num_corruptions(self) -> int:
        return len(self.records)

    def payload(self) -> dict[str, Any]:
        return {
            "manifest_hash": self.manifest_hash,
            "evaluation_seed": int(self.evaluation_seed),
            "timesteps": int(self.timesteps),
            "corruptions_per_image": int(self.corruptions_per_image),
            "noise_bins": [list(item) for item in self.noise_bins],
            "records": [record.to_dict() for record in self.records],
            "split": self.split,
            "timestep_distribution": self.timestep_distribution,
            "schema_version": self.schema_version,
        }

    @property
    def bank_hash(self) -> str:
        return _sha256_json(self.payload())

    @property
    def hash(self) -> str:
        """Compatibility alias for ``bank_hash``."""

        return self.bank_hash

    def to_dict(self) -> dict[str, Any]:
        result = self.payload()
        result["corruption_bank_hash"] = self.bank_hash
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorruptionBank":
        bank = cls(
            manifest_hash=str(value["manifest_hash"]),
            evaluation_seed=int(value["evaluation_seed"]),
            timesteps=int(value["timesteps"]),
            corruptions_per_image=int(value["corruptions_per_image"]),
            noise_bins=tuple(tuple(item) for item in value["noise_bins"]),
            records=tuple(CorruptionRecord(**record) for record in value["records"]),
            split=str(value.get("split", "validation")),
            timestep_distribution=str(value.get("timestep_distribution", "uniform")),
            schema_version=str(value.get("schema_version", CORRUPTION_BANK_SCHEMA_VERSION)),
        )
        expected = value.get("corruption_bank_hash")
        if expected is not None and str(expected) != bank.bank_hash:
            raise ValueError("corruption bank hash does not match its payload")
        return bank


def create_corruption_bank(
    *,
    manifest_hash: str,
    evaluation_seed: int,
    timesteps: int,
    corruptions_per_image: int,
    num_images: int | None = None,
    image_indices: Sequence[int] | None = None,
    noise_bins: Sequence[Sequence[Any]] | Mapping[str, Sequence[float]] | None = None,
    split: str = "validation",
    schema_version: str = CORRUPTION_BANK_SCHEMA_VERSION,
) -> CorruptionBank:
    """Create a deterministic direct-uniform-t bank.

    Exactly one of ``num_images`` and ``image_indices`` may be supplied.  Image
    indices are positions in the manifest-ordered evaluation dataset.
    """

    if image_indices is not None and num_images is not None:
        raise ValueError("provide image_indices or num_images, not both")
    if image_indices is None:
        if num_images is None or int(num_images) < 1:
            raise ValueError("num_images must be positive when image_indices is omitted")
        positions = tuple(range(int(num_images)))
    else:
        positions = tuple(int(index) for index in image_indices)
        if not positions or len(set(positions)) != len(positions) or min(positions) < 0:
            raise ValueError("image_indices must be a non-empty set of unique non-negative positions")
    if int(timesteps) < 1 or int(corruptions_per_image) < 1:
        raise ValueError("timesteps and corruptions_per_image must be positive")

    records: list[CorruptionRecord] = []
    normalized_bins = normalize_noise_bins(noise_bins)
    header = _canonical_json(
        {
            "manifest_hash": str(manifest_hash),
            "evaluation_seed": int(evaluation_seed),
            "timesteps": int(timesteps),
            "corruptions_per_image": int(corruptions_per_image),
            "noise_bins": normalized_bins,
            "split": str(split),
            "schema_version": str(schema_version),
        }
    )
    for image_index in positions:
        for corruption_index in range(int(corruptions_per_image)):
            # A counter-mode SHA256 stream is stable across torch/Python RNG
            # implementation changes and includes every bank-defining field.
            digest = hashlib.sha256(
                f"{header}|{image_index}|{corruption_index}".encode("utf-8")
            ).digest()
            records.append(
                CorruptionRecord(
                    image_index=image_index,
                    timestep=int.from_bytes(digest[:8], "big") % int(timesteps),
                    noise_seed=int.from_bytes(digest[8:16], "big") % (2**31 - 1),
                )
            )
    return CorruptionBank(
        manifest_hash=str(manifest_hash),
        evaluation_seed=int(evaluation_seed),
        timesteps=int(timesteps),
        corruptions_per_image=int(corruptions_per_image),
        noise_bins=normalized_bins,
        records=tuple(records),
        split=str(split),
        schema_version=str(schema_version),
    )


# Compatibility alias with the verb used in early design notes.
generate_corruption_bank = create_corruption_bank


def save_corruption_bank(bank: CorruptionBank, path: str | Path) -> Path:
    """Atomically save a bank as canonical JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    data = (_canonical_json(bank.to_dict()) + "\n").encode("utf-8")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_corruption_bank(path: str | Path) -> CorruptionBank:
    with open(path, "r", encoding="utf-8") as handle:
        return CorruptionBank.from_dict(json.load(handle))


def timestep_bin_name(bank: CorruptionBank, timestep: int) -> str:
    fraction = int(timestep) / bank.timesteps
    for name, lower, upper in bank.noise_bins:
        if lower <= fraction < upper:
            return name
    raise RuntimeError(f"timestep {timestep} was not covered by the bank noise bins")


def _dataset_from_input(dataset_or_loader):
    if hasattr(dataset_or_loader, "dataset"):
        return dataset_or_loader.dataset
    return dataset_or_loader


def _image_at(dataset, index: int) -> torch.Tensor:
    item = dataset[index]
    image = item[0] if isinstance(item, (tuple, list)) else item
    if not isinstance(image, torch.Tensor):
        raise TypeError("corruption evaluation datasets must return torch tensors")
    return image


def _clustered_standard_error(values: Sequence[float], clusters: Sequence[int]) -> float:
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for value, cluster in zip(values, clusters):
        sums[cluster] = sums.get(cluster, 0.0) + float(value)
        counts[cluster] = counts.get(cluster, 0) + 1
    image_means = torch.tensor([sums[key] / counts[key] for key in sorted(sums)], dtype=torch.float64)
    if image_means.numel() < 2:
        return float("nan")
    return float(image_means.std(unbiased=True) / math.sqrt(image_means.numel()))


@torch.no_grad()
def evaluate_corruption_bank(
    model,
    diffusion,
    dataset_or_loader,
    bank: CorruptionBank,
    device: str | torch.device,
    *,
    label: int | None = None,
    batch_size: int = 64,
    metric_prefix: str = "validation",
) -> dict[str, float | int | str]:
    """Evaluate epsilon MSE on exactly the records in ``bank``.

    Per-record loss is averaged over channel and spatial dimensions.  Overall
    MSE is the direct average over a uniform-t bank; low/mid/high metrics are
    conditional summaries and are never equally re-averaged.  The reported SE
    first aggregates corruptions within image and then treats images as the
    independent clusters.
    """

    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if int(diffusion.timesteps) != bank.timesteps:
        raise ValueError("corruption bank timesteps do not match the diffusion process")
    dataset = _dataset_from_input(dataset_or_loader)
    if not hasattr(dataset, "__getitem__"):
        raise TypeError("dataset_or_loader must provide a map-style dataset")
    if bank.records and max(record.image_index for record in bank.records) >= len(dataset):
        raise IndexError("corruption bank image index exceeds the evaluation dataset")
    evaluation_device = torch.device(device)
    all_losses: list[float] = []
    all_images: list[int] = []
    losses_by_bin: dict[str, list[float]] = {name: [] for name, _, _ in bank.noise_bins}
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        for start in range(0, len(bank.records), int(batch_size)):
            records = bank.records[start : start + int(batch_size)]
            images = torch.stack([_image_at(dataset, record.image_index) for record in records]).to(evaluation_device)
            timesteps = torch.tensor([record.timestep for record in records], dtype=torch.long, device=evaluation_device)
            noises = []
            for image, record in zip(images, records):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(record.noise_seed))
                noise = torch.randn(tuple(image.shape), generator=generator, dtype=torch.float32)
                noises.append(noise.to(dtype=image.dtype))
            epsilon = torch.stack(noises).to(evaluation_device)
            corrupted, _ = diffusion.q_sample(images, timesteps, noise=epsilon)
            labels = (
                None
                if label is None
                else torch.full((len(records),), int(label), dtype=torch.long, device=evaluation_device)
            )
            prediction = model(corrupted, timesteps, labels)
            per_record = (prediction - epsilon).square().flatten(1).mean(dim=1).detach().double().cpu()
            for value, record in zip(per_record.tolist(), records):
                all_losses.append(float(value))
                all_images.append(record.image_index)
                losses_by_bin[timestep_bin_name(bank, record.timestep)].append(float(value))
    finally:
        if was_training:
            model.train()

    overall = float(torch.tensor(all_losses, dtype=torch.float64).mean()) if all_losses else float("nan")
    result: dict[str, float | int | str] = {
        f"{metric_prefix}_epsilon_mse_target": overall,
        f"{metric_prefix}_epsilon_mse_standard_error": _clustered_standard_error(all_losses, all_images),
        f"num_{metric_prefix}_images": bank.num_images,
        "num_corruptions": bank.num_corruptions,
        "corruptions_per_image": bank.corruptions_per_image,
        "corruption_bank_hash": bank.bank_hash,
        "corruption_bank_manifest_hash": bank.manifest_hash,
        "corruption_timestep_distribution": bank.timestep_distribution,
        "epsilon_mse_dimension_reduction": "mean_over_channels_and_spatial_dimensions",
    }
    for name, _, _ in bank.noise_bins:
        values = losses_by_bin[name]
        result[f"{metric_prefix}_epsilon_mse_{name}_noise"] = (
            float(torch.tensor(values, dtype=torch.float64).mean()) if values else float("nan")
        )
    return result


__all__ = [
    "CORRUPTION_BANK_SCHEMA_VERSION",
    "DEFAULT_NOISE_BINS",
    "CorruptionBank",
    "CorruptionRecord",
    "create_corruption_bank",
    "evaluate_corruption_bank",
    "generate_corruption_bank",
    "load_corruption_bank",
    "normalize_noise_bins",
    "save_corruption_bank",
    "timestep_bin_name",
]
