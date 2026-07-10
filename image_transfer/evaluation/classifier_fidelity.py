"""Classifier-based target fidelity with exact class-index mappings."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch


# Canonical ILSVRC-2012 indices for every synset used by the checked-in image
# experiment definitions.  Unknown synsets are explicitly unavailable unless a
# reviewed complete mapping is supplied; fuzzy matching category names is never
# used.
IMAGENET_SYNSET_TO_INDEX: dict[str, int] = {
    "n01580077": 17,   # jay
    "n02085620": 151,  # Chihuahua
    "n02091032": 171,  # Italian greyhound
    "n02096585": 195,  # Boston bull
    "n02099601": 207,  # golden retriever
    "n02108089": 242,  # boxer
    "n02108915": 245,  # French bulldog
    "n02110958": 254,  # pug
    "n02114367": 269,  # timber wolf
    "n02119022": 277,  # red fox
    "n02123045": 281,  # tabby cat
    "n02123159": 282,  # tiger cat
    "n02124075": 285,  # Egyptian cat
    "n03028079": 497,  # church
    "n03457902": 580,  # greenhouse
    "n03594945": 609,  # jeep
    "n03710193": 637,  # mailbox
    "n04146614": 779,  # school bus
}
CIFAR10_CLASS_NAMES = {
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
}


def load_synset_index_mapping(path: str | Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("synset-index mapping must be a JSON object")
    result = {str(key): int(index) for key, index in value.items()}
    if len(set(result.values())) != len(result):
        raise ValueError("synset-index mapping contains duplicate classifier indices")
    if any(index < 0 for index in result.values()):
        raise ValueError("synset-index mapping contains a negative index")
    return result


def imagenet_synset_to_index(
    synset: str,
    mapping: Mapping[str, int] | None = None,
) -> int | None:
    source = IMAGENET_SYNSET_TO_INDEX if mapping is None else mapping
    value = source.get(str(synset))
    return None if value is None else int(value)


# Retain the old private name for callers/tests that imported it.
def _imagenet_index(synset: str) -> int | None:
    return imagenet_synset_to_index(synset)


def _normalize_dataset_name(dataset_name: str | None, target_synset: str) -> str:
    inferred = dataset_name or ("cifar10" if target_synset in CIFAR10_CLASS_NAMES else "imagenet")
    inferred = inferred.lower()
    return "imagenet" if inferred.startswith("imagenet") else inferred


def preflight_classifier_fidelity(
    target_synset: str,
    aux_synsets: Sequence[str],
    *,
    dataset_name: str | None = None,
    synset_to_index: Mapping[str, int] | None = None,
) -> dict[str, str | int]:
    """Validate mappings and initialize the pinned paper classifier weights."""

    inferred_dataset = _normalize_dataset_name(dataset_name, target_synset)
    if inferred_dataset in {"cifar", "cifar10"}:
        raise RuntimeError(
            "no version-pinned CIFAR-10 classifier is configured; ImageNet classifier use is disabled"
        )
    if inferred_dataset not in {"imagenet", "imagenet1k", "ilsvrc2012"}:
        raise RuntimeError(f"no classifier implementation is configured for dataset {inferred_dataset!r}")
    mapping = IMAGENET_SYNSET_TO_INDEX if synset_to_index is None else synset_to_index
    missing = [
        synset
        for synset in [target_synset, *aux_synsets]
        if imagenet_synset_to_index(synset, mapping) is None
    ]
    if missing:
        raise KeyError("synsets absent from the reviewed exact classifier mapping: " + ",".join(sorted(missing)))
    try:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT
        classifier = resnet50(weights=weights)
        preprocess = weights.transforms()
    except Exception as exc:
        raise RuntimeError(
            f"ImageNet classifier backend/weights unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    del classifier
    return {
        "classifier_dataset": inferred_dataset,
        "classifier_architecture": "torchvision.models.resnet50",
        "classifier_weights": f"{weights.__class__.__name__}.{weights.name}",
        "classifier_preprocessing": repr(preprocess),
        "classifier_target_index": int(mapping[target_synset]),
    }


def _unavailable(reason: str, *, dataset_name: str, architecture: str = "unavailable") -> dict[str, object]:
    return {
        "classifier_target_top1_acc": float("nan"),
        "classifier_target_top5_acc": float("nan"),
        "auxiliary_leakage_rate": float("nan"),
        "top1_prediction_histogram_json": "{}",
        "classifier_fidelity_status": f"unavailable: {reason}",
        "classifier_unavailable_reason": reason,
        "classifier_dataset": dataset_name,
        "classifier_architecture": architecture,
        "classifier_weights": "unavailable",
        "classifier_preprocessing": "unavailable",
        "classifier_synset_mapping": "exact",
    }


@torch.no_grad()
def evaluate_classifier_fidelity(
    samples: torch.Tensor,
    target_synset: str,
    aux_synsets: Sequence[str],
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    *,
    dataset_name: str | None = None,
    synset_to_index: Mapping[str, int] | None = None,
    classifier=None,
    preprocess=None,
    classifier_architecture: str = "resnet50",
    classifier_weights: str | None = None,
    strict: bool = False,
) -> dict[str, object]:
    """Evaluate target accuracy and top-1 leakage into auxiliary classes.

    CIFAR-10 is deliberately unavailable by default because applying an ImageNet
    classifier to CIFAR images would not be a valid semantic metric.  A caller
    may supply a separately versioned CIFAR classifier, preprocessing function,
    and exact class mapping in a future extension.
    """

    inferred_dataset = _normalize_dataset_name(dataset_name, target_synset)
    if inferred_dataset in {"cifar", "cifar10"} and classifier is None:
        reason = "no version-pinned CIFAR-10 classifier is configured; ImageNet classifier use is disabled"
        if strict:
            raise RuntimeError(reason)
        return _unavailable(reason, dataset_name="cifar10")
    if inferred_dataset not in {"imagenet", "imagenet1k", "ilsvrc2012"} and classifier is None:
        reason = f"no classifier implementation is configured for dataset {inferred_dataset!r}"
        if strict:
            raise RuntimeError(reason)
        return _unavailable(reason, dataset_name=inferred_dataset)

    mapping = IMAGENET_SYNSET_TO_INDEX if synset_to_index is None else synset_to_index
    target_index = imagenet_synset_to_index(target_synset, mapping)
    missing_aux = [synset for synset in aux_synsets if imagenet_synset_to_index(synset, mapping) is None]
    if target_index is None:
        reason = f"target synset {target_synset!r} is absent from the reviewed exact classifier mapping"
        if strict:
            raise KeyError(reason)
        return _unavailable(reason, dataset_name=inferred_dataset, architecture=classifier_architecture)
    if missing_aux and strict:
        raise KeyError(
            "auxiliary synsets absent from the reviewed exact classifier mapping: "
            + ",".join(sorted(missing_aux))
        )

    weights_name = classifier_weights or "provided"
    if classifier is None:
        try:
            from torchvision.models import ResNet50_Weights, resnet50

            weights = ResNet50_Weights.DEFAULT
            classifier = resnet50(weights=weights)
            preprocess = weights.transforms()
            weights_name = f"{weights.__class__.__name__}.{weights.name}"
            classifier_architecture = "torchvision.models.resnet50"
        except Exception as exc:
            reason = f"ImageNet classifier backend/weights unavailable: {type(exc).__name__}: {exc}"
            if strict:
                raise RuntimeError(reason) from exc
            return _unavailable(reason, dataset_name=inferred_dataset, architecture=classifier_architecture)
    if preprocess is None:
        reason = "classifier preprocessing must be supplied with a custom classifier"
        if strict:
            raise ValueError(reason)
        return _unavailable(reason, dataset_name=inferred_dataset, architecture=classifier_architecture)
    if int(batch_size) < 1:
        raise ValueError("classifier batch_size must be positive")

    classifier = classifier.to(device).eval()
    top5_chunks: list[torch.Tensor] = []
    try:
        for start in range(0, samples.shape[0], int(batch_size)):
            batch = samples[start : start + int(batch_size)]
            inputs = ((batch.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0).to(device)
            logits = classifier(preprocess(inputs))
            if logits.ndim != 2:
                raise ValueError("classifier must return [samples, classes] logits")
            top5_chunks.append(logits.topk(min(5, logits.shape[1]), dim=1).indices.cpu())
    except Exception as exc:
        reason = f"classifier inference failed: {type(exc).__name__}: {exc}"
        if strict:
            raise RuntimeError(reason) from exc
        return _unavailable(reason, dataset_name=inferred_dataset, architecture=classifier_architecture)

    top5 = torch.cat(top5_chunks, dim=0) if top5_chunks else torch.empty(0, 5, dtype=torch.long)
    top1 = top5[:, 0].tolist() if top5.numel() else []
    histogram = Counter(str(index) for index in top1)
    if not top1:
        top1_accuracy = float("nan")
        top5_accuracy = float("nan")
    else:
        top1_accuracy = float((top5[:, 0] == target_index).float().mean())
        top5_accuracy = float((top5 == target_index).any(dim=1).float().mean())

    if missing_aux:
        leakage = float("nan")
        status = "partial: target fidelity available but leakage unavailable because auxiliary mappings are missing"
        unavailable_reason = "missing auxiliary synsets: " + ",".join(sorted(missing_aux))
    else:
        auxiliary_indices = {int(mapping[synset]) for synset in aux_synsets}
        leakage = (
            float(sum(prediction in auxiliary_indices for prediction in top1) / len(top1))
            if auxiliary_indices and top1
            else float("nan")
        )
        status = "ok" if top1 else "unavailable: no generated samples"
        unavailable_reason = "" if top1 else "no generated samples"
    return {
        "classifier_target_top1_acc": top1_accuracy,
        "classifier_target_top5_acc": top5_accuracy,
        "auxiliary_leakage_rate": leakage,
        "top1_prediction_histogram_json": json.dumps(histogram, sort_keys=True),
        "classifier_fidelity_status": status,
        "classifier_unavailable_reason": unavailable_reason,
        "classifier_dataset": inferred_dataset,
        "classifier_architecture": classifier_architecture,
        "classifier_weights": weights_name,
        "classifier_preprocessing": repr(preprocess),
        "classifier_synset_mapping": "exact",
        "classifier_target_index": int(target_index),
    }


__all__ = [
    "CIFAR10_CLASS_NAMES",
    "IMAGENET_SYNSET_TO_INDEX",
    "evaluate_classifier_fidelity",
    "imagenet_synset_to_index",
    "load_synset_index_mapping",
    "preflight_classifier_fidelity",
]
