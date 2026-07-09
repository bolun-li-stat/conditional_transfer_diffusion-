from __future__ import annotations

import json
from collections import Counter

import torch


def _imagenet_index(synset: str) -> int | None:
    try:
        from torchvision.models import ResNet50_Weights
        from image_transfer.data.class_sets import class_name
    except Exception:
        return None
    categories = ResNet50_Weights.DEFAULT.meta.get("categories", [])
    wanted = class_name(synset).lower().replace(" / ", ", ")
    for idx, category in enumerate(categories):
        text = str(category).lower()
        if text == wanted or wanted in text or text in wanted:
            return idx
    return None


@torch.no_grad()
def evaluate_classifier_fidelity(samples: torch.Tensor, target_synset: str, aux_synsets: list[str], device="cpu") -> dict[str, float | str]:
    try:
        from torchvision.models import ResNet50_Weights, resnet50
    except Exception:
        return {
            "classifier_target_top1_acc": float("nan"),
            "classifier_target_top5_acc": float("nan"),
            "auxiliary_leakage_rate": float("nan"),
            "top1_prediction_histogram_json": "{}",
        }
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device).eval()
    preprocess = weights.transforms()
    x = ((samples.detach().cpu().clamp(-1, 1) + 1) / 2).to(device)
    logits = model(preprocess(x))
    top5 = logits.topk(5, dim=1).indices.cpu()
    top1 = top5[:, 0].tolist()
    hist = Counter(str(i) for i in top1)
    target_index = _imagenet_index(target_synset)
    aux_indices = {_imagenet_index(s) for s in aux_synsets}
    aux_indices.discard(None)
    if target_index is None:
        top1_acc = float("nan")
        top5_acc = float("nan")
    else:
        top1_acc = float((top5[:, 0] == target_index).float().mean())
        top5_acc = float((top5 == target_index).any(dim=1).float().mean())
    leakage = float("nan") if not aux_indices else sum(1 for pred in top1 if pred in aux_indices) / max(len(top1), 1)
    return {
        "classifier_target_top1_acc": top1_acc,
        "classifier_target_top5_acc": top5_acc,
        "auxiliary_leakage_rate": leakage,
        "top1_prediction_histogram_json": json.dumps(hist, sort_keys=True),
    }
