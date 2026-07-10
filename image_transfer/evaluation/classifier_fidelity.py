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
def evaluate_classifier_fidelity(samples: torch.Tensor, target_synset: str, aux_synsets: list[str], device="cpu", batch_size: int = 64) -> dict[str, float | str]:
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
    top5_chunks = []
    for start in range(0, samples.shape[0], batch_size):
        batch = samples[start : start + batch_size]
        x = ((batch.detach().cpu().clamp(-1, 1) + 1) / 2).to(device)
        logits = model(preprocess(x))
        top5_chunks.append(logits.topk(5, dim=1).indices.cpu())
    top5 = torch.cat(top5_chunks, dim=0) if top5_chunks else torch.empty(0, 5, dtype=torch.long)
    top1 = top5[:, 0].tolist() if len(top5) else []
    hist = Counter(str(i) for i in top1)
    target_index = _imagenet_index(target_synset)
    aux_indices = {_imagenet_index(s) for s in aux_synsets}
    aux_indices.discard(None)
    if target_index is None or len(top5) == 0:
        top1_acc = float("nan")
        top5_acc = float("nan")
    else:
        top1_acc = float((top5[:, 0] == target_index).float().mean())
        top5_acc = float((top5 == target_index).any(dim=1).float().mean())
    leakage = float("nan") if not aux_indices or not top1 else sum(1 for pred in top1 if pred in aux_indices) / max(len(top1), 1)
    return {
        "classifier_target_top1_acc": top1_acc,
        "classifier_target_top5_acc": top5_acc,
        "auxiliary_leakage_rate": leakage,
        "top1_prediction_histogram_json": json.dumps(hist, sort_keys=True),
    }
