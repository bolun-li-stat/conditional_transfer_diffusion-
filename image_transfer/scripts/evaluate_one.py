"""Standalone evaluation using the same fixed manifest and strict metric API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from image_transfer.data import build_datasets_for_job
from image_transfer.evaluation.classifier_fidelity import evaluate_classifier_fidelity
from image_transfer.evaluation.feature_metrics import compute_feature_metrics
from image_transfer.scripts.train_one import _collect_dataset, _disabled_classifier_metrics
from image_transfer.utils.device import get_device
from image_transfer.utils.io import atomic_write_json, load_yaml


def _load_tensor(path: str | Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older PyTorch
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError("--samples must contain one NCHW tensor")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out-metrics", required=True)
    parser.add_argument("--target-synset", required=True)
    parser.add_argument("--model-type", default="unconditional_n0")
    parser.add_argument("--experiment", choices=["A", "B", "C"], default="A")
    parser.add_argument("--n0", type=int, required=True)
    parser.add_argument("--m-per-aux", type=int, default=0)
    parser.add_argument("--K-aux", type=int, default=0, dest="K_aux")
    parser.add_argument("--data-split-seed", type=int, default=0)
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument("--aux-composition", default="[]")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=["strict", "debug"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    evaluation_cfg = cfg.get("evaluation", {})
    samples = _load_tensor(args.samples)
    job = {
        "experiment": args.experiment,
        "target_synset": args.target_synset,
        "aux_composition": args.aux_composition,
        "model_type": args.model_type,
        "data_split_seed": args.data_split_seed,
    }
    bundle = build_datasets_for_job(
        cfg,
        job,
        n0=args.n0,
        m_per_aux=args.m_per_aux,
        k_aux=args.K_aux,
        seed=args.data_split_seed,
        model_type=args.model_type,
    )
    real_limit = evaluation_cfg.get("real_eval_max")
    real = _collect_dataset(
        bundle.target_eval,
        limit=None if real_limit is None else int(real_limit),
        batch_size=int(evaluation_cfg.get("feature_batch_size", 64)),
    )
    device = get_device(args.device)
    mode = args.mode or str(evaluation_cfg.get("mode", "strict"))
    metrics = compute_feature_metrics(
        samples,
        real,
        mode=mode,
        real_manifest_hash=bundle.manifest_hash,
        cache_dir=Path(args.out_metrics).parent / "real_feature_cache",
        compute_fid=bool(evaluation_cfg.get("compute_fid", evaluation_cfg.get("compute_fid_kid", True))),
        compute_kid=bool(evaluation_cfg.get("compute_kid", evaluation_cfg.get("compute_fid_kid", True))),
        compute_prdc_metrics=bool(evaluation_cfg.get("compute_prdc", False)),
        compute_inception_score=bool(evaluation_cfg.get("compute_inception_score", False)),
        kid_subset_size=int(evaluation_cfg.get("kid_subset_size", 100)),
        kid_num_subsets=int(evaluation_cfg.get("kid_num_subsets", 100)),
        prdc_k=int(evaluation_cfg.get("prdc_k", 5)),
        feature_batch_size=int(evaluation_cfg.get("feature_batch_size", 64)),
        evaluation_seed=args.evaluation_seed,
        device=device,
    )
    if bool(evaluation_cfg.get("compute_classifier_fidelity", evaluation_cfg.get("compute_classifier", False))):
        metrics.update(
            evaluate_classifier_fidelity(
                samples,
                args.target_synset,
                bundle.aux_synsets,
                dataset_name=str(cfg.get("dataset", "imagenet")),
                device=device,
                batch_size=int(evaluation_cfg.get("classifier_batch_size", 64)),
                strict=mode == "strict",
            )
        )
    else:
        metrics.update(_disabled_classifier_metrics(evaluation_cfg))
    record = {
        "samples_path": str(args.samples),
        "manifest_hash": bundle.manifest_hash,
        "manifest_path": bundle.manifest_path,
        "target_synset": args.target_synset,
        "model_type": args.model_type,
        "aux_synsets_json": json.dumps(bundle.aux_synsets),
        "metrics": metrics,
    }
    destination = atomic_write_json(record, args.out_metrics)
    print(destination)


if __name__ == "__main__":
    main()
