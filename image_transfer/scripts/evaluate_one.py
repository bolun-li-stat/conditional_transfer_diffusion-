from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from image_transfer.data import build_datasets_for_job
from image_transfer.evaluation.classifier_fidelity import evaluate_classifier_fidelity
from image_transfer.evaluation.fid_kid import compute_fid_kid
from image_transfer.utils.device import get_device
from image_transfer.utils.io import append_csv_row, load_yaml
from image_transfer.scripts.train_one import FIELDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out-metrics", required=True)
    parser.add_argument("--target-synset", required=True)
    parser.add_argument("--model-type", default="unconditional_n0")
    parser.add_argument("--n0", type=int, required=True)
    parser.add_argument("--m-per-aux", type=int, default=0)
    parser.add_argument("--K-aux", type=int, default=0, dest="K_aux")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aux-composition", default="[]")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    samples = torch.load(args.samples, map_location="cpu")
    job = {"target_synset": args.target_synset, "aux_composition": args.aux_composition, "model_type": args.model_type}
    bundle = build_datasets_for_job(cfg, job, n0=args.n0, m_per_aux=args.m_per_aux, k_aux=args.K_aux, seed=args.seed, model_type=args.model_type)
    real_batches = []
    for x, _ in DataLoader(bundle.target_eval, batch_size=64, shuffle=False):
        real_batches.append(x)
        if sum(batch.shape[0] for batch in real_batches) >= samples.shape[0]:
            break
    real = torch.cat(real_batches, dim=0)[: samples.shape[0]]
    row = {field: "" for field in FIELDS}
    row.update(compute_fid_kid(samples, real, Path(args.out_metrics).with_suffix(".real_cache.pt")))
    row.update(evaluate_classifier_fidelity(samples, args.target_synset, bundle.aux_synsets, device=get_device(args.device)) if cfg.get("evaluation", {}).get("compute_classifier", False) else {"classifier_target_top1_acc": float("nan"), "classifier_target_top5_acc": float("nan"), "auxiliary_leakage_rate": float("nan"), "top1_prediction_histogram_json": "{}"})
    row.update({"dataset": cfg.get("dataset"), "target_synset": args.target_synset, "model_type": args.model_type, "n0": args.n0, "m_per_aux": args.m_per_aux, "K_aux": args.K_aux, "seed": args.seed, "num_generated": int(samples.shape[0]), "num_real_eval": int(real.shape[0])})
    append_csv_row(args.out_metrics, row, FIELDS)
    print(args.out_metrics)


if __name__ == "__main__":
    main()
