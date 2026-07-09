from __future__ import annotations

import argparse
import csv
import os
from types import SimpleNamespace

from image_transfer.scripts.train_one import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-csv", required=True)
    parser.add_argument("--job-index", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    index = args.job_index if args.job_index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    with open(args.jobs_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if index < 0 or index >= len(rows):
        raise IndexError(f"job-index {index} out of range for {len(rows)} jobs")
    job = rows[index]
    namespace = SimpleNamespace(
        config=job["config_path"],
        experiment=job["experiment"],
        model_type=job["model_type"],
        max_steps=None,
        n0=int(job["n0"]),
        m_per_aux=int(job["m_per_aux"]),
        K_aux=int(job["K_aux"]),
        num_generated=None,
        device=args.device,
        seed=int(job["seed"]),
        image_size=None,
        resume=args.resume,
        force=args.force,
        dry_run=args.dry_run,
    )
    run(namespace, job)


if __name__ == "__main__":
    main()
