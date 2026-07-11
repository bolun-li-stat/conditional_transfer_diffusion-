from __future__ import annotations

import argparse
import csv
import os
from types import SimpleNamespace

from image_transfer.scripts.train_one import run
from image_transfer.utils.io import (
    get_git_sha,
    load_yaml,
    resolve_env_path,
    write_failure_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-csv", required=True)
    parser.add_argument("--job-index", type=int, default=None)
    parser.add_argument("--device", default=None)
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true")
    restart.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--override-readiness-gate", action="store_true")
    args = parser.parse_args()
    index = args.job_index if args.job_index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    with open(args.jobs_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if index < 0 or index >= len(rows):
        raise IndexError(f"job-index {index} out of range for {len(rows)} jobs")
    job = rows[index]
    cfg = load_yaml(job["config_path"])
    results_root = resolve_env_path(cfg.get("output_root"), "image_transfer_results")
    run_id = job.get("run_id") or f"job_{index}"

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
        override_readiness_gate=args.override_readiness_gate,
    )
    try:
        run(namespace, job)
    except BaseException as exception:
        failure_path = write_failure_result(
            results_root,
            run_id,
            exception,
            config=cfg,
            job=job,
            git_sha=get_git_sha(),
        )
        print(f"failure record: {failure_path}")
        raise


if __name__ == "__main__":
    main()
