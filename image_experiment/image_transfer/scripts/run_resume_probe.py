"""Run save/destroy/rebuild/load/continue checks for both training protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from image_transfer.config import load_resolved_config
from image_transfer.scripts.inspect_environment import validate_environment_report
from image_transfer.training.runtime_probe import PROTOCOLS, RUNTIME_PROBE_SCHEMA_VERSION, run_resume_roundtrip
from image_transfer.utils.io import atomic_write_json, canonical_json_hash, get_git_sha, load_json, utc_timestamp


def run_resume_probe(
    config_path: str | Path,
    *,
    environment_report_path: str | Path,
    work_dir: str | Path | None = None,
    allow_cpu: bool = True,
) -> dict[str, Any]:
    resolved = load_resolved_config(config_path)
    environment = load_json(environment_report_path)
    validate_environment_report(environment)
    cuda = bool(torch.cuda.is_available())
    if not cuda and not allow_cpu:
        raise RuntimeError("CUDA is unavailable and CPU resume probing was disabled")
    device = torch.device("cuda" if cuda else "cpu")
    results: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    for protocol in PROTOCOLS:
        try:
            result = run_resume_roundtrip(
                resolved.resolved,
                device=device,
                protocol=protocol,
                work_dir=Path(work_dir) / protocol if work_dir is not None else None,
            )
            results[protocol] = result
            if not result["passed"]:
                failures.append(
                    {
                        "protocol": protocol,
                        "exception_type": "ResumeMismatch",
                        "message": "continuous and resumed states differ",
                        "resolved_config_hash": resolved.resolved_hash,
                    }
                )
        except Exception as exception:
            failures.append(
                {
                    "protocol": protocol,
                    "exception_type": type(exception).__name__,
                    "message": str(exception),
                    "resolved_config_hash": resolved.resolved_hash,
                }
            )
    report: dict[str, Any] = {
        "schema_version": RUNTIME_PROBE_SCHEMA_VERSION,
        "probe_type": "resume",
        "created_at": utc_timestamp(),
        "status": "passed" if not failures and len(results) == len(PROTOCOLS) else "failed",
        "passed": bool(not failures and len(results) == len(PROTOCOLS)),
        "device": str(device),
        "comparison_mode": "finite_tolerance" if cuda else "bitwise",
        "resolved_config_hash": resolved.resolved_hash,
        "model_config_hash": resolved.model_hash,
        "git_sha": get_git_sha(),
        "environment_runtime_hash": environment["environment_runtime_hash"],
        "environment_report_hash": environment["environment_report_hash"],
        "protocol_results": results,
        "runtime_probe_failures": failures,
    }
    report["probe_hash"] = canonical_json_hash(
        {key: value for key, value in report.items() if key not in {"created_at", "probe_hash"}}
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment-report", required=True)
    parser.add_argument("--work-dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = run_resume_probe(
        args.config,
        environment_report_path=args.environment_report,
        work_dir=args.work_dir,
        allow_cpu=not args.require_cuda,
    )
    atomic_write_json(report, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
