"""Exercise the configured training path at its declared GPU batch sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from image_transfer.config import load_resolved_config
from image_transfer.scripts.inspect_environment import inspect_environment, validate_environment_report
from image_transfer.training.runtime_probe import (
    GPU_MINIMUM_FREE_FORMULA,
    PROTOCOLS,
    RUNTIME_PROBE_SCHEMA_VERSION,
    run_load_probe,
)
from image_transfer.utils.io import atomic_write_json, canonical_json_hash, get_git_sha, load_json, utc_timestamp


def run_gpu_probe(
    config_path: str | Path,
    *,
    environment_report_path: str | Path | None = None,
    allow_cpu_smoke: bool = False,
) -> dict[str, Any]:
    resolved = load_resolved_config(config_path)
    config = resolved.resolved
    if environment_report_path is not None:
        environment = load_json(environment_report_path)
    else:
        lock = config.get("exact_environment_lock_path") or config.get("environment_lock_path")
        environment = inspect_environment(str(lock))
    validate_environment_report(environment)
    cuda = bool(torch.cuda.is_available())
    device = torch.device("cuda" if cuda else "cpu")
    failures: list[dict[str, str]] = []
    results: dict[str, Any] = {}
    if cuda or allow_cpu_smoke:
        for protocol in PROTOCOLS:
            try:
                results[protocol] = run_load_probe(config, device=device, protocol=protocol)
            except Exception as exception:
                failures.append(
                    {
                        "protocol": protocol,
                        "exception_type": type(exception).__name__,
                        "message": str(exception),
                        "resolved_config_hash": resolved.resolved_hash,
                    }
                )
    else:
        failures.append(
            {
                "protocol": "all",
                "exception_type": "CUDAUnavailable",
                "message": "CUDA is required for a release GPU probe",
                "resolved_config_hash": resolved.resolved_hash,
            }
        )
    minimum_headroom = int((config.get("training") or {}).get("minimum_gpu_headroom_bytes", 0))
    for protocol, result in results.items():
        if result.get("step_finite") is not True:
            failures.append(
                {
                    "protocol": protocol,
                    "exception_type": "NonFiniteTrainingStep",
                    "message": "loss or gradient norm was non-finite",
                    "resolved_config_hash": resolved.resolved_hash,
                }
            )
    if cuda:
        for protocol, result in results.items():
            estimated_free = int(result.get("estimated_minimum_free_during_step_bytes", -1))
            if estimated_free < minimum_headroom:
                failures.append(
                    {
                        "protocol": protocol,
                        "exception_type": "InsufficientGPUHeadroom",
                        "message": (
                            f"conservative free-memory estimate {estimated_free} bytes is below "
                            f"the configured minimum {minimum_headroom} bytes"
                        ),
                        "resolved_config_hash": resolved.resolved_hash,
                    }
                )
    report: dict[str, Any] = {
        "schema_version": RUNTIME_PROBE_SCHEMA_VERSION,
        "probe_type": "gpu_load",
        "created_at": utc_timestamp(),
        "status": "passed" if cuda and not failures and len(results) == len(PROTOCOLS) else (
            "cpu_smoke_only" if not cuda and allow_cpu_smoke and not failures else "failed"
        ),
        "passed": bool(cuda and not failures and len(results) == len(PROTOCOLS)),
        "cuda_required": True,
        "cuda_available": cuda,
        "resolved_config_hash": resolved.resolved_hash,
        "model_config_hash": resolved.model_hash,
        "git_sha": get_git_sha(),
        "environment_runtime_hash": environment["environment_runtime_hash"],
        "environment_report_hash": environment["environment_report_hash"],
        "minimum_gpu_headroom_bytes": minimum_headroom,
        "minimum_gpu_headroom_semantics": "estimated_minimum_free_during_step_bytes",
        "estimated_minimum_free_during_step_formula": GPU_MINIMUM_FREE_FORMULA,
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
    parser.add_argument("--environment-report")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args()
    report = run_gpu_probe(
        args.config,
        environment_report_path=args.environment_report,
        allow_cpu_smoke=args.allow_cpu_smoke,
    )
    atomic_write_json(report, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"] and not args.allow_cpu_smoke:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
