# Environment and offline assets

Environment evidence has three distinct layers:

1. `environment/image-transfer-cuda.yml` describes the intended CUDA environment;
2. `freeze_environment` records the complete installed pip set and conda explicit URLs as an exact resolved lock;
3. `inspect_environment` records the actual Python, package, PyTorch, CUDA, GPU, and Git runtime and binds it to that exact lock.

The source YAML is never accepted as the exact resolved lock.

```bash
python -m image_transfer.scripts.freeze_environment \
  --source-spec environment/image-transfer-cuda.yml \
  --out readiness/environment_exact_lock.json
python -m image_transfer.scripts.inspect_environment \
  --lock readiness/environment_exact_lock.json \
  --source-spec environment/image-transfer-cuda.yml \
  --out readiness/environment_runtime_report.json \
  --require-cuda-exact
```

`lock_matches_runtime` must be `true` before submission. The report contains separate canonical hashes for runtime identity and the full report, and comparison rejects missing, changed, or unexpected packages. The exact lock and report must be generated on the target cluster after activating the final environment.

After the report passes, run the configured GPU load probe and real resume probe. The load probe exercises actual batches for both protocols with AMP, GradScaler, AdamW, clipping, and EMA; the resume probe rebuilds state and continues training after a checkpoint reload. CPU smoke output is never marked as GPU evidence.

GPU load reports use runtime-probe schema 4. For each protocol they record the CUDA driver's actual free memory before the step and after synchronization, together with the process peak allocated/reserved memory. `process_capacity_headroom_bytes = gpu_total_memory_bytes - peak_process_memory_reserved_bytes` is retained only as a process-capacity diagnostic; it is not actual system free memory. The conservative readiness quantity is

```text
estimated_minimum_free_during_step_bytes = max(
    0,
    min(
        actual_free_memory_after_step_bytes,
        actual_free_memory_before_bytes - peak_process_memory_reserved_bytes,
    ),
)
```

`training.minimum_gpu_headroom_bytes` is compared with that conservative free-memory estimate. This accounts for pre-existing and shared-node usage more safely than subtracting this process's peak reservation from total capacity. CPU probes record zero-valued memory fields with `gpu_memory_measurement_status: not_applicable`; they cannot satisfy release GPU evidence.

Metric weights are evaluation assets, not model initialization. Prepare them outside array workers and verify them offline. The asset manifest records relative filename, byte size, SHA256, and package versions. Missing or modified assets make strict preflight fail rather than trigger a compute-node download.

Use a dedicated `TORCH_HOME` and an explicit manifest path; filesystem roots and the repository root are rejected:

```bash
export TORCH_HOME="${METRIC_ASSET_ROOT:?set METRIC_ASSET_ROOT to a dedicated cache directory}"
export METRIC_ASSETS_MANIFEST="$TORCH_HOME/metric_assets_manifest.json"
python -m image_transfer.scripts.prepare_metric_assets \
  --asset-root "$TORCH_HOME" \
  --manifest "$METRIC_ASSETS_MANIFEST"
python -m image_transfer.scripts.prepare_metric_assets \
  --asset-root "$TORCH_HOME" \
  --manifest "$METRIC_ASSETS_MANIFEST" \
  --offline-check
```

The second command verifies the manifest self-hash, runtime package versions, Inception and ResNet files, then initializes both backends while socket connections are blocked. Run it from the same offline filesystem and environment used by workers.

Diffusion models always start from random initialization. Metric ResNet/Inception weights are used after training only and never guide sampling.
