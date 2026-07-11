# Environment and offline assets

Exact CPU and intended CUDA environments live under `environment/`. The CPU lock was exercised locally; the CUDA definition must be validated on the target cluster. Every run records the selected lock-file SHA256 and runtime package/device report.

```bash
python -m image_transfer.scripts.inspect_environment \
  --lock environment/requirements-image-lock.txt \
  --out readiness/environment_report.json
```

`lock_matches_runtime` must be `true` before a run is submitted. The audit parses exact direct pins, reports every mismatch, and emits a stable `environment_runtime_hash`. GPU configurations reference `environment/image-transfer-cuda.yml`, so their provenance hashes the intended CUDA 12.8 definition rather than the locally tested CPU lock. This CUDA definition is intentionally described as unvalidated until the same audit passes on the target cluster.

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
