# Experiment readiness

Run these gates before submitting real-data jobs:

1. Activate the target CUDA environment, freeze its exact package/build lock, and capture the actual runtime report.
2. Freeze and verify the real dataset content identity.
3. Prepare metric assets on a network-enabled login node and verify them offline on the compute-node filesystem.
4. Run the configured GPU load probe and save→destroy→rebuild→load→continue resume probe.
5. Run `preflight_experiment` against the intended configuration.
6. Complete the small real-data release pilot.
7. Run `validate_release_pilot` and inspect every failure and artifact count.
8. Only then generate a main-stage grid.

```bash
export METRIC_ASSETS_MANIFEST="$TORCH_HOME/metric_assets_manifest.json"
export EXACT_ENVIRONMENT_LOCK_PATH="$RESULTS_ROOT/readiness/environment_exact_lock.json"
export ENVIRONMENT_RUNTIME_REPORT_PATH="$RESULTS_ROOT/readiness/environment_runtime_report.json"
export DATASET_IDENTITY_PATH="$RESULTS_ROOT/readiness/dataset_identity.json"
export GPU_RUNTIME_PROBE_PATH="$RESULTS_ROOT/readiness/gpu_runtime_probe.json"
export RESUME_PROBE_PATH="$RESULTS_ROOT/readiness/resume_probe.json"
python -m image_transfer.scripts.freeze_environment \
  --source-spec environment/image-transfer-cuda.yml --out "$EXACT_ENVIRONMENT_LOCK_PATH"
python -m image_transfer.scripts.inspect_environment \
  --lock "$EXACT_ENVIRONMENT_LOCK_PATH" \
  --source-spec environment/image-transfer-cuda.yml \
  --out "$ENVIRONMENT_RUNTIME_REPORT_PATH" --require-cuda-exact
python -m image_transfer.scripts.freeze_dataset_identity \
  --config image_transfer/configs/imagenet64_release_pilot.yaml \
  --out "$DATASET_IDENTITY_PATH"
python -m image_transfer.scripts.prepare_metric_assets \
  --asset-root "$TORCH_HOME" --manifest "$METRIC_ASSETS_MANIFEST"
python -m image_transfer.scripts.prepare_metric_assets \
  --asset-root "$TORCH_HOME" --manifest "$METRIC_ASSETS_MANIFEST" --offline-check
python -m image_transfer.scripts.gpu_runtime_probe \
  --config image_transfer/configs/imagenet64_release_pilot.yaml \
  --environment-report "$ENVIRONMENT_RUNTIME_REPORT_PATH" --out "$GPU_RUNTIME_PROBE_PATH"
python -m image_transfer.scripts.run_resume_probe \
  --config image_transfer/configs/imagenet64_release_pilot.yaml \
  --environment-report "$ENVIRONMENT_RUNTIME_REPORT_PATH" \
  --out "$RESUME_PROBE_PATH" --require-cuda
python -m image_transfer.scripts.preflight_experiment \
  --config image_transfer/configs/imagenet64_release_pilot.yaml \
  --out-dir readiness
```

`readiness/pilot_status.json` is checked in with schema 3 and status `not_run`. A real validation run is the only operation that may write `passed`. The validator requires exact environment/runtime hashes, dataset identity, GPU load and resume probes, every expected result and pair, and checkpoint/sample/metric/figure/provenance artifacts. That status means engineering health only; it does not support a scientific conclusion. Main-stage configs reject a missing, failed, or provenance-mismatched status unless the explicit override is used and recorded.

The GPU load evidence itself uses runtime-probe schema 4. The validator requires actual free-memory snapshots from `torch.cuda.mem_get_info()`, process peak allocated/reserved memory, the separate process-capacity diagnostic, and the conservative `estimated_minimum_free_during_step_bytes`. The configured minimum-memory gate applies only to that conservative estimate. Older probe schemas or reports that expose only total-minus-process-reservation cannot pass current readiness validation.

The current repository state is code-ready for target-cluster GPU and real-data release-pilot validation only. No real GPU probe, GPU resume probe, full ImageNet run, real-data metric improvement, or general transfer conclusion is claimed.
