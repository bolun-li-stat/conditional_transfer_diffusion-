# Experiment readiness

Run these gates before submitting real-data jobs:

1. Create the exact environment and record its lock hash.
2. Prepare metric assets on a network-enabled login node.
3. Run the offline asset check on the compute-node filesystem.
4. Run `preflight_experiment` against the intended configuration.
5. Complete the small real-data release pilot.
6. Run `validate_release_pilot` and inspect every failure.
7. Only then generate a main-stage grid.

```bash
export METRIC_ASSETS_MANIFEST="$TORCH_HOME/metric_assets_manifest.json"
python -m image_transfer.scripts.prepare_metric_assets \
  --asset-root "$TORCH_HOME" --manifest "$METRIC_ASSETS_MANIFEST"
python -m image_transfer.scripts.prepare_metric_assets \
  --asset-root "$TORCH_HOME" --manifest "$METRIC_ASSETS_MANIFEST" --offline-check
python -m image_transfer.scripts.preflight_experiment \
  --config image_transfer/configs/imagenet64_release_pilot.yaml \
  --out-dir readiness
```

`readiness/pilot_status.json` is checked in as `not_run`. A real validation run is the only operation that may write `passed`. That status means engineering health only; it does not support a scientific conclusion. Main-stage configs reject a missing, failed, or provenance-mismatched status unless the explicit override is used and recorded.

The current repository state is code-ready for a real-data release pilot only. No full ImageNet run, real-data metric improvement, or general transfer conclusion is claimed.
