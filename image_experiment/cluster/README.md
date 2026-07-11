# Cluster launch

Run all preparation commands from the `image_experiment` module root. `PROJECT_ROOT` must point to that directory, not the repository root.

Before creating a runnable grid, export the real data, result, metric-asset, and evidence paths described in [`docs/EXPERIMENT_READINESS.md`](../docs/EXPERIMENT_READINESS.md). Then generate the grid and inspect its count and resource estimate:

```bash
python -m image_transfer.scripts.make_job_grid \
  --experiment A \
  --config image_transfer/configs/imagenet64_release_pilot.yaml \
  --out "$RESULTS_ROOT/release_jobs.csv" \
  --max-jobs 12
```

The release grid must contain 12 rows and every row must have `runnable=true` before submission.

Initial launch:

```bash
mkdir -p logs
sbatch --array=0-11 cluster/run_image_transfer_array.sh "$RESULTS_ROOT/release_jobs.csv"
```

To continue interrupted rows from checkpoints, explicitly set the resume switch for the resubmission:

```bash
export IMAGE_TRANSFER_RESUME=1
sbatch --array=0-11 cluster/run_image_transfer_array.sh "$RESULTS_ROOT/release_jobs.csv"
```

The launcher never enables resume on an initial submission and never supplies `--force`. It does not select an account, partition, environment module, or storage location; review those cluster-specific choices before submitting.
