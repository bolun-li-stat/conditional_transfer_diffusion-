# Image transfer diffusion experiments

This module studies positive and negative transfer in class-conditional image DDPMs without changing the existing Gaussian-mixture pipeline.

## Research question

For a fixed target class `c0`, compare a target-only unconditional DDPM trained on `n0` target images with a class-conditional DDPM trained on the same target images plus auxiliary classes, then sample and evaluate only at `y=c0`.

## Experiments

- **Experiment A: equal target sample size.** Keeps target data fixed and compares `unconditional_n0`, `conditional_close`, `conditional_medium`, `conditional_far`, and `conditional_mix`.
- **Experiment B: equal total sample size.** Compares each conditional setting against `unconditional_equal_total` trained on `N_total = n0 + K_aux*m_per_aux` target images. If insufficient target images are available, the baseline is skipped and recorded.
- **Experiment C: auxiliary similarity/composition sweep.** Keeps total auxiliary budget fixed and evaluates `close_only`, `mostly_close`, `balanced_mix`, `mostly_far`, and `far_only`.

Experiment B and C are not run automatically after A. Generate separate job CSV files, or pass `--experiment all` when you intentionally want all grids.

## CIFAR sanity check

```bash
python -m image_transfer.scripts.train_one \
  --config image_transfer/configs/cifar10_sanity.yaml \
  --experiment A \
  --max-steps 2 \
  --n0 8 \
  --m-per-aux 8 \
  --num-generated 4 \
  --device cpu
```

Generate job grids:

```bash
python -m image_transfer.scripts.make_job_grid --experiment A --config image_transfer/configs/cifar10_sanity.yaml --out image_transfer_results/A_equal_target/jobs/cifar_A_jobs.csv
python -m image_transfer.scripts.make_job_grid --experiment B --config image_transfer/configs/cifar10_sanity.yaml --out image_transfer_results/B_equal_total/jobs/cifar_B_jobs.csv
python -m image_transfer.scripts.make_job_grid --experiment C --config image_transfer/configs/cifar10_sanity.yaml --out image_transfer_results/C_similarity_sweep/jobs/cifar_C_jobs.csv
```

Run one generated job:

```bash
python -m image_transfer.scripts.run_job --jobs-csv image_transfer_results/A_equal_target/jobs/cifar_A_jobs.csv --job-index 0
```

## ImageNet layout

ImageNet is not downloaded automatically. Place or symlink ILSVRC2012 data as:

```text
DATA_ROOT/
  train/
    n02108915/
    ...
  val/
    n02108915/
    ...
```

Required synsets are checked explicitly. Missing synsets raise an error and are not silently ignored. Use symlinked subset directories to avoid copying images.

Primary target: French bulldog (`n02108915`). The pilot config fixes close, medium, far, and mix auxiliary sets before observing FID/KID results.

## Output layout

All outputs go under `output_root`/`RESULTS_ROOT`:

```text
$RESULTS_ROOT/
  A_equal_target/{jobs,checkpoints,logs,samples,metrics.csv,figures}
  B_equal_total/{jobs,checkpoints,logs,samples,metrics.csv,figures}
  C_similarity_sweep/{jobs,checkpoints,logs,samples,metrics.csv,figures}
  all_metrics.csv
```

## Ginsburg setup

Scratch storage is not backed up. Do not commit ImageNet data, checkpoints, samples, or private credentials.

```bash
ssh bl3147@ginsburg.rcs.columbia.edu
mkdir -p /burg/stats/users/bl3147/conditional_transfer_diffusion/{repo,data,results,logs,cache}
cd /burg/stats/users/bl3147/conditional_transfer_diffusion/repo
git clone https://github.com/bolun-li-stat/conditional_transfer_diffusion-.git .
module load anaconda/3-2023.09
conda create -n ctdiff_img python=3.11 -y
conda activate ctdiff_img
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-image.txt
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

Interactive GPU test:

```bash
srun --pty -t 0-01:00 --gres=gpu:1 -A stats /bin/bash
```

## Run A first, then B and C later

```bash
export PROJECT_ROOT=/burg/stats/users/bl3147/conditional_transfer_diffusion/repo
export RESULTS_ROOT=/burg/stats/users/bl3147/conditional_transfer_diffusion/results
cd $PROJECT_ROOT
python -m image_transfer.scripts.make_job_grid \
  --experiment A \
  --config image_transfer/configs/imagenet64_main_grid.yaml \
  --out $RESULTS_ROOT/A_equal_target/jobs/imagenet64_A_jobs.csv
N_A=$(($(wc -l < $RESULTS_ROOT/A_equal_target/jobs/imagenet64_A_jobs.csv)-2))
sbatch --array=0-${N_A} scripts/ginsburg/run_image_transfer_array.sh \
  $RESULTS_ROOT/A_equal_target/jobs/imagenet64_A_jobs.csv
```

Later:

```bash
python -m image_transfer.scripts.make_job_grid \
  --experiment B \
  --config image_transfer/configs/imagenet64_main_grid.yaml \
  --out $RESULTS_ROOT/B_equal_total/jobs/imagenet64_B_jobs.csv
N_B=$(($(wc -l < $RESULTS_ROOT/B_equal_total/jobs/imagenet64_B_jobs.csv)-2))
sbatch --array=0-${N_B} scripts/ginsburg/run_image_transfer_array.sh \
  $RESULTS_ROOT/B_equal_total/jobs/imagenet64_B_jobs.csv

python -m image_transfer.scripts.make_job_grid \
  --experiment C \
  --config image_transfer/configs/imagenet64_main_grid.yaml \
  --out $RESULTS_ROOT/C_similarity_sweep/jobs/imagenet64_C_jobs.csv
N_C=$(($(wc -l < $RESULTS_ROOT/C_similarity_sweep/jobs/imagenet64_C_jobs.csv)-2))
sbatch --array=0-${N_C} scripts/ginsburg/run_image_transfer_array.sh \
  $RESULTS_ROOT/C_similarity_sweep/jobs/imagenet64_C_jobs.csv
```

Monitor:

```bash
squeue -u bl3147
tail -f /burg/stats/users/bl3147/conditional_transfer_diffusion/logs/<logfile>
```

## Metrics and plots

Each run appends `metrics.csv` with target denoising MSE by noise bin, FID/KID placeholders or computed values, classifier target accuracy, leakage rate, sample count, checkpoint path, and skip metadata. Plot with:

```bash
python -m image_transfer.scripts.plot_results --results-root $RESULTS_ROOT --experiment A
python -m image_transfer.scripts.plot_results --results-root $RESULTS_ROOT --experiment B
python -m image_transfer.scripts.plot_results --results-root $RESULTS_ROOT --experiment C
```

Close/medium/far auxiliary sets are fixed semantic groups. Mix uses the same total auxiliary budget as close/medium/far by default, with `2 close + 1 medium + 2 far` when `K_aux=5`.
