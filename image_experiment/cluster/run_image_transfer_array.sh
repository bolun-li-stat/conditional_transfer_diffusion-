#!/bin/bash
#SBATCH --job-name=ctdiff_img
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --time=0-12:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 JOBS_CSV" >&2
  exit 2
fi

JOBS_CSV=$1
: "${PROJECT_ROOT:?export PROJECT_ROOT before submitting}"
: "${DATA_ROOT:?export DATA_ROOT before submitting}"
: "${RESULTS_ROOT:?export RESULTS_ROOT before submitting}"
: "${TORCH_HOME:?export TORCH_HOME before submitting}"
: "${METRIC_ASSETS_MANIFEST:?export METRIC_ASSETS_MANIFEST before submitting}"
: "${EXACT_ENVIRONMENT_LOCK_PATH:?export EXACT_ENVIRONMENT_LOCK_PATH before submitting}"
: "${ENVIRONMENT_RUNTIME_REPORT_PATH:?export ENVIRONMENT_RUNTIME_REPORT_PATH before submitting}"
: "${DATASET_IDENTITY_PATH:?export DATASET_IDENTITY_PATH before submitting}"
: "${GPU_RUNTIME_PROBE_PATH:?export GPU_RUNTIME_PROBE_PATH before submitting}"
: "${RESUME_PROBE_PATH:?export RESUME_PROBE_PATH before submitting}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as an array job}"

export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"
RUN_ARGS=(
  --jobs-csv "$JOBS_CSV"
  --job-index "$SLURM_ARRAY_TASK_ID"
  --device cuda
)
if [[ "${IMAGE_TRANSFER_RESUME:-0}" == "1" ]]; then
  RUN_ARGS+=(--resume)
fi
python -m image_transfer.scripts.run_job "${RUN_ARGS[@]}"
