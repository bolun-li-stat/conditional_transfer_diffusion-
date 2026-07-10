#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=ctdiff_img
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx8000
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --time=0-12:00
#SBATCH --output=/burg/stats/users/bl3147/conditional_transfer_diffusion/logs/%x_%A_%a.out
#SBATCH --error=/burg/stats/users/bl3147/conditional_transfer_diffusion/logs/%x_%A_%a.err
set -euo pipefail
JOBS_CSV=$1
module purge
module load anaconda/3-2023.09
module load cuda12.0/toolkit || true
source activate ctdiff_img
export PROJECT_ROOT=${PROJECT_ROOT:-/burg/stats/users/bl3147/conditional_transfer_diffusion/repo}
export DATA_ROOT=${DATA_ROOT:-/burg/stats/users/bl3147/conditional_transfer_diffusion/data}
export RESULTS_ROOT=${RESULTS_ROOT:-/burg/stats/users/bl3147/conditional_transfer_diffusion/results}
export TORCH_HOME=${TORCH_HOME:-/burg/stats/users/bl3147/conditional_transfer_diffusion/cache/torch}
export HF_HOME=${HF_HOME:-/burg/stats/users/bl3147/conditional_transfer_diffusion/cache/hf}
export PYTHONUNBUFFERED=1
cd $PROJECT_ROOT
python -m image_transfer.scripts.run_job --jobs-csv ${JOBS_CSV} --job-index ${SLURM_ARRAY_TASK_ID}
