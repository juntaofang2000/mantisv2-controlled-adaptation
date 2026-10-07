#!/usr/bin/env bash
#SBATCH --job-name=mantisv2-adapt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

set -euo pipefail
mkdir -p logs

python -m mantis_adaptation.run_one \
  --task-file "${TASK_FILE}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda

