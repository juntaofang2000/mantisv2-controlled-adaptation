#!/usr/bin/env bash
set -euo pipefail

task_file="${1:-configs/tasks.csv}"
output_root="${2:-results}"
task_count="$(( $(wc -l < "$task_file") - 1 ))"
if (( task_count < 1 )); then
  echo "No tasks found in $task_file" >&2
  exit 1
fi
mkdir -p logs

sbatch \
  --array="0-$((task_count - 1))" \
  --export="ALL,TASK_FILE=$task_file,OUTPUT_ROOT=$output_root" \
  scripts/slurm_array.sbatch
