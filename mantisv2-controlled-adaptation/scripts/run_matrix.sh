#!/usr/bin/env bash
set -euo pipefail

task_file="${1:-configs/tasks.csv}"
output_root="${2:-results}"
device="${DEVICE:-auto}"

task_count="$(( $(wc -l < "$task_file") - 1 ))"
failures=0
for ((task_index=0; task_index<task_count; task_index++)); do
  if ! python -m mantis_adaptation.run_one \
      --task-file "$task_file" \
      --task-index "$task_index" \
      --output-root "$output_root" \
      --device "$device"; then
    failures="$((failures + 1))"
  fi
done

if (( failures > 0 )); then
  echo "$failures task(s) failed; inspect their JSON records." >&2
  exit 1
fi
