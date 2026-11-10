#!/usr/bin/env bash
# Validate Oracle at the same 30% top-k setting on the other three tasks.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export BUDGETS=0.30 STRATEGIES=oracle RECENT_WINDOW=20
export OUTPUT_DIR="${OUTPUT_DIR:-results/neural_controls_validation_20261017}"
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
mkdir -p "${OUTPUT_DIR}/logs"

run_task() {
  local task_name="$1"
  local gpu_index="$2"
  for scope in shared per_head; do
    TASK="${task_name}" CUDA_VISIBLE_DEVICES="${gpu_index}" INDEX_SCOPE="${scope}" \
      bash run_neural_reviewer_controls.sh > "${OUTPUT_DIR}/logs/${task_name}_oracle_${scope}.log" 2>&1
  done
}
run_task openbookqa 0 &
open_pid=$!
run_task copa 1 &
copa_pid=$!
run_task rte 2 &
rte_pid=$!
status=0
for comparison_pid in "${open_pid}" "${copa_pid}" "${rte_pid}"; do
  if ! wait "${comparison_pid}"; then status=1; fi
done
echo "Other-task validation exit status: ${status}"
exit "${status}"
