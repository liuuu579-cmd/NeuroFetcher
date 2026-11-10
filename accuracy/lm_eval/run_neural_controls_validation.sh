#!/usr/bin/env bash
# Five Winogrande/30% comparisons, at most four GPU jobs and 40 CPU threads.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export TASK="${TASK:-winogrande}"
export BUDGETS=0.30
export RECENT_WINDOW=20
export OUTPUT_DIR="${OUTPUT_DIR:-results/neural_controls_validation_20261017}"
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
mkdir -p "${OUTPUT_DIR}/logs"

CUDA_VISIBLE_DEVICES=0 STRATEGIES=oracle INDEX_SCOPE=shared \
  bash run_neural_reviewer_controls.sh > "${OUTPUT_DIR}/logs/oracle_shared.log" 2>&1 &
oracle_shared_pid=$!
CUDA_VISIBLE_DEVICES=1 STRATEGIES=oracle INDEX_SCOPE=per_head \
  bash run_neural_reviewer_controls.sh > "${OUTPUT_DIR}/logs/oracle_per_head.log" 2>&1 &
oracle_head_pid=$!
CUDA_VISIBLE_DEVICES=2 STRATEGIES=neural INDEX_SCOPE=shared \
  bash run_neural_reviewer_controls.sh > "${OUTPUT_DIR}/logs/neural_shared.log" 2>&1 &
neural_pid=$!
CUDA_VISIBLE_DEVICES=3 STRATEGIES="full previous_step" INDEX_SCOPE=shared \
  bash run_neural_reviewer_controls.sh > "${OUTPUT_DIR}/logs/full_previous_step.log" 2>&1 &
previous_pid=$!

status=0
for comparison_pid in "${oracle_shared_pid}" "${oracle_head_pid}" "${neural_pid}" "${previous_pid}"; do
  if ! wait "${comparison_pid}"; then status=1; fi
done
echo "Validation exit status: ${status}; results: ${OUTPUT_DIR}"
exit "${status}"
