#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-42}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-42}"
RESUME="${RESUME:-0}"

TASKS=("${TASKS:-openbookqa winogrande copa rte}")
SHOTS="${SHOTS:-5}"
IFS=' ' read -r -a TASK_LIST <<< "${TASKS[0]}"
BUDGETS=("${BUDGETS:-0.30 0.20 0.15 0.08}")
IFS=' ' read -r -a BUDGET_LIST <<< "${BUDGETS[0]}"

run_condition() {
  local task="$1"
  local strategy="$2"
  local budget="${3:-}"
  local tag=""
  local suffix=""

  if [[ "${strategy}" != "full" ]]; then
    printf -v tag '%g' "${budget}"
    tag="${tag//./p}"
    suffix="-b${tag}"
  fi

  if [[ "${RESUME}" == "1" ]]; then
    local stem="results/${task}-${SHOTS}-opt-13b-${strategy}${suffix}"
    if [[ -s "${stem}.jsonl" && -s "${stem}.metrics.json" && -s "${stem}.accuracy.json" ]]; then
      echo "===== task=${task} strategy=${strategy}${budget:+ budget=${budget}} already complete; skipping ====="
      return
    fi
  fi

  if [[ "${strategy}" == "full" ]]; then
    echo "===== task=${task} strategy=full ====="
    TASK="${task}" bash reviewer_controls.sh full
  else
    echo "===== task=${task} strategy=${strategy} budget=${budget} ====="
    TASK="${task}" BUDGET_RATIO="${budget}" bash reviewer_controls.sh "${strategy}"
  fi
}

for task in "${TASK_LIST[@]}"; do
  run_condition "${task}" full
  for strategy in previous_step oracle; do
    for budget in "${BUDGET_LIST[@]}"; do
      run_condition "${task}" "${strategy}" "${budget}"
    done
  done
done
