#!/usr/bin/env bash
# Four 13B workers, GPUs 0-3 only, 40/64 CPU cores total, no added recent-20.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/liushenyu/anaconda3/envs/infinigen/bin/python}"
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
OUTPUT_DIR="${OUTPUT_DIR:-results/strict_controls_20261017}"
mkdir -p "${OUTPUT_DIR}/logs"
exec 9>"${OUTPUT_DIR}/.suite.lock"
flock -n 9 || { echo "A suite already holds the output-directory lock" >&2; exit 1; }
export OUTPUT_DIR
worker() {
  local task_name="$1" gpu_index="$2" cpu_range="$3"
  CUDA_VISIBLE_DEVICES="${gpu_index}" taskset -c "${cpu_range}" "${PYTHON_BIN}" -u strict_reviewer_controls.py \
    --task "${task_name}" --output-dir "${OUTPUT_DIR}" --threads 10
  local result_path
  for result_path in "${OUTPUT_DIR}/${task_name}-5-opt-13b-"*.metrics.json; do
    [[ -f "${result_path}" ]] || continue
    local jsonl_path="${result_path%.metrics.json}.jsonl"
    local accuracy_path="${result_path%.metrics.json}.accuracy.json"
    if [[ ! -s "${accuracy_path}" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu_index}" taskset -c "${cpu_range}" "${PYTHON_BIN}" -u evaluate_reviewer_result.py \
        --result-file "${jsonl_path}" --task-name "${task_name}" --num-fewshot 5 \
        --tokenizer-path /home/liushenyu/opt-13b --seq 1024
    fi
  done
}
worker winogrande 0 0-9 > "${OUTPUT_DIR}/logs/winogrande.log" 2>&1 &
wino_pid=$!
worker openbookqa 1 10-19 > "${OUTPUT_DIR}/logs/openbookqa.log" 2>&1 &
open_pid=$!
worker copa 2 20-29 > "${OUTPUT_DIR}/logs/copa.log" 2>&1 &
copa_pid=$!
worker rte 3 30-39 > "${OUTPUT_DIR}/logs/rte.log" 2>&1 &
rte_pid=$!
status=0
for worker_pid in "${wino_pid}" "${open_pid}" "${copa_pid}" "${rte_pid}"; do
  if ! wait "${worker_pid}"; then status=1; fi
done
echo "Strict reviewer controls suite exit status: ${status}; output=${OUTPUT_DIR}"
exit "${status}"
