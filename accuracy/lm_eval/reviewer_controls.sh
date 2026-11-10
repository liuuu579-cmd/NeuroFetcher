#!/usr/bin/env bash
set -euo pipefail

# The first four visible GPUs are reserved for this experiment.  Forty-two
# worker threads is two thirds of the 64 logical CPUs on this host.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-42}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-42}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TASK="${TASK:-openbookqa}"
SHOTS="${SHOTS:-5}"
MODEL_PATH="${MODEL_PATH:-/home/liushenyu/opt-13b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/liushenyu/opt-13b}"
DATASETS_CACHE="${DATASETS_CACHE:-${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}}"
BUDGET_RATIO="${BUDGET_RATIO:-0.20}"
RECENT_WINDOW="${RECENT_WINDOW:-1}"
printf -v BUDGET_TAG '%g' "${BUDGET_RATIO}"
BUDGET_TAG="${BUDGET_TAG//./p}"
LIMIT="${LIMIT:-0}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-float16}"
echo "model=${MODEL_PATH} tokenizer=${TOKENIZER_PATH} gpus=${CUDA_VISIBLE_DEVICES} cpu_threads=${OMP_NUM_THREADS}"

usage() {
  echo "Usage: $0 [full|previous_step|oracle]"
  echo "Environment: TASK, SHOTS, MODEL_PATH, TOKENIZER_PATH, DATASETS_CACHE, BUDGET_RATIO, LIMIT, DEVICE, DTYPE"
}

STRATEGY="${1:-previous_step}"
case "${STRATEGY}" in
  full|previous_step|oracle) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

python -u reviewer_controls.py \
  --task "${TASK}" \
  --shots "${SHOTS}" \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --strategy "${STRATEGY}" \
  --budget-ratio "${BUDGET_RATIO}" \
  --recent-window "${RECENT_WINDOW}" \
  --limit "${LIMIT}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}"

# Accuracy is evaluated by the repository's existing task evaluator.  Keep
# the JSONL and metrics files so the reviewer can inspect timing and budgets.
RESULT_SUFFIX=""
if [[ "${STRATEGY}" != "full" ]]; then
  RESULT_SUFFIX="-b${BUDGET_TAG}"
fi
python -u evaluate_reviewer_result.py \
  --result-file "results/${TASK}-${SHOTS}-opt-13b-${STRATEGY}${RESULT_SUFFIX}.jsonl" \
  --task-name "${TASK}" \
  --num-fewshot "${SHOTS}" \
  --model-type opt \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --datasets-cache-dir "${DATASETS_CACHE}" \
  --seq 2048
