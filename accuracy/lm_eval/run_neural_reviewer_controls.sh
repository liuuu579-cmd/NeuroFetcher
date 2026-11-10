#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-10}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
PYTHON_BIN="${PYTHON_BIN:-/home/liushenyu/anaconda3/envs/infinigen/bin/python}"

TASK="${TASK:-winogrande}"
SHOTS="${SHOTS:-5}"
MODEL_PATH="${MODEL_PATH:-/home/liushenyu/opt-13b}"
STRATEGIES="${STRATEGIES:-full neural oracle previous_step}"
BUDGETS="${BUDGETS:-0.30 0.20 0.15 0.08}"
INDEX_SCOPE="${INDEX_SCOPE:-shared}"
RECENT_WINDOW="${RECENT_WINDOW:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-results/neural_controls_v2}"
RESUME="${RESUME:-1}"
PREDICTOR_EMBEDDING_MODE="${PREDICTOR_EMBEDDING_MODE:-restore}"

for strategy in ${STRATEGIES}; do
  for budget in ${BUDGETS}; do
    printf -v tag '%g' "${budget}"
    tag="${tag//./p}"
    stem="${OUTPUT_DIR}/${TASK}-${SHOTS}-$(basename "${MODEL_PATH}")-${strategy}-${INDEX_SCOPE}-b${tag}-r${RECENT_WINDOW}"
    if [[ "${strategy}" == "neural" && "${PREDICTOR_EMBEDDING_MODE}" == "restore" ]]; then
      stem="${stem}-embedding-restored"
    fi
    if [[ "${RESUME}" == "1" && -s "${stem}.metrics.json" && -s "${stem}.accuracy.json" ]]; then
      echo "Complete; skip ${stem}"
    else
      if [[ ! -s "${stem}.metrics.json" ]]; then
        "${PYTHON_BIN}" -u neural_reviewer_controls.py \
          --task "${TASK}" --shots "${SHOTS}" --model-path "${MODEL_PATH}" \
          --strategy "${strategy}" --index-scope "${INDEX_SCOPE}" \
          --budget-ratio "${budget}" --recent-window "${RECENT_WINDOW}" \
          --output-dir "${OUTPUT_DIR}" \
          --predictor-embedding-mode "${PREDICTOR_EMBEDDING_MODE}" \
          --predictor-path "${PREDICTOR_PATH:-/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth}"
      fi
      "${PYTHON_BIN}" -u evaluate_reviewer_result.py \
        --result-file "${stem}.jsonl" --task-name "${TASK}" \
        --num-fewshot "${SHOTS}" --tokenizer-path "${MODEL_PATH}" --seq 1024
    fi
    # Full has no sparse budget; a single run is sufficient.
    if [[ "${strategy}" == "full" ]]; then break; fi
  done
done
