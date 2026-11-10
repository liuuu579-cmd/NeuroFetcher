#!/usr/bin/env bash
# Explicit diagnostic: restore the documented predictor embedding after loading.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
export TASK=winogrande BUDGETS=0.30 STRATEGIES=neural INDEX_SCOPE=shared
export OUTPUT_DIR=results/neural_controls_validation_20261017
export PREDICTOR_EMBEDDING_MODE=restore
mkdir -p "${OUTPUT_DIR}/logs"
bash run_neural_reviewer_controls.sh > "${OUTPUT_DIR}/logs/neural_embedding_restored.log" 2>&1
