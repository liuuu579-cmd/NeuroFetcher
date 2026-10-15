#!/bin/bash

# ==============================================================================
# Script to run LM Evaluation Harness with the InfiniGen strategy.
#
# Usage:
# ./infinigen.sh [task] [model_path] [model] [model_arch] [shots] \
#                [partial_weight] [alpha] [capacity] [budget] [no_skewing (optional)]
# ==============================================================================

# --- 1. Assign arguments to readable variables ---
task=$1
model_path=$2
model=$3
model_arch=$4
shots=$5
partial_weight=$6
alpha=$7
capacity=$8
budget=$9
prefetch_start_layer=${10}
prefetch_interval=${11}

# --- 2. Set up paths ---
base_name=$(basename "${model}")
weight_path="../setup/weights/${base_name}_${partial_weight}"
skewing_path="../setup/skewing_matrix/${base_name}.pt"
output_file="results/${task}-${shots}-${base_name}-infinigen.jsonl"

echo "======================================================"
echo "Running with InfiniGen Strategy"
echo "------------------------------------------------------"
echo "           Task: ${task}"
echo "          Model: ${model}"
echo "    Weight Path: ${weight_path}"
echo "   Skewing Path: ${skewing_path}"
echo "          Alpha: ${alpha}"
echo "       Capacity: ${capacity}"
echo "         Budget: ${budget}"
echo "          Start: ${prefetch_start_layer}"
echo "       Interval: ${prefetch_interval}"
echo "======================================================"

# --- 3. Inference: Run the main Python script ---
python -u run_lm_eval_harness.py \
  --input-path "results/${task}-${shots}.jsonl" \
  --output-path "${output_file}" \
  --model-name "${model}" \
  --model-path "${model_path}" \
  --model-type "${model_arch}" \
  --ours \
  --strategy infinigen \
  --partial-weight-ratio "${partial_weight}" \
  --partial-weight-path "${weight_path}" \
  --skewing-matrix-path "${skewing_path}" \
  --alpha "${alpha}" \
  --capacity "${capacity}" \
  --budget "${budget}" \
  --prefetch-start-layer "${prefetch_start_layer}" \
  --prefetch-interval "${prefetch_interval}"

if [ $? -ne 0 ]; then
    echo "Error: Inference script failed. Aborting."
    exit 1
fi

# --- 4. Evaluate results ---
echo "Evaluating results for task: ${task}..."
python -u evaluate_task_result.py \
  --result-file "${output_file}" \
  --task-name "${task}" \
  --num-fewshot "${shots}" \
  --model-type "${model_arch}"

# --- 5. Clean up the temporary result file ---
echo "Cleaning up ${output_file}..."
rm "${output_file}"

echo "Done."