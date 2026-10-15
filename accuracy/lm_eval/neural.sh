#!/bin/bash

# ==============================================================================
# Script to run LM Evaluation Harness with the NEURAL Prefetching strategy.
#
# Usage:
# ./run_neural.sh [task] [model_path] [model] [model_arch] [shots] \
#                 [predictor_path] [prefetch_budget_ratio]
#
# Example:
# ./run_neural.sh c4_accuracy /path/to/opt-1.3b facebook/opt-1.3b opt 0 \
#                 /path/to/predictor.pt 0.3
# ==============================================================================

# --- 1. Assign arguments to readable variables ---
task=$1
model_path=$2
model=$3
model_arch=$4
shots=$5
predictor_path=$6
prefetch_budget_ratio=$7

# Check if essential arguments are provided
if [ -z "$task" ] || [ -z "$model" ] || [ -z "$predictor_path" ] || [ -z "$prefetch_budget_ratio" ]; then
  echo "Error: Missing required arguments."
  echo "Usage: $0 [task] [model_path] [model] [model_arch] [shots] [predictor_path] [prefetch_budget_ratio]"
  exit 1
fi

# --- 2. Set up paths and print configuration ---
base_name=$(basename "${model}")
output_file="results/${task}-${shots}-${base_name}-neural.jsonl"

echo "======================================================"
echo "Running with NEURAL Prefetch Strategy"
echo "------------------------------------------------------"
echo "           Task: ${task}"
echo "          Model: ${model}"
echo "   Predictor Path: ${predictor_path}"
echo " Budget Ratio (%): $(echo "${prefetch_budget_ratio} * 100" | bc)%"
echo "======================================================"

# --- 3. Inference: Run the main Python script ---
python -u run_lm_eval_harness.py \
  --input-path "results/${task}-${shots}.jsonl" \
  --output-path "${output_file}" \
  --model-name "${model}" \
  --model-path "${model_path}" \
  --model-type "${model_arch}" \
  --ours \
  --strategy neural \
  --predictor-path "${predictor_path}" \
  --prefetch-budget-ratio "${prefetch_budget_ratio}"

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