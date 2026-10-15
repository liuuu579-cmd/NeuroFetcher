#!/bin/bash

# (前面的参数定义不变)
task=$1
shots=$5
model_path=$2
model=$3
model_arch=$4
partial_weight=$6
alpha=$7
capacity=$8
budget=$9
prefetch_start_layer=${10}
prefetch_interval=${11}
prefetch_strategy=${12:-infinigen} 
prefetch_topk_percent=${13:-0.12}

base_name=$(basename "${model}")
predictor_model_path="/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth" 

# 根据策略打印信息
if [ "$prefetch_strategy" == "neural" ]; then
  echo "Running with NEURAL Prefetch Strategy:"
  echo "  - Predictor Path: $predictor_model_path"
  echo "  - Prefetch Top-K Percent: $prefetch_topk_percent"
elif [ "$prefetch_strategy" == "infinigen" ]; then
  echo "Running with INFINIGEN Prefetch Strategy:"
  echo "  - Alpha: $alpha"
  echo "  - Capacity: $capacity"
  echo "  - Budget: $budget"
else
  echo "Error: Unknown prefetch strategy '$prefetch_strategy'. Use 'neural' or 'infinigen'."
  exit 1
fi

if [ -z $no_skewing ]; then
  weight_path="../setup/weights/${base_name}_${partial_weight}"
else 
  weight_path="../setup/weights-no-skew/${base_name}_${partial_weight}"
fi
skewing_path="../setup/skewing_matrix/${base_name}.pt"
#--input-path results/${task}-${shots}.jsonl \
#--input-path results/c4_accuracy_task.jsonl \
# --- 关键修改在这里 ---
python -u run_lm_eval_harness.py \
  --input-path results/${task}-${shots}.jsonl \
  --output-path results/${task}-${shots}-${base_name}-ours.jsonl \
  --model-name ${model} \
  --model-type ${model_arch} \
  --ours \
  --model-path ${model_path} \
  \
  --prefetch-strategy ${prefetch_strategy} \
  --predictor-path ${predictor_model_path} \
  --prefetch-budget-ratio ${prefetch_topk_percent} \
  \
  --alpha ${alpha} \
  --capacity ${capacity} \
  --budget ${budget} \
  --prefetch-start-layer ${prefetch_start_layer} \
  --prefetch-interval ${prefetch_interval}

# --- 修改结束 ---
#python -u run_lm_eval_harness.py \
#  --input-path results/${task}-${shots}.jsonl \
#  --output-path results/${task}-${shots}-${base_name}-ours.jsonl \
#  --model-name ${model} \
#  --model-type ${model_arch} \
#  --ours \
#  --model-path ${model_path} \
#  \
#  --prefetch-strategy ${prefetch_strategy} \
#  --predictor-path ${predictor_model_path} \
#  --prefetch-budget-ratio ${prefetch_topk_percent} \
#  \
#  --partial_weight_ratio ${partial_weight} \
#  --partial_weight_path ${weight_path} \
#  --skewing_matrix_path ${skewing_path} \
#  --alpha ${alpha} \
#  --capacity ${capacity} \
#  --budget ${budget} \
#  --prefetch-start-layer ${prefetch_start_layer} \
#  --prefetch-interval ${prefetch_interval}
# (后续评估命令不变)
python -u evaluate_task_result.py \
  --result-file results/${task}-${shots}-${base_name}-ours.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --model-type ${model_arch}

#rm results/${task}-${shots}-${base_name}-ours.jsonl