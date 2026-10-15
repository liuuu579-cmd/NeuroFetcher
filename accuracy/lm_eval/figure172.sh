#!/bin/bash

shots=5
# Prepare dataset
echo "prepare dataset"
for task in lambada_openai; do
  python -u generate_task_data.py \
  --output-file "results/${task}-${shots}.jsonl" \
  --task-name ${task} \
  --num-fewshot ${shots} 
done


# Threshold (alpha) sweep
partial=0.2
capacity=1.0
budget=0.2
alpha=5
start_layer=5


echo "baseline  lambada_openai"
for interval in 1 2 3 4 5 50; do
  echo "interval ${interval} start_layer 5"
  bash ours.sh lambada_openai "../setup/opt-model/opt-13b" "/home/liushenyu/opt-13b" opt ${shots} ${partial} ${alpha} ${capacity} ${budget} 5 ${interval}
done



# Partial weight sweep
# alpha=4
# capacity=1.0
# budget=0.2

# echo "Partial weight ratio sweep"
# for partial in 0.1 0.2 0.4 0.6 0.8 1; do
#   echo partial ratio "${partial}"
#   bash ours.sh winogrande "../setup/opt-model/opt-13b" "/home/liushenyu/opt-13b" opt ${shots} ${partial} ${alpha} ${capacity} ${budget}
# done

