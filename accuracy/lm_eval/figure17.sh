#!/bin/bash

shots=5
# Prepare dataset
#echo "prepare dataset"
#for task in winogrande openbookqa hellaswag mathqa lambada_openai ; do
#  python -u generate_task_data.py \
#  --output-file "results/${task}-${shots}.jsonl" \
#  --task-name ${task} \
#  --num-fewshot ${shots} 
#done


# Threshold (alpha) sweep
partial=0.3
capacity=1.0
budget=0.3
alpha=10000
start_layer=5


#echo "baseline  mathqa"
#for interval in 50; do
#  echo "interval ${interval} start_layer 5"
  #bash ours.sh openbookqa "../setup/opt-model/opt-13b" "/home/liushenyu/opt-13b" opt ${shots} ${partial} ${alpha} ${capacity} ${budget} 0 1
#done

echo "lambada_openai"
for task in winogrande  ; do
  #echo "baseline task ${task} start_layer 50"
  #bash full_cache.sh ${task} "/home/liushenyu/opt-13b/" opt ${shots}
  #bash ours.sh ${task} "../setup/opt-model/opt-13b" "/home/liushenyu/opt-13b" opt ${shots} ${partial} ${alpha} ${capacity} ${budget} 0 1 neural 1   #使用经infinigen初始化的模型
  bash neural.sh ${task} "/home/liushenyu/opt-13b" "/home/liushenyu/opt-13b" opt ${shots} ${partial} ${alpha} ${capacity} ${budget} 5 50 neural 0.3  #使用原模型
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

