#!/bin/bash

CWD=${PWD}

cd ${CWD}



# generate partial weight matrices for prediction
PARTIAL_RATIO=0.3
# opt
for size in 13b;do
  python gen_partial_weight.py \
    --our_model_path "./opt-model/opt-${size}" \
    --model "/home/liushenyu/opt-${size}" \
    --model_type "opt" \
    --partial_weight_ratio $PARTIAL_RATIO \
    --output "./weights"
done



