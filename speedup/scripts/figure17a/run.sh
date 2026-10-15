#!/bin/bash

# 设置 FlexGen 路径
FLEXGEN_PATH=$PWD/../../flexgen

# 创建软链接（您的原始逻辑）
rm -f $FLEXGEN_PATH/flexgen/flex_opt.py
rm -f $FLEXGEN_PATH/flexgen/pytorch_backend.py
ln -s ../infinigen/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
ln -s ../infinigen/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py
ALPHA=4
# 设置预测器路径 (只保留一个)
#PREDICTOR_PATH="/home/liushenyu/test/predictor_alpha_2_dyn_posw_epoch_20.pth"
PREDICTOR_PATH="/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth"
# 构建命令字符串
# 基础命令部分
CMD="--model facebook/opt-13b --path /home/liushenyu/opt-13b-flexgen-np/ --percent 100 0 0 100 100 0  --gpu-batch-size 20 --num-gpu-batches 1 --prompt-len 1920 --gen-len 128 --warmup-input-path pg19_firstbook.txt --test-input-path pg19_firstbook.txt"

# 拼接策略相关参数
# 注意: 已移除 --alpha $ALPHA，因为它在 neural 策略下不生效
#CMD=$CMD" --overlap false --alpha $ALPHA --partial-weight-ratio 0.2 --max-num-kv 409"
#CMD=$CMD" --overlap false --alpha $ALPHA --partial-weight-ratio 0.2 --max-num-kv 409"   #flexgen
CMD=$CMD" --overlap true --max-num-kv 400 --prefetch-strategy neural --predictor-path ${PREDICTOR_PATH} --prefetch-budget-ratio 0.2"

# 执行命令
python -W ignore -m flexgen.flex_opt $CMD
