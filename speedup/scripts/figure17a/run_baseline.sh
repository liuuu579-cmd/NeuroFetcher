#!/bin/bash

echo "--- Running Baseline Performance Test (Full Cache Loading) ---"

# 设置 FlexGen 路径
FLEXGEN_PATH=$PWD/../../flexgen

# 1. 使用原始的 FlexGen 代码
# 为了得到纯净的基准性能，我们链接到 original/ 目录下的原始代码，
# 而不是您修改过的 infinigen/ 版本。
# 假设您的原始代码位于 ../original/ 目录下。
echo "Linking to original FlexGen source files..."
rm -f $FLEXGEN_PATH/flexgen/flex_opt.py
rm -f $FLEXGEN_PATH/flexgen/pytorch_backend.py
ln -s ../original/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
ln -s ../original/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py

# 2. 构建命令字符串 (核心参数保持一致)
# 我们保留了模型、路径、内存百分比、批处理大小和任务长度等核心参数，
# 以确保与您的预取策略进行公平的 apples-to-apples 对比。
CMD="--model facebook/opt-13b --path /home/liushenyu/opt-13b-flexgen-np/ --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 8 --num-gpu-batches 1 --prompt-len 1920 --gen-len 128 --warmup-input-path pg19_firstbook.txt --test-input-path pg19_firstbook.txt"

# 3. 移除所有与预取/推测相关的参数
# 基准测试不应包含任何特殊的缓存加载策略。
# 因此，我们不添加 --alpha, --partial-weight-ratio, --max-num-kv,
# 以及 --prefetch-strategy 等任何相关参数。
# FlexGen 在没有这些参数时，会默认执行全量缓存加载。

# 执行命令
echo "Executing FlexGen with baseline configuration..."
echo "Command: python -W ignore -m flexgen.flex_opt $CMD"
python -W ignore -m flexgen.flex_opt $CMD

# 恢复链接 (可选，但推荐)
# 运行结束后，您可以选择将链接指回您正在开发的版本。
echo "--- Baseline test finished. Restoring links to infinigen version. ---"
rm -f $FLEXGEN_PATH/flexgen/flex_opt.py
rm -f $FLEXGEN_PATH/flexgen/pytorch_backend.py
ln -s ../infinigen/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
ln -s ../infinigen/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py
echo "Links restored."