#!/bin/bash

## =========================================================================
##  保留原有的 InfiniGen 测试 (适配新参数)
## =========================================================================
#echo "== InfiniGen =="
#partial=0.2
#seqlen=2048
#alpha=4.0
#budget=0.2
#echo opt-13b alpha $alpha budget $budget
#python opt.py --model "../setup/opt-model/opt-13b" \
#  --eval_dataset "c4" \
#  --seq_len ${seqlen} \
#  --eval_samples 50 \
#  --model_name "/home/liushenyu/opt-13b/" \
#  --print_blk_ppl \
#  --ours --strategy infinigen \
#  --partial_weight_ratio ${partial} \
#  --partial_weight_path "../setup/weights/opt-13b_${partial}" \
#  --alpha ${alpha} \
#  --budget ${budget} \
#  --capacity 1.0
#
#
## =========================================================================
##  保留原有的 H2O 测试 (适配新参数)
## =========================================================================
#echo "==    H2O   =="
#partial=0.2
#seqlen=2048
#heavy=0.01875
#recent=0.01875
#echo opt-13b heavy $heavy recent $recent
#python opt.py --model "/home/liushenyu/opt-13b/" \
#  --eval_dataset "c4" \
#  --seq_len ${seqlen} \
#  --eval_samples 50 \
#  --model_name "/home/liushenyu/opt-13b/" \
#  --print_blk_ppl \
#  --enable-small-cache \
#  --heavy_ratio ${heavy} \
#  --recent_ratio ${recent}


# =========================================================================
#  新增：Neural Prefetcher 测试
# =========================================================================
echo "==   Neural  =="
seqlen=2048
budget_ratio=0.4

#predictor_path="/home/liushenyu/test/predictor_alpha_2_dyn_posw_epoch_20.pth"
predictor_path="/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth"
echo opt-13b neural budget ${budget_ratio}
python opt.py --model "/home/liushenyu/opt-13b/" \
  --eval_dataset "c4" \
  --seq_len ${seqlen} \
  --eval_samples 50 \
  --model_name "/home/liushenyu/opt-13b/" \
  --print_blk_ppl \
  --ours --strategy neural \
  --predictor-path "${predictor_path}" \
  --prefetch-budget-ratio ${budget_ratio}


## =========================================================================
##  新增：Full Cache (Baseline) 测试
## =========================================================================
#echo "== Full Cache =="
#seqlen=2048
#echo opt-13b full cache
#python opt.py --model "/home/liushenyu/opt-13b/" \
#  --eval_dataset "c4" \
#  --seq_len ${seqlen} \
#  --eval_samples 50 \
#  --model_name "/home/liushenyu/opt-13b/" \
#  --print_blk_ppl

#
## =========================================================================
##  保留原有的 InfiniGen 测试 (适配新参数)
## =========================================================================
#echo "== InfiniGen =="
#partial=0.2
#seqlen=2048
#alpha=4.0
#budget=0.2
#echo opt-13b alpha $alpha budget $budget
#python opt.py --model "../setup/opt-model/opt-13b" \
#  --eval_dataset "pg19" \
#  --seq_len ${seqlen} \
#  --eval_samples 50 \
#  --model_name "/home/liushenyu/opt-13b/" \
#  --print_blk_ppl \
#  --ours --strategy infinigen \
#  --partial_weight_ratio ${partial} \
#  --partial_weight_path "../setup/weights/opt-13b_${partial}" \
#  --alpha ${alpha} \
#  --budget ${budget} \
#  --capacity 1.0
#
#
## =========================================================================
##  保留原有的 H2O 测试 (适配新参数)
## =========================================================================
#echo "==    H2O   =="
#partial=0.2
#seqlen=2048
#heavy=0.01875
#recent=0.01875
#echo opt-13b heavy $heavy recent $recent
#python opt.py --model "/home/liushenyu/opt-13b/" \
#  --eval_dataset "pg19" \
#  --seq_len ${seqlen} \
#  --eval_samples 50 \
#  --model_name "/home/liushenyu/opt-13b/" \
#  --print_blk_ppl \
#  --enable-small-cache \
#  --heavy_ratio ${heavy} \
#  --recent_ratio ${recent}
#
#
## =========================================================================
##  新增：Neural Prefetcher 测试
## =========================================================================
#echo "==   Neural  =="
#seqlen=2048
#budget_ratio=0.4
#
##predictor_path="/home/liushenyu/test/predictor_alpha_2_dyn_posw_epoch_20.pth"
#predictor_path="/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth"
#echo opt-13b neural budget ${budget_ratio}
#python opt.py --model "/home/liushenyu/opt-13b/" \
#  --eval_dataset "pg19" \
#  --seq_len ${seqlen} \
#  --eval_samples 50 \
#  --model_name "/home/liushenyu/opt-13b/" \
#  --print_blk_ppl \
#  --ours --strategy neural \
#  --predictor-path "${predictor_path}" \
#  --prefetch-budget-ratio ${budget_ratio}
#
#
## =========================================================================
##  新增：Full Cache (Baseline) 测试
## =========================================================================
#echo "== Full Cache =="
#seqlen=2048
#echo opt-13b full cache
#python opt.py --model "/home/liushenyu/opt-13b/" \
#  --eval_dataset "pg19" \
#  --seq_len ${seqlen} \
#  --eval_samples 50 \
#  --model_name "/home/liushenyu/opt-13b/" \
#  --print_blk_ppl

