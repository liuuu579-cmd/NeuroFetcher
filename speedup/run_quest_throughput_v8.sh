#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 {2|3} CPU_RANGE RUN_ID" >&2
  exit 2
fi
GPU="$1"
CPU_RANGE="$2"
RUN_ID="$3"
if [[ "$GPU" != 2 && "$GPU" != 3 ]]; then
  echo "Only GPUs 2 and 3 are authorized for this run" >&2
  exit 2
fi

ROOT="/home/liushenyu/infinigen"
OUT_DIR="$ROOT/speedup/results/native_flexgen_v8_${RUN_ID}/quest_native"
mkdir -p "$OUT_DIR"
for PROMPT_LEN in 512 1024 1536 1920; do
  STEM="quest_p${PROMPT_LEN}_g128_b4"
  echo "===== method=quest_native gpu=$GPU prompt=$PROMPT_LEN gen=128 batch=4 ====="
  taskset -c "$CPU_RANGE" env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    /home/liushenyu/anaconda3/envs/infinigen/bin/python -u \
    "$ROOT/speedup/run_quest_native_cached_v8.py" \
    --gpu "$GPU" --prompt-len "$PROMPT_LEN" --gen-len 128 \
    --batch 4 --threads 10 --page-size 16 --minimum-pages 3 \
    --output "$OUT_DIR/${STEM}.json" \
    2>&1 | tee "$OUT_DIR/${STEM}.log"
done
