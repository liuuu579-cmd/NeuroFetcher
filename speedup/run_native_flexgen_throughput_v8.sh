#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 {infinigen|h2o} {2|3} CPU_RANGE RUN_ID" >&2
  exit 2
fi

METHOD="$1"
GPU="$2"
CPU_RANGE="$3"
RUN_ID="$4"
case "$METHOD:$GPU" in
  infinigen:2|h2o:3) ;;
  *) echo "Use InfiniGen on GPU 2 or H2O on GPU 3" >&2; exit 2 ;;
esac

ROOT="/home/liushenyu/infinigen"
OUT_DIR="$ROOT/speedup/results/native_flexgen_v8_${RUN_ID}/${METHOD}"
mkdir -p "$OUT_DIR"

for PROMPT_LEN in 512 1024 1536 1920; do
  STEM="${METHOD}_p${PROMPT_LEN}_g128_b4"
  echo "===== method=$METHOD gpu=$GPU prompt=$PROMPT_LEN gen=128 batch=4 ====="
  taskset -c "$CPU_RANGE" env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    /home/liushenyu/anaconda3/envs/infinigen/bin/python -u \
    "$ROOT/speedup/run_native_flexgen_benchmark_v8.py" \
    --method "$METHOD" --gpu "$GPU" --prompt-len "$PROMPT_LEN" \
    --gen-len 128 --batch 4 --threads 10 \
    --output "$OUT_DIR/${STEM}.json" \
    2>&1 | tee "$OUT_DIR/${STEM}.log"
done
