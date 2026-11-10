#!/usr/bin/env bash
# Uses GPUs 0-3 only, 10 CPU cores per worker, and preserves per-worker logs.
set -euo pipefail
cd /home/liushenyu/infinigen/accuracy/lm_eval
PYTHON_BIN=/home/liushenyu/anaconda3/envs/infinigen/bin/python
INPUT_PATH="${INPUT_PATH:-results/long_context_retrieval_combined_20261017.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/long_context_controls_balanced_20261017}"
mkdir -p "$OUTPUT_DIR/logs"
exec 9>"$OUTPUT_DIR/.suite.lock"
flock -n 9 || { echo 'Another suite holds the directory lock'; exit 1; }
[[ -s "$INPUT_PATH" ]] || { echo "Missing input: $INPUT_PATH"; exit 1; }
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
run_one() {
  local gpu="$1" strategy="$2"
  env CUDA_VISIBLE_DEVICES="$gpu" taskset -c "$((gpu * 10))-$((gpu * 10 + 9))" \
    "$PYTHON_BIN" -u run_long_context_controls.py --input "$INPUT_PATH" \
    --output-dir "$OUTPUT_DIR" --strategies "$strategy" --threads 10
}
export -f run_one
export PYTHON_BIN INPUT_PATH OUTPUT_DIR
# Each worker runs one strategy because neural_offline uses a separate model implementation.
strategies=(full oracle previous_step layer5_reuse quest_capped quest_native neural_offline)
run_wave() {
  local start="$1" end="$2" wave="$3"; local pids=(); local gpu strategy
  for ((gpu=0; gpu<4 && start+gpu<end; gpu++)); do
    strategy="${strategies[$((start + gpu))]}"
    echo "wave=$wave gpu=$gpu strategy=$strategy"
    (run_one "$gpu" "$strategy") >"$OUTPUT_DIR/logs/wave${wave}_gpu${gpu}_${strategy}.log" 2>&1 &
    pids+=("$!")
  done
  local status=0 pid
  for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
  return "$status"
}
status=0
run_wave 0 4 1 || status=1
run_wave 4 7 2 || status=1
if [[ "$status" == 0 ]]; then
  "$PYTHON_BIN" summarize_long_context_controls.py --input "$INPUT_PATH" --output-dir "$OUTPUT_DIR" || status=1
fi
echo "$status" > "$OUTPUT_DIR/suite.exitcode"
echo "Long-context suite exit=$status"
exit "$status"
