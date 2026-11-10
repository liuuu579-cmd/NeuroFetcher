#!/usr/bin/env bash
# Add only three native baselines; avoid occupied GPU 0, use 30 of 64 CPU cores.
set -euo pipefail
cd /home/liushenyu/infinigen/accuracy/lm_eval
PYTHON_BIN=/home/liushenyu/anaconda3/envs/infinigen/bin/python
OUTPUT_DIR="${OUTPUT_DIR:-results/long_context_native_baselines_20261018}"
args=()
if [[ "${1:-}" == --smoke ]]; then
  args+=(--smoke)
  OUTPUT_DIR="${OUTPUT_DIR}_smoke"
fi
mkdir -p "$OUTPUT_DIR/logs"
exec 9>"$OUTPUT_DIR/.suite.lock"
flock -n 9 || { echo 'Suite already running'; exit 1; }
# Select only currently free GPUs 1–3; leave existing jobs alone.
mapfile -t free_gpus < <("$PYTHON_BIN" - <<'PY'
import csv, io, subprocess
gpus = dict(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)), skipinitialspace=True))
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
busy = {r[0] for r in csv.reader(io.StringIO(apps), skipinitialspace=True) if r}
for i in (1,2,3):
    if gpus[str(i)] not in busy:
        print(i)
PY
)
[[ "${#free_gpus[@]}" -gt 0 ]] || { echo 'No authorized free GPU'; exit 1; }
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
strategies=(infinigen h2o quant4)
run_worker() {
  local gpu="$1" offset="$2" index strategy worker_status=0
  for ((index=offset; index<3; index+=${#free_gpus[@]})); do
    strategy="${strategies[$index]}"
    env CUDA_VISIBLE_DEVICES="$gpu" taskset -c "$((gpu*10))-$((gpu*10+9))" \
    "$PYTHON_BIN" -u run_long_context_native_baselines.py --strategy "$strategy" \
    --output-dir "$OUTPUT_DIR" --threads 10 "${args[@]}" \
      >"$OUTPUT_DIR/logs/gpu${gpu}_${strategy}.log" 2>&1 || worker_status=1
  done
  return "$worker_status"
}
pids=()
for offset in "${!free_gpus[@]}"; do
  gpu="${free_gpus[$offset]}"
  run_worker "$gpu" "$offset" &
  pids+=("$!")
  echo "gpu=$gpu worker=$offset pid=$!"
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [[ "$status" == 0 && "${1:-}" != --smoke ]]; then
  "$PYTHON_BIN" summarize_long_context_native_baselines.py --output-dir "$OUTPUT_DIR" || status=1
fi
echo "$status" >"$OUTPUT_DIR/suite.exitcode"
echo "Native baseline suite exit=$status"
exit "$status"
