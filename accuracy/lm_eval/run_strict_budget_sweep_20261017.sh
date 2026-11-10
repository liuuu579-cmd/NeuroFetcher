#!/usr/bin/env bash
# Additive rerun using the corrected per-head definitions at other budgets.
set -euo pipefail
cd /home/liushenyu/infinigen/accuracy/lm_eval
PYTHON_BIN=/home/liushenyu/anaconda3/envs/infinigen/bin/python
ROOT=results/strict_budget_sweep_20261017
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/.suite.lock"
flock -n 9 || exit 1
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
worker() {
  local task="$1" gpu="$2" cpus="$3" ratio tag
  for ratio in 0.20 0.15 0.08; do
    tag="${ratio/./p}"
    "$PYTHON_BIN" -u strict_reviewer_controls.py --task "$task" --budget-ratio "$ratio" \
      --output-dir "$ROOT/b$tag" --threads 10
    local metrics result
    for metrics in "$ROOT/b$tag/$task-5-opt-13b-"*.metrics.json; do
      result="${metrics%.metrics.json}.jsonl"
      if [[ ! -s "${metrics%.metrics.json}.accuracy.json" ]]; then
        "$PYTHON_BIN" -u evaluate_reviewer_result.py --result-file "$result" --task-name "$task" \
          --num-fewshot 5 --tokenizer-path /home/liushenyu/opt-13b --seq 1024
      fi
    done
  done
}
export -f worker
export PYTHON_BIN ROOT
pids=()
tasks=(winogrande openbookqa copa rte)
for gpu in 0 1 2 3; do
  env CUDA_VISIBLE_DEVICES="$gpu" taskset -c "$((gpu * 10))-$((gpu * 10 + 9))" \
    bash -ec 'worker "$@"' _ "${tasks[$gpu]}" "$gpu" "$((gpu * 10))-$((gpu * 10 + 9))" \
    >"$ROOT/logs/${tasks[$gpu]}.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
echo "$status" > "$ROOT/suite.exitcode"
echo "Corrected budget sweep exit=$status"
exit "$status"
