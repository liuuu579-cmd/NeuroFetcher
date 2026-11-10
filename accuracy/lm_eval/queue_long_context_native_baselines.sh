#!/usr/bin/env bash
# Wait without reserving memory or interfering with existing GPU processes.
set -euo pipefail
cd /home/liushenyu/infinigen/accuracy/lm_eval
OUTPUT_DIR="${OUTPUT_DIR:-results/long_context_native_baselines_20261018}"
mkdir -p "$OUTPUT_DIR"
exec 8>"$OUTPUT_DIR/.queue.lock"
flock -n 8 || { echo 'Queue already active'; exit 1; }
export OUTPUT_DIR
python_bin=/home/liushenyu/anaconda3/envs/infinigen/bin/python
stage=smoke
while true; do
  if "$python_bin" - <<'PY'
import csv, io, subprocess
gpus=dict(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)),skipinitialspace=True))
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
busy={x.strip() for x in apps.splitlines()}
raise SystemExit(0 if any(gpus[str(i)] not in busy for i in (1,2,3)) else 1)
PY
  then
    echo "$(date -Is) stage=$stage: free authorized GPU found"
    if [[ "$stage" == smoke ]]; then
      if bash run_long_context_native_baselines.sh --smoke; then
        stage=full
      else
        echo "$(date -Is) smoke failed; inspect logs before rerunning"
        echo 1 >"$OUTPUT_DIR/queue.exitcode"
        exit 1
      fi
    else
      status=0
      bash run_long_context_native_baselines.sh || status=$?
      echo "$status" >"$OUTPUT_DIR/queue.exitcode"
      exit "$status"
    fi
  else
    echo "$(date -Is) stage=$stage: waiting for a free GPU in 1–3"
    sleep 30
  fi
done
