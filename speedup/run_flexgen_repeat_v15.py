#!/usr/bin/env python3
"""Repeat the existing native FlexGen memory/timing workload without edits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path('/home/liushenyu/anaconda3/envs/infinigen/bin/python')


def idle(gpu):
    apps = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    used = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if apps.strip() or int(used.strip()) > 100:
        raise RuntimeError(f'GPU {gpu} is busy; stopping without affecting it')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpu', type=int, choices=(2, 3), default=3)
    p.add_argument('--cores', default='30-39')
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    idle(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='10',
        MKL_NUM_THREADS='10', OPENBLAS_NUM_THREADS='10', HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    all_runs = []
    for i in range(args.repeats):
        idle(args.gpu)
        stem = f'flexgen_p1920_g128_b4_repeat{i+1}'
        output = args.output_dir / (stem + '.json')
        command = ['taskset', '-c', args.cores, str(PYTHON), '-u',
            str(ROOT/'speedup/run_native_memory_v9.py'), '--method', 'flexgen',
            '--gpu', str(args.gpu), '--prompt-len', '1920', '--gen-len', '128',
            '--batch', '4', '--threads', '10', '--output', str(output)]
        print(f'START {stem}', flush=True)
        with (args.output_dir/(stem+'.log')).open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f'END {stem} exit={result.returncode}', flush=True)
        if result.returncode:
            raise RuntimeError(f'{stem} failed; see its log')
        d = json.loads(output.read_text())
        if not d['complete'] or d['prompt_len'] != 1920 or d['batch_size'] != 4 or d['generation_len'] != 128:
            raise RuntimeError(f'Unexpected result metadata in {output}')
        all_runs.append(d)
    with (args.output_dir/'summary.json').open('x') as stream:
        json.dump(dict(method='native FlexGen', gpu=args.gpu, repeats=len(all_runs),
            decode_tokens_per_s=[d['decode_tokens_per_s'] for d in all_runs],
            mean_decode_tokens_per_s=sum(d['decode_tokens_per_s'] for d in all_runs)/len(all_runs),
            peak_allocated_gib=[d['memory']['peak_allocated_bytes']/2**30 for d in all_runs],
            mean_peak_allocated_gib=sum(d['memory']['peak_allocated_bytes'] for d in all_runs)/len(all_runs)/2**30,
            sources=[str(args.output_dir/(f'flexgen_p1920_g128_b4_repeat{i+1}.json')) for i in range(len(all_runs))]),
            stream, indent=2)
    with (args.output_dir/'suite_status.json').open('x') as stream:
        json.dump(dict(complete=True, runs=len(all_runs), failures=[]), stream, indent=2)


if __name__ == '__main__':
    main()
