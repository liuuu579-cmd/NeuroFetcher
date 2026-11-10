#!/usr/bin/env python3
"""Two authorized GPU lanes for exact-selection compact staging benchmarks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from run_memory_suite_v9 import idle, PYTHON, ROOT


def lane(output, gpu, cores, method):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='10', MKL_NUM_THREADS='10',
               OPENBLAS_NUM_THREADS='10', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
               PYTORCH_CUDA_ALLOC_CONF='backend:cudaMallocAsync')
    for length in (1920, 512, 1024, 1536):
        idle(gpu)
        stem = f'p{length}-g128-b4-{method}'
        cmd = ['taskset', '-c', cores, str(PYTHON), '-u',
               str(ROOT / 'speedup/benchmark_reviewer_offload_v13.py'),
               '--strategy', method, '--prompt-len', str(length), '--gen-len', '128',
               '--batch', '4', '--warmups', '1', '--repeats', '5', '--threads', '10',
               '--output', str(output / (stem + '.json'))]
        print(f'START GPU{gpu} {stem}', flush=True)
        with (output / (stem + '.log')).open('x') as stream:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        print(f'END GPU{gpu} {stem} exit={result.returncode}', flush=True)
        if result.returncode:
            raise RuntimeError(f'{stem} failed; see its new log')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    checks = args.output_dir.parent
    for name in ('tiny_correctness.json', 'opt13b_correctness.json'):
        assert json.loads((checks/name).read_text())['passed']
    idle(2)
    idle(3)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'manifest.json').open('x') as stream:
        json.dump(dict(start_utc=datetime.now(timezone.utc).isoformat(),
            model='OPT-13B FP16', contexts=[1920, 512, 1024, 1536], batch=4,
            generation_len=128, warmups=1, repeats=5,
            gpu2=dict(method='neural_serial', cores='20-29'),
            gpu3=dict(method='neural_async', cores='30-39'),
            change='Compact adaptive staging; exact shared source embedding; cudaMallocAsync; request/prefill cache release; original predictor precision, selection and attention preserved'),
            stream, indent=2)
    failures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(lane, args.output_dir, gpu, cores, method)
                   for gpu, cores, method in ((2, '20-29', 'neural_serial'),
                                             (3, '30-39', 'neural_async'))]
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                failures.append(str(exc))
    with (args.output_dir/'suite_status.json').open('x') as stream:
        json.dump(dict(complete=not failures, failures=failures), stream, indent=2)
    if failures:
        raise RuntimeError('; '.join(failures))


if __name__ == '__main__':
    main()
