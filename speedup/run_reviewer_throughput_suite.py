#!/usr/bin/env python3
"""Run the first matched-backend matrix sequentially on ONE idle GPU.

No legacy scripts, symlinks, checkpoints or running processes are changed.
Each subprocess gets at most ten CPU cores. A nonempty GPU stops the queue.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path('/home/liushenyu/anaconda3/envs/infinigen/bin/python')


def check_idle(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True)
    used_mb = int(result.stdout.strip())
    if used_mb > 100:
        raise RuntimeError(f'GPU {gpu} is busy ({used_mb} MiB); stopping without affecting it')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2, 3), default=2)
    parser.add_argument('--cpu-cores', default='20-29')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--contexts', nargs='+', type=int, default=(512, 1024, 1536, 1920))
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--gen-len', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if sys.executable != str(PYTHON) and Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError('Use the infinigen environment')
    if not all(16 <= length <= 2048 - args.gen_len for length in args.contexts):
        raise ValueError('Every prompt plus generation must fit OPT 2048 positions')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='10',
               MKL_NUM_THREADS='10', OPENBLAS_NUM_THREADS='10',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    base = ['taskset', '-c', args.cpu_cores, str(PYTHON), '-u']
    (args.output_dir / 'manifest.json').write_text(json.dumps({
        **vars(args), 'output_dir': str(args.output_dir),
        'start_utc': datetime.now(timezone.utc).isoformat(),
        'strategies': ['full_gpu', 'full_offload', 'neural_serial', 'neural_async'],
        'generation_policy': 'greedy, exactly gen_len tokens including first prefill output; EOS does not shorten runs',
        'repetitions': 'one complete warmup + measured repetitions in a fresh process per method/context',
        'scope': 'first backend/scheduling ablation only; other algorithmic baselines remain to integrate',
    }, indent=2) + '\n')

    def run(name, script, options):
        check_idle(args.gpu)
        command = base + [str(ROOT / 'speedup' / script)] + list(map(str, options))
        print(datetime.now(timezone.utc).isoformat(), name, flush=True)
        with (args.output_dir / f'{name}.log').open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} exited {result.returncode}; see {name}.log')

    try:
        run('tiny_correctness', 'test_reviewer_offload_backend.py', ['--gpu'])
        run('opt13b_correctness', 'reviewer_offload_smoke.py', [
            '--prompt-len', 512, '--gen-len', 8, '--batch', args.batch,
            '--output-dir', args.output_dir / 'opt13b_correctness'])
        rows = []
        rng = random.Random(20261026)
        for length in args.contexts:
            strategies = ['full_gpu', 'full_offload', 'neural_serial', 'neural_async']
            rng.shuffle(strategies)
            for strategy in strategies:
                name = f'p{length}-g{args.gen_len}-b{args.batch}-{strategy}'
                output = args.output_dir / f'{name}.json'
                run(name, 'benchmark_reviewer_offload.py', [
                    '--strategy', strategy, '--prompt-len', length,
                    '--gen-len', args.gen_len, '--batch', args.batch,
                    '--repeats', args.repeats, '--output', output])
                document = json.loads(output.read_text())
                row = {'strategy': strategy, 'prompt_len': length, 'gen_len': args.gen_len, 'batch': args.batch}
                for field, values in document['summary'].items():
                    row[field + '_mean'] = values['mean']
                    row[field + '_stdev'] = values['stdev']
                rows.append(row)
                with (args.output_dir / 'summary.csv').open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
        (args.output_dir / 'suite.exitcode').write_text('0\n')
        print('All correctness gates and benchmark conditions completed', flush=True)
    except Exception as error:
        (args.output_dir / 'suite.exitcode').write_text('1\n')
        (args.output_dir / 'failure.txt').write_text(str(error) + '\n')
        raise


if __name__ == '__main__':
    main()
