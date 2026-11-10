#!/usr/bin/env python3
"""Supplement memory on GPU2/GPU3, one process per card and ten CPU cores each."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
        raise RuntimeError(f'GPU {gpu} became busy; no new experiment started')


def lane(output, gpu, cores, methods, contexts, gen_len, batch):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='10', MKL_NUM_THREADS='10',
               OPENBLAS_NUM_THREADS='10', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    for method in methods:
        for length in contexts:
            idle(gpu)
            stem = f'{method}_p{length}_g{gen_len}_b{batch}'
            path = output / method
            path.mkdir(exist_ok=True)
            cmd = ['taskset', '-c', cores, str(PYTHON), '-u']
            if method.startswith('quest'):
                cmd += [str(ROOT / 'speedup/run_quest_memory_v9.py'), '--variant',
                        'native' if method == 'quest_native' else 'capped']
            else:
                cmd += [str(ROOT / 'speedup/run_native_memory_v9.py'), '--method', method]
            cmd += ['--gpu', str(gpu), '--prompt-len', str(length), '--gen-len', str(gen_len),
                    '--batch', str(batch), '--threads', '10', '--output', str(path / (stem + '.json'))]
            print(f'START GPU{gpu} {stem}', flush=True)
            with (path / (stem + '.log')).open('x') as log:
                completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(f'END GPU{gpu} {stem} exit={completed.returncode}', flush=True)
            if completed.returncode:
                raise RuntimeError(f'{stem} failed; see {path / (stem + ".log")}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError('Use infinigen environment')
    idle(2)
    idle(3)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    contexts, gen_len, batch = ([512], 4, 1) if args.smoke else ([512, 1024, 1536, 1920], 128, 4)
    manifest = dict(contexts=contexts, generation_len=gen_len, batch=batch,
                    purpose='Supplementary memory measurements, no existing results replaced',
                    lanes={'2': dict(cores='20-29', methods=['flexgen', 'infinigen']),
                           '3': dict(cores='30-39', methods=['h2o', 'quest_native', 'quest_capped'])})
    with (args.output_dir / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
    failures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(lane, args.output_dir, gpu, cores, methods, contexts, gen_len, batch)
                   for gpu, cores, methods in ((2, '20-29', ['flexgen', 'infinigen']),
                                               (3, '30-39', ['h2o', 'quest_native', 'quest_capped']))]
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                failures.append(str(exc))
    with (args.output_dir / 'suite_status.json').open('x') as stream:
        json.dump(dict(complete=not failures, failures=failures), stream, indent=2)
    if failures:
        raise RuntimeError('; '.join(failures))


if __name__ == '__main__':
    main()
