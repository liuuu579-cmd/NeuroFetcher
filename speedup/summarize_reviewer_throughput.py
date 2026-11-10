#!/usr/bin/env python3
"""Read completed conditions, check matched selections, and print a compact table."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result_dir', type=Path)
    args = p.parse_args()
    documents = []
    for path in args.result_dir.glob('p*-g*-b*-*.json'):
        doc = json.loads(path.read_text())
        if doc.get('complete'):
            documents.append(doc)
    documents.sort(key=lambda d: (d['prompt_len'], d['strategy']))
    print('| Context | Method | Decode tok/s (mean ± SD) | Total s | History % | GPU peak GiB |')
    print('|---:|---|---:|---:|---:|---:|')
    for d in documents:
        s = d['summary']
        print(f"| {d['prompt_len']} | {d['strategy']} | "
              f"{s['decode_tokens_per_s']['mean']:.2f} ± {s['decode_tokens_per_s']['stdev']:.2f} | "
              f"{s['total_s']['mean']:.2f} | {100*s['actual_history_selection_ratio']['mean']:.2f} | "
              f"{s['peak_allocated_bytes']['mean']/2**30:.2f} |")
    keys = {(d['prompt_len'], d['batch_size'], d['generation_len']) for d in documents}
    for key in sorted(keys):
        subset = {d['strategy']: d for d in documents
                  if (d['prompt_len'], d['batch_size'], d['generation_len']) == key}
        if 'neural_serial' not in subset or 'neural_async' not in subset:
            continue
        serial, asynchronous = subset['neural_serial'], subset['neural_async']
        if len(serial['runs']) != len(asynchronous['runs']):
            raise AssertionError(f'Repetition count differs at {key}')
        for s, a in zip(serial['runs'], asynchronous['runs']):
            if s['output_sha256'] != a['output_sha256']:
                raise AssertionError(f'Free-generation outputs differ at {key}')
            if [x['index_hash'] for x in s['plans']] != [x['index_hash'] for x in a['plans']]:
                raise AssertionError(f'Serial/async selections differ at {key}')
        speedup = (asynchronous['summary']['decode_tokens_per_s']['mean'] /
                   serial['summary']['decode_tokens_per_s']['mean'])
        print(f'Context {key[0]}: matching free outputs and indices; async/serial throughput={speedup:.3f}×')
    exitcode = args.result_dir/'suite.exitcode'
    print('Queue status:', exitcode.read_text().strip() if exitcode.exists() else 'not finished')


if __name__ == '__main__':
    main()
