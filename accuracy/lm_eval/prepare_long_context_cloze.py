#!/usr/bin/env python3
"""Derive answer-prefix retrieval inputs without appending any answer token.

Full OPT tends to begin QA answers with 'The'. Supply that constant grammatical
prefix to measure open-vocabulary value retrieval rather than answer formatting.
Every condition retains the same total length by removing an equally short
piece from the start of the background, before the first inserted fact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'transformers/src'))
from transformers import AutoTokenizer


def convert(row, tokenizer):
    row = dict(row)
    match = re.fullmatch(r'\nQuestion: What is the ([a-z]+) marker\?\nAnswer:', row['question'])
    if not match:
        raise ValueError(row['question'])
    suffix = f' The {match[1]} marker is'
    extra = tokenizer.encode(suffix, add_special_tokens=False)
    cut = len(extra)
    assert row['needle_start'] > cut
    row['input_ids'] = row['input_ids'][cut:] + extra
    row['needle_start'] -= cut
    row['needle_actual_depth'] = row['needle_start'] / row['length']
    row['id'] = 'cloze-' + row['id']
    row['question'] += suffix
    row['answer_prefix'] = suffix
    row['prompt_protocol'] = 'QA with constant grammatical answer prefix; next token predicted over full vocabulary; no gold answer appended'
    assert len(row['input_ids']) == row['length']
    needle = tokenizer.decode(row['input_ids'][row['needle_start']:row['needle_start'] + row['needle_length']])
    assert ' ' + row['answer'] + '.' in needle
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained('/home/liushenyu/opt-13b', local_files_only=True, use_fast=False)
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.output.open('x') as target:
        for row in rows:
            converted = convert(row, tok)
            converted['derived_from_input_sha256'] = digest
            target.write(json.dumps(converted, ensure_ascii=False) + '\n')
    print(f'Created {args.output}: {len(rows)} records')


if __name__ == '__main__':
    main()
