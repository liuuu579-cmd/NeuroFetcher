#!/usr/bin/env python3
"""Create deterministic OPT-length needle-in-a-haystack retrieval inputs.

The source model has a 2048-token learned positional embedding table. This
generator therefore exercises long contexts up to 1900 tokens without
silently truncating or extending the checkpoint. Inputs are stored as token
ids so the runner can preserve the requested lengths exactly.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "transformers" / "src"))

from transformers import AutoTokenizer

WORDS = ("amber", "silver", "gold", "red", "blue", "green", "black", "white")
LENGTHS = (512, 1024, 1536, 1900)
DEPTHS = (0.25, 0.50, 0.75, 0.90)
REPEATS = 8
NEEDLE = "\nThe archive marker is {word}."
QUESTION = "\nQuestion: What is the archive marker?\nAnswer:"


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/home/liushenyu/opt-13b"))
    parser.add_argument("--source-text", type=Path,
                        default=Path("/home/liushenyu/test/c4_subset_1024mb.txt"))
    parser.add_argument("--output", type=Path,
                        default=HERE / "results/long_context_needle_balanced_20261017.jsonl")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument('--task', choices=['needle_retrieval', 'associative_retrieval'], default='needle_retrieval')
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if not args.source_text.is_file():
        raise FileNotFoundError(args.source_text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), use_fast=False,
                                               local_files_only=True)
    needle_tokens = {
        word: tokenizer.encode(NEEDLE.format(word=word), add_special_tokens=False)
        for word in WORDS
    }
    question_tokens = tokenizer.encode(QUESTION, add_special_tokens=False)
    candidate_ids = [tokenizer.encode(" " + word, add_special_tokens=False) for word in WORDS]
    if any(len(ids) != 1 for ids in candidate_ids):
        raise RuntimeError("Every answer candidate must be exactly one tokenizer token")

    # Original predictor data construction consumes the beginning of this
    # file. Use a separate later region, record exact byte provenance, and
    # never claim this excludes unknown alternative training data.
    byte_offset = 64 * 1024 * 1024
    with args.source_text.open('rb') as source:
        source.seek(byte_offset)
        source.readline()  # Start on a complete UTF-8/text line.
        actual_byte_offset = source.tell()
        text = source.read(8 * 1024 * 1024).decode('utf-8', errors='ignore')
    source_hash = sha256(args.source_text)
    filler = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    needed = sum(LENGTHS) * len(DEPTHS) * args.repeats + 4096
    if len(filler) < needed:
        raise RuntimeError(f"Source text has only {len(filler)} tokens; need at least {needed}")

    records = []
    sample_number = 0
    source_cursor = 0
    for length in LENGTHS:
        for depth in DEPTHS:
            for repeat in range(args.repeats):
                word = WORDS[repeat % len(WORDS)]
                needle = needle_tokens[word]
                question = QUESTION
                question_ids = question_tokens
                events = [(depth, True, needle)]
                if args.task == 'associative_retrieval':
                    keys = ['archive', 'vault', 'locker', 'cabinet', 'safe']
                    target_key = keys[repeat % len(keys)]
                    question = f'\nQuestion: What is the {target_key} marker?\nAnswer:'
                    question_ids = tokenizer.encode(question, add_special_tokens=False)
                    needle = tokenizer.encode(f'\nThe {target_key} marker is {word}.', add_special_tokens=False)
                    events = [(depth, True, needle)]
                    for distractor, key in enumerate(k for k in keys if k != target_key):
                        distractor_word = WORDS[(repeat + distractor + 1) % len(WORDS)]
                        fact = tokenizer.encode(f'\nThe {key} marker is {distractor_word}.', add_special_tokens=False)
                        events.append(((0.10, 0.35, 0.65, 0.95)[distractor], False, fact))
                filler_length = length - sum(len(e[2]) for e in events) - len(question_ids)
                if filler_length <= 0:
                    raise ValueError(f"Length {length} is too short for the prompt")
                source_start = source_cursor
                source_cursor += filler_length
                context = filler[source_start:source_start + filler_length]
                input_ids = []
                cursor = 0
                for event_depth, is_target, fact in sorted(events):
                    at = int(round(filler_length * event_depth))
                    input_ids.extend(context[cursor:at])
                    if is_target:
                        needle_at = len(input_ids)
                    input_ids.extend(fact)
                    cursor = at
                input_ids.extend(context[cursor:])
                input_ids.extend(question_ids)
                if len(input_ids) != length:
                    raise AssertionError((length, len(input_ids)))
                records.append({
                    "id": f"{args.task}-l{length}-d{int(depth * 100):02d}-r{repeat}",
                    "task": args.task,
                    "input_ids": input_ids,
                    "length": length,
                    "needle_start": needle_at,
                    "needle_length": len(needle),
                    "needle_depth": depth,
                    "answer": word,
                    "candidate_ids": [ids[0] for ids in candidate_ids],
                    "candidate_words": list(WORDS),
                    "question": question,
                    "source_text_sha256": source_hash,
                    "source_byte_offset": actual_byte_offset,
                    "source_token_offset_in_region": source_start,
                    "needle_actual_depth": needle_at / length,
                    "generator_sha256": sha256(Path(__file__).resolve()),
                    "generator": str(Path(__file__).resolve()),
                })
                sample_number += 1

    with args.output.open("x", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output), "records": len(records),
        "lengths": list(LENGTHS), "depths": list(DEPTHS),
        "repeats": args.repeats, "model_path": str(args.model_path),
        "source_text": str(args.source_text), "output_sha256": sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
