#!/usr/bin/env python3
"""Validate Full/offload parity and identical serial vs Ping-Pong schedules."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from reviewer_offload_backend import OffloadEngine, load_opt
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/home/liushenyu/opt-13b')
PREDICTOR = Path('/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth')
EMBED = Path('/home/liushenyu/test/opt_13b_embed_weights.pth')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--predictor', type=Path, default=PREDICTOR)
    p.add_argument('--embedding', type=Path, default=EMBED)
    p.add_argument('--output-dir', type=Path, default=ROOT/'speedup/results/reviewer_smoke')
    p.add_argument('--prompt-len', type=int, default=128)
    p.add_argument('--gen-len', type=int, default=6)
    p.add_argument('--threads', type=int, default=10)
    p.add_argument('--batch', type=int, default=2)
    args = p.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible not in ('0', '1', '2', '3') or not torch.cuda.is_available():
        raise RuntimeError('Use one authorized visible GPU 0–3')
    if torch.get_num_threads() > args.threads:
        torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not 16 <= args.prompt_len or args.prompt_len+args.gen_len > 2048:
        raise ValueError('Prompt and generation lengths must fit OPT-13B context')
    for path in (args.model/'config.json', args.predictor, args.embedding):
        if not path.is_file():
            raise FileNotFoundError(path)

    module = load_opt()
    model, loading = module.OPTForCausalLM.from_pretrained(
        str(args.model), torch_dtype=torch.float16, device_map={'': 0},
        local_files_only=True, output_loading_info=True)
    if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('mismatched_keys'):
        raise RuntimeError(f'OPT checkpoint mismatch: {loading}')
    from predictor_model2A import PredictorTransformer
    predictor = PredictorTransformer(input_dim=5120, num_opt_layers=5, nhead=4,
                                     num_encoder_layers=1, d_hid=2048)
    pred_state = torch.load(args.predictor, map_location='cpu', weights_only=True)
    predictor.load_state_dict(pred_state, strict=True)
    predictor.to('cuda').eval()
    embedding = torch.load(args.embedding, map_location='cpu', weights_only=True)
    if embedding.shape != (model.config.vocab_size, model.config.hidden_size):
        raise ValueError(f'Embedding shape mismatch: {tuple(embedding.shape)}')
    embedding = embedding.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), use_fast=False, local_files_only=True)
    prompts = ['The reader opened the old archive and searched the following page.',
               'A scientist measured the temperature before beginning a new experiment.']
    ids = torch.cat([tokenizer(' '.join([prompts[b % 2]]*256), add_special_tokens=False,
                              return_tensors='pt').input_ids[:, :args.prompt_len]
                     for b in range(args.batch)]).cuda()
    engine = OffloadEngine(model, predictor, embedding)
    try:
        full, full_pre, full_logits = engine.generate(ids, args.gen_len, 'full_gpu')
        forced = torch.tensor(full['output_ids'], device='cuda', dtype=torch.long)
        offload, off_pre, off_logits = engine.generate(ids, args.gen_len, 'full_offload', forced=forced)
        torch.testing.assert_close(full_pre, off_pre, rtol=0, atol=0)
        torch.testing.assert_close(full_logits, off_logits, rtol=.02, atol=.15)
        serial, _, serial_logits = engine.generate(ids, args.gen_len, 'neural_serial', forced=forced)
        async_result, _, async_logits = engine.generate(ids, args.gen_len, 'neural_async', forced=forced)
        if serial['output_ids'] != async_result['output_ids']:
            raise AssertionError('Forced input token streams diverged')
        torch.testing.assert_close(serial_logits, async_logits, rtol=0, atol=0)
        if not torch.equal(serial_logits.argmax(-1), async_logits.argmax(-1)):
            raise AssertionError('Serial/async predictions differ')
        if serial['plans'] and async_result['plans']:
            sh = [x['index_hash'] for x in serial['plans']]
            ah = [x['index_hash'] for x in async_result['plans']]
            if sh != ah:
                raise AssertionError(f'Selection indices differ: serial={sh}, async={ah}')
        result = {
            'complete': True, 'purpose': 'correctness_smoke_only_not_throughput_evidence',
            'cuda_visible_devices': visible, 'model': str(args.model),
            'model_sha256': digest(args.model/'config.json'),
            'predictor': str(args.predictor), 'predictor_sha256': digest(args.predictor),
            'predictor_embedding': str(args.embedding), 'embedding_sha256': digest(args.embedding),
            'prompt_len': args.prompt_len, 'gen_len': args.gen_len, 'batch': args.batch,
            'full_offload': {k:v for k,v in offload.items() if k != 'output_ids'},
            'full_gpu_offload_max_logit_abs_diff': float((full_logits-off_logits).abs().max()),
            'full_gpu': {k:v for k,v in full.items() if k != 'output_ids'},
            'serial': {k:v for k,v in serial.items() if k != 'output_ids'},
            'async': {k:v for k,v in async_result.items() if k != 'output_ids'},
            'serial_async_max_logit_abs_diff': float((serial_logits-async_logits).abs().max()),
            'serial_indices': [x['index_hash'] for x in serial['plans']],
            'async_indices': [x['index_hash'] for x in async_result['plans']],
            'output_ids': serial['output_ids'],
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        path = args.output_dir/'smoke_metrics.json'
        path.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('complete','prompt_len','gen_len',
                                                  'serial_async_max_logit_abs_diff',
                                                  'serial_indices','async_indices')}, indent=2))
        print(path)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
