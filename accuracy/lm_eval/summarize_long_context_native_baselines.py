#!/usr/bin/env python3
"""Audit new baseline results and combine with existing long-context controls."""
import argparse
import csv
import json
import math
from pathlib import Path

from summarize_long_context_controls import audit, sha, stats, LABELS, STRATEGIES

HERE = Path(__file__).resolve().parent
NATIVE = ('infinigen', 'h2o', 'quant4')
NAMES = dict(LABELS, infinigen='InfiniGen 原生', h2o='H2O 原生', quant4='4-bit KV 量化')


def native_audit(path, inputs, input_hash):
    data = json.loads(path.read_text())
    assert data['complete'] and not data['smoke'] and data['strategy'] in NATIVE
    assert data['input_sha256'] == input_hash and data['requests'] == len(inputs)
    assert data['underlying_model'] == '/home/liushenyu/opt-13b'
    assert data['recent20_added'] is False
    assert data['cuda_visible_devices'] in ('0', '1', '2', '3')
    assert data['torch_num_threads'] <= 10 and len(data['cpu_affinity']) <= 10
    for src, digest in {**data['source_sha256'], **(data['partial_weight_sha256'] or {})}.items():
        assert sha(src) == digest, src
    loading = data['loading_info']
    assert not any(loading.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs'))
    assert not loading['unexpected_keys'] or (data['strategy'] == 'infinigen' and all(
        k.endswith(('self_attn.q_proj.bias', 'self_attn.k_proj.bias')) for k in loading['unexpected_keys']))
    assert len(data['full_equivalence_checks']) == 16
    for check in data['full_equivalence_checks']:
        assert check['rank'] == check['reference_rank']
        assert check['max_logprob_delta'] <= (.25 if data['strategy'] == 'infinigen' else .1)
    outputs = [json.loads(s) for s in path.with_name(path.name.replace('.metrics.json', '.jsonl')).read_text().splitlines()]
    assert len(outputs) == len(inputs) == len(data['per_request'])
    for row, request, timing in zip(outputs, inputs, data['per_request']):
        assert row['id'] == request['id'] == timing['id']
        assert row['length'] == timing['tokens'] == len(request['input_ids']) <= 2048
        for key in ('task', 'needle_start', 'needle_depth', 'answer', 'candidate_words'):
            assert row[key] == request[key]
        probs = row['candidate_logprobs']
        assert len(probs) == len(request['candidate_ids']) and all(math.isfinite(v) for v in probs)
        gold = request['candidate_words'].index(request['answer'])
        order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
        assert row['rank'] == order.index(gold)+1
        assert row['correct'] == (row['rank'] == 1)
        assert row['greedy_correct'] == (row['greedy_token_id'] == request['candidate_ids'][gold])
        assert row['gold_logprob'] == probs[gold]
    candidate = sum(r['length']*(r['length']+1)//2 for r in inputs)*1600
    assert data['candidate_entries'] == candidate
    assert 0 < data['selected_entries'] <= candidate
    assert data['actual_budget_ratio'] == data['selected_entries']/candidate
    assert data['final_candidate_entries'] == sum(r['length'] for r in inputs)*1600
    assert data['final_step_density'] == data['final_selected_entries']/data['final_candidate_entries']
    if data['strategy'] == 'quant4':
        assert data['actual_budget_ratio'] == data['final_step_density'] == 1
        assert data['native_configuration']['qbits'] == 4
    overall = stats(outputs)
    for k in ('samples', 'top1', 'greedy_accuracy', 'mean_rank', 'mean_gold_logprob'):
        assert math.isclose(overall[k], data['aggregate']['overall'][k])
    return data, outputs, overall


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=HERE/'results/long_context_native_baselines_20261018')
    args = p.parse_args()
    all_rows, detailed = [], []
    for protocol, input_file, reference_dir in [
        ('qa', 'long_context_retrieval_combined_20261017.jsonl', 'long_context_controls_balanced_20261017'),
        ('cloze', 'long_context_cloze_balanced_20261018.jsonl', 'long_context_cloze_controls_20261018')]:
        input_path = HERE/'results'/input_file
        inputs = [json.loads(s) for s in input_path.read_text().splitlines()]
        input_hash = sha(input_path)
        existing = {}
        for path in (HERE/'results'/reference_dir).glob('*.metrics.json'):
            data, outputs, overall = audit(path, inputs, input_hash)
            assert data['strategy'] not in existing
            existing[data['strategy']] = (data, outputs, overall)
        assert set(existing) == set(STRATEGIES)
        for strategy in NATIVE:
            existing[strategy] = native_audit(args.output_dir/f'{protocol}-{strategy}.metrics.json', inputs, input_hash)
        for strategy in (*STRATEGIES, *NATIVE):
            data, outputs, overall = existing[strategy]
            row = dict(protocol=protocol, strategy=strategy, **overall,
                       actual_budget_ratio=data['actual_budget_ratio'],
                       final_step_density=data.get('final_step_density'),
                       nominal_budget_ratio=data.get('nominal_budget_ratio', data.get('budget_ratio')))
            all_rows.append(row)
            for task in sorted({r['task'] for r in outputs}):
                for length in (512, 1024, 1536, 1900):
                    subset = [r for r in outputs if r['task'] == task and r['length'] == length]
                    detailed.append(dict(protocol=protocol, strategy=strategy, task=task, length=length, **stats(subset)))
    for name, rows in [('summary.csv', all_rows), ('by_task_length.csv', detailed)]:
        with (args.output_dir/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir/'validated_summary.json').write_text(json.dumps(dict(
        complete=True, overall=all_rows, by_task_length=detailed,
        validation='Input order, finite candidate scores, independently recomputed ranks, correctness, source/partial hashes, loading, numerical Full cross-checks and selected-entry counts checked.'), indent=2, ensure_ascii=False)+'\n')
    lookup = {(r['protocol'], r['strategy']): r for r in all_rows}
    lines = ['# 长文本新增原生基线对比', '',
             'OPT-13B；同一组256条自建检索输入，512/1024/1536/1900 token。未修改已有代码或结果。', '',
             '| 方法 | QA候选排名 | 填空候选排名 | 填空全词表首token | 填空实际选择密度 | 填空最后一步密度 |',
             '|---|---:|---:|---:|---:|---:|']
    for strategy in (*STRATEGIES, *NATIVE):
        qa, cloze = lookup['qa', strategy], lookup['cloze', strategy]
        final = f"{cloze['final_step_density']:.2%}" if cloze['final_step_density'] is not None else '原报告未记录'
        lines.append(f"| {NAMES[strategy]} | {qa['top1']:.2%} | {cloze['top1']:.2%} | {cloze['greedy_accuracy']:.2%} | {cloze['actual_budget_ratio']:.2%} | {final} |")
    lines += ['', 'InfiniGen：仓库原始 `modeling_opt_inf.py`，配套变换OPT-13B checkpoint和0.3 partial weights，alpha=7、capacity=1、budget=0.3，前两层Full。', '',
              'H2O：直接调用仓库 `modify_opt.py`，15% heavy + 15% recent、前两层Full。保留原生recent窗口的当前token和边界规则，不添加Neuro-Fetcher的recent-20。', '',
              '两种原生稀疏方法按完整输入长度设容量；预热和Full层使全序列加权密度不等于严格逐位置30%。最后一步密度也包括Full层。', '',
              '4-bit量化：原仓库64维组内KV量化/反量化，前两层Full，其余38层4-bit；保留所有token，选择密度100%。理想KV值payload为FP16的28.75%，尚未计scale/min等元数据；实际仍存FP16张量，不能据此声称物理显存或速度改善。', '',
              '原生方法的softmax先上转FP32，历史控制参考使用FP16 softmax；InfiniGen另有FP16 Q/K变换。新增结果保留16条Full数值核对和checkpoint加载审计，不宣称逐位等价。', '',
              '这是2K内全位置稀疏前向压力测试，不是标准LongBench，也不是Full prefill + sparse decode或真实系统延迟测量。历史Neuro-Fetcher仍是双向整段prompt离线参照。', '',
              '置信区间与按任务/长度切片见CSV/JSON。']
    (args.output_dir/'summary_zh.md').write_text('\n'.join(lines)+'\n')
    print(args.output_dir/'summary_zh.md')


if __name__ == '__main__':
    main()
