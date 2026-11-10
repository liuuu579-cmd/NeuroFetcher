#!/usr/bin/env python3
"""Summarize completed long-context control metrics into CSV/Markdown."""
import argparse
import csv
import json
import hashlib
import math
from fractions import Fraction
from pathlib import Path

STRATEGIES = ('full', 'oracle', 'previous_step', 'layer5_reuse', 'quest_capped', 'quest_native', 'neural_offline')
LABELS = {'full': 'Full', 'oracle': 'Oracle（每 head）', 'previous_step': 'Previous-step（每 head）',
          'layer5_reuse': '第5层复用（每 head）', 'quest_capped': 'Quest 严格30%适配',
          'quest_native': 'Quest 官方规则 OPT 适配', 'neural_offline': 'Neuro-Fetcher（历史离线版）'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(correct, n):
    z = 1.959963984540054
    p = correct / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0, center - half), min(1, center + half)]


def stats(items):
    n = len(items)
    count = sum(r['correct'] for r in items)
    greedy = sum(r['greedy_correct'] for r in items)
    return dict(samples=n, top1=count / n, top1_wilson95=wilson(count, n),
                greedy_accuracy=greedy / n, greedy_wilson95=wilson(greedy, n),
                mean_rank=sum(r['rank'] for r in items) / n,
                mean_gold_logprob=sum(r['gold_logprob'] for r in items) / n)


def audit(path, inputs, input_hash):
    data = json.loads(path.read_text())
    strategy = data['strategy']
    assert strategy in STRATEGIES and data['complete']
    assert data['input_sha256'] == input_hash and data['requests'] == len(inputs)
    assert data['model_path'] == '/home/liushenyu/opt-13b'
    assert data['max_position_embeddings'] == 2048 and data['total_layers'] == data['num_attention_heads'] == 40
    assert data['torch_num_threads'] <= 10 and len(data['cpu_affinity']) <= 10
    assert data['cuda_visible_devices'] in ('0', '1', '2', '3')
    assert data['recent_window'] == (20 if strategy == 'neural_offline' else 0)
    assert data['index_scope'] == ('shared' if strategy == 'neural_offline' else 'per_head')
    for src, digest in data['source_sha256'].items():
        assert sha(src) == digest, src
    for src, digest in (data['predictor_sha256'] or {}).items():
        assert sha(src) == digest, src
    loading = data['loading_info']
    allowed = ('predictor.', 'predictor_embed_13b.') if strategy == 'neural_offline' else ()
    assert all(k.startswith(allowed) for k in loading.get('missing_keys', []))
    assert all(not loading.get(k) for k in ('unexpected_keys', 'mismatched_keys', 'error_msgs'))
    output_path = path.with_name(path.name.replace('.metrics.json', '.jsonl'))
    outputs = [json.loads(s) for s in output_path.read_text().splitlines() if s.strip()]
    assert len(outputs) == len(inputs) == len(data['per_request'])
    for row, request, timing in zip(outputs, inputs, data['per_request']):
        assert row['id'] == request['id'] == timing['id']
        assert row['length'] == timing['tokens'] == len(request['input_ids']) <= 2048
        for key in ('task', 'needle_start', 'needle_depth', 'answer', 'candidate_words'):
            assert row[key] == request[key], (key, row['id'])
        probs = row['candidate_logprobs']
        assert len(probs) == len(request['candidate_ids']) and all(math.isfinite(v) for v in probs)
        gold = request['candidate_words'].index(request['answer'])
        order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
        assert row['rank'] == order.index(gold) + 1
        assert row['correct'] == (row['rank'] == 1)
        assert row['greedy_correct'] == (row['greedy_token_id'] == request['candidate_ids'][gold])
        assert row['gold_logprob'] == probs[gold]
    candidate = sum(r['length'] * (r['length'] + 1) // 2 for r in inputs) * 1600
    assert data['candidate_entries'] == candidate
    assert math.isclose(data['actual_budget_ratio'], data['selected_entries'] / candidate)
    assert 0 < data['selected_entries'] <= candidate
    if strategy == 'full':
        assert data['actual_budget_ratio'] == 1
    elif strategy != 'neural_offline':
        f = Fraction(str(data['budget_ratio']))
        target = sum(sum(max(1, t * f.numerator // f.denominator) for t in range(1, r['length'] + 1)) for r in inputs) * 1600
        assert target == data['expected_floor_entries']
        if strategy != 'quest_native':
            assert data['selected_entries'] == target
            assert data['rows_with_budget_mismatch'] == data['rows_above_budget'] == 0
    overall = stats(outputs)
    for key in ('samples', 'top1', 'greedy_accuracy', 'mean_rank', 'mean_gold_logprob'):
        assert math.isclose(overall[key], data['aggregate']['overall'][key])
    return data, outputs, overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('results/long_context_retrieval_combined_20261017.jsonl'))
    p.add_argument('--output-dir', type=Path, default=Path('results/long_context_controls_balanced_20261017'))
    args = p.parse_args()
    inputs = [json.loads(s) for s in args.input.read_text().splitlines() if s.strip()]
    input_hash = sha(args.input)
    metrics = sorted(args.output_dir.glob('*.metrics.json'))
    if not metrics:
        raise SystemExit(f'No completed metrics in {args.output_dir}')
    rows = []
    detailed = []
    by_strategy = {}
    for path in metrics:
        data, outputs, overall = audit(path, inputs, input_hash)
        by_strategy[data['strategy']] = outputs
        rows.append({
            'strategy': data['strategy'], 'samples': overall['samples'],
            'top1': overall['top1'], 'greedy_accuracy': overall.get('greedy_accuracy'),
            'top1_wilson95': stats(outputs)['top1_wilson95'],
            'greedy_wilson95': stats(outputs)['greedy_wilson95'],
            'mean_rank': overall['mean_rank'], 'mean_gold_logprob': overall.get('mean_gold_logprob'),
            'actual_budget_ratio': data['actual_budget_ratio'],
            'forward_seconds': data['forward_seconds'], 'wall_seconds': data['wall_seconds'],
        })
        for task in sorted({r['task'] for r in outputs}):
            for length in (512, 1024, 1536, 1900):
                subset = [r for r in outputs if r['task'] == task and r['length'] == length]
                detailed.append(dict(strategy=data['strategy'], task=task, length=length,
                                     **stats(subset), actual_budget_ratio=data['actual_budget_ratio']))
    rows.sort(key=lambda r: r['strategy'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'summary.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = [
        '# Long-context retrieval controls', '',
        f'Input: `{args.input}`; completed conditions: {len(rows)}.', '',
        'This is a 2048-position OPT-13B stress test, not standard full-length LongBench.', '',
        '| strategy | samples | top-1 | greedy | mean rank | actual budget | forward s |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append('| {strategy} | {samples} | {top1:.4f} | {greedy_accuracy:.4f} | {mean_rank:.3f} | {actual_budget_ratio:.4f} | {forward_seconds:.1f} |'.format(**row))
    (args.output_dir / 'summary.md').write_text('\n'.join(lines) + '\n')
    with (args.output_dir / 'by_task_length.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detailed[0]))
        writer.writeheader(); writer.writerows(detailed)
    (args.output_dir / 'validated_summary.json').write_text(json.dumps({
        'complete': set(by_strategy) == set(STRATEGIES), 'input_sha256': input_hash,
        'validation': 'All outputs independently checked for request order, finite scores, rankings, greedy correctness, source/input/predictor hashes, checkpoint loading and actual budget counts.',
        'overall': rows, 'by_task_length': detailed,
        'interval_definition': 'Descriptive binomial Wilson 95% intervals over this synthetic set; not run-to-run system variability.',
    }, indent=2, ensure_ascii=False) + '\n')
    chinese = ['# OPT-13B 长文本检索测试', '',
               '标准OPT-13B，FP16，batch size 1，未修改位置编码。两项自建任务各128条，共256条；长度512/1024/1536/1900，目标深度25%/50%/75%/90%，每格8个平衡答案。', '',
               '单条needle检索：从C4背景中提取唯一指定记录；关联检索：加入4条相似记录作干扰，按问题指定的记录名取值。', '',
               'QA提示基线结束于 `Answer:`；填空提示在末尾提供固定语法前缀，如 `The archive marker is`，不附加答案token。候选排名使用8个单token词；另报告不限制词表的全词表首token精确匹配。QA和填空提示使用同一背景记录与标签。', '',
               '| 方法 | QA候选排名准确率 | 填空候选排名准确率 [Wilson 95%] | 填空全词表首token准确率 [Wilson 95%] | 实际选择密度 |',
               '|---|---:|---:|---:|---:|']
    lookup = {r['strategy']: r for r in rows}
    qa_lookup = {}
    if any(r.get('prompt_protocol') for r in inputs):
        qa_csv = args.output_dir.parent / 'long_context_controls_balanced_20261017' / 'summary.csv'
    else:
        qa_csv = args.output_dir / 'summary.csv'
    if qa_csv.is_file():
        with qa_csv.open() as stream:
            for row in csv.DictReader(stream):
                qa_lookup[row['strategy']] = float(row['top1'])
    for strategy in STRATEGIES:
        if strategy not in lookup: continue
        r = lookup[strategy]
        qa = qa_lookup.get(strategy)
        qa_value = f"{qa:.2%}" if qa is not None else '见QA报告'
        ci1, ci2 = r['top1_wilson95'], r['greedy_wilson95']
        chinese.append(f"| {LABELS[strategy]} | {qa_value} | {r['top1']:.2%} [{ci1[0]:.2%}, {ci1[1]:.2%}] | {r['greedy_accuracy']:.2%} [{ci2[0]:.2%}, {ci2[1]:.2%}] | {r['actual_budget_ratio']:.2%} |")
    for task, label in [('needle_retrieval', '单条needle检索'), ('associative_retrieval', '带干扰的关联检索')]:
        chinese += ['', f'## {label}：按长度的首token精确匹配', '',
                    '| 方法 | 512排名/全词表 | 1024排名/全词表 | 1536排名/全词表 | 1900排名/全词表 |', '|---|---:|---:|---:|---:|']
        for strategy in STRATEGIES:
            if strategy not in lookup: continue
            values = [next(d for d in detailed if d['strategy'] == strategy and d['task'] == task and d['length'] == length) for length in (512, 1024, 1536, 1900)]
            chinese.append('| ' + LABELS[strategy] + ' | ' + ' | '.join(f"{v['top1']:.2%}/{v['greedy_accuracy']:.2%}" for v in values) + ' |')
    chinese += ['', '## 解释范围', '',
                '表中每格为候选排名准确率/全词表首token准确率。统计来自每格32条样本；各个长度和深度切片每格仅8条，切片比例只作描述。' , '',
                '这是原始OPT-13B的2K内上下文压力测试，不是标准LongBench结果，也不能回应4K/8K/16K或GQA模型表现。仍需额外的长上下文模型与相应选择器适配来完整回应审稿人的现代模型/长上下文要求。', '',
                'Oracle、Previous-step、第5层复用、Quest严格预算版本均使用每head索引和严格floor预算，且无recent-20。Quest官方规则OPT适配保留最少3页和前两层Full，实际密度单独列示。', '',
                '原有Neuro-Fetcher保留recent-20、shared索引、1层/4head/FFN2048 predictor和已恢复的13B embedding；双向predictor在整段prompt上运行，属于历史离线实现参照。其早期token打分能依赖后续问题，不能当成已验证的在线逐步预测，更不能将差异全归因为学习或同预算优势。', '',
                '所有方法均在整段输入的每个位置使用相应稀疏选择，未使用常见的Full prefill + sparse decode协议。预测的是一个答案token；该设置不测长回答生成。', '',
                '实际密度按所有层、head和因果位置的选择项加权计算，不等于最后一步的预算比例、实际传输字节或物理KV内存；记录的前向时间只用于作业诊断，不用于系统速度结论。', '',
                'C4背景来自本地1GB文件64MiB之后的区域，以避开已知从文件开头生成的predictor训练文本；不能排除该语料曾见于OPT预训练或未知训练版本。', '',
                '详细CSV/JSON保留按任务和长度的统计及Wilson 95%区间；区间描述当前合成集合，不代表跨运行系统波动。']
    (args.output_dir / 'summary_zh.md').write_text('\n'.join(chinese) + '\n')
    print(csv_path)
    print(args.output_dir / 'summary.md')


if __name__ == '__main__':
    main()
