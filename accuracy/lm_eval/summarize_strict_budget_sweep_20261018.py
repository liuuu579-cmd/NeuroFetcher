#!/usr/bin/env python3
"""Independently audit corrected 8/15/20/30 percent artifacts and summarize."""
import csv
from datetime import datetime, timezone, timedelta
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = ('winogrande', 'openbookqa', 'copa', 'rte')
STRATEGIES = ('oracle', 'previous_step', 'layer5_reuse', 'quest_capped', 'quest_native')
LABELS = {'oracle': 'Oracle（每 head）', 'previous_step': 'Previous-step（每 head）',
          'layer5_reuse': '第5层复用（每 head）', 'quest_capped': 'Quest 严格预算适配',
          'quest_native': 'Quest 官方选择规则的 OPT 适配'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(path, ratio, requests):
    m = json.loads(path.read_text())
    task, strategy = m['task'], m['strategy']
    assert m['complete'] and m['budget_ratio'] == ratio
    assert m['model_path'] == '/home/liushenyu/opt-13b' and m['shots'] == 5
    assert m['index_scope'] == 'per_head' and m['recent_window'] == 0 and not m['force_current_token']
    assert m['total_layers'] == 40 and m['torch_num_threads'] <= 10 and len(m['cpu_affinity']) <= 10
    assert m['cuda_visible_devices'] in ('0', '1', '2', '3')
    assert all(not m['loading_info'].get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    assert m['input_sha256'] == sha(HERE / f'results/{task}-5.jsonl')
    for source, digest in m['source_sha256'].items():
        assert sha(Path(source)) == digest, source
    assert m['requests'] == len(requests)
    output = path.with_name(path.name.replace('.metrics.json', '.jsonl'))
    lengths = []
    count = 0
    with output.open() as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            assert row['request'] == requests[count - 1], (output, count)
            probs = row['result']['choices'][0]['logprobs']
            tokens, logprobs, top = (probs[k] for k in ('tokens', 'token_logprobs', 'top_logprobs'))
            assert len(tokens) == len(logprobs) == len(top) == m['per_request'][count - 1]['tokens']
            assert logprobs[0] is None and top[0] is None
            assert all(math.isfinite(p) for p in logprobs[1:])
            assert all(len(v) == 1 and all(math.isfinite(p) for p in v.values()) for v in top[1:])
            lengths.append(len(tokens))
    assert count == len(requests)
    f = Fraction(str(ratio))
    candidate = sum(n * (n + 1) // 2 for n in lengths) * 1600
    expected = sum(sum(max(1, t * f.numerator // f.denominator) for t in range(1, n + 1)) for n in lengths) * 1600
    assert candidate == m['candidate_entries'] and expected == m['expected_floor_entries']
    assert sum(lengths) == m['tokens']
    assert math.isclose(m['actual_budget_ratio'], m['selected_entries'] / candidate)
    assert m['quest_native_exceptions'] == (strategy == 'quest_native')
    if strategy != 'quest_native':
        assert m['rows_above_budget'] == m['rows_with_budget_mismatch'] == 0
        assert m['selected_entries'] == expected
    if strategy == 'layer5_reuse':
        assert m['reuse_layer_one_based'] == 5
    if strategy.startswith('quest_'):
        assert m['quest_page_size'] == 16
    acc_path = path.with_name(path.name.replace('.metrics.json', '.accuracy.json'))
    acc = json.loads(acc_path.read_text())['results'][task]
    return {'budget_ratio': ratio, 'task': task, 'strategy': strategy,
            'acc': acc['acc'], 'acc_norm': acc.get('acc_norm'), 'acc_stderr': acc.get('acc_stderr'),
            'actual_budget_ratio': m['actual_budget_ratio'], 'requests': count,
            'rows_with_budget_mismatch': m['rows_with_budget_mismatch'],
            'forward_seconds': m['forward_seconds'], 'wall_seconds': m['wall_seconds'],
            'metrics_path': str(path.resolve()), 'output_sha256': sha(output)}


def main():
    requests = {task: [json.loads(s) for s in (HERE / f'results/{task}-5.jsonl').read_text().splitlines() if s.strip()]
                for task in TASKS}
    root = HERE / 'results/strict_budget_sweep_20261017'
    assert (root / 'suite.exitcode').read_text().strip() == '0'
    dirs = {0.30: HERE / 'results/strict_controls_20261017',
            0.20: root / 'b0p20', 0.15: root / 'b0p15', 0.08: root / 'b0p08'}
    rows = []
    for ratio, directory in dirs.items():
        paths = sorted(directory.glob('*.metrics.json'))
        assert len(paths) == 20, (directory, len(paths))
        audited = [audit(p, ratio, requests[json.loads(p.read_text())['task']]) for p in paths]
        assert {(r['task'], r['strategy']) for r in audited} == {(t, s) for t in TASKS for s in STRATEGIES}
        rows.extend(audited)
        print(f'Validated budget={ratio}: 20 configurations', flush=True)
    prefix = HERE / 'strict_budget_sweep_summary_20261018'
    assert not prefix.with_suffix('.json').exists(), 'Use a new report filename to preserve the previous audit'
    with prefix.with_suffix('.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
    report = {'generated_at': stamp, 'validated_configurations': len(rows),
              'validated_requests': sum(r['requests'] for r in rows),
              'validation': 'Exact request order/content, finite logprobs, checkpoint/source/input hashes, independently recomputed budgets and worker resource caps checked.',
              'results': rows}
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    lines = ['# OPT-13B 修正定义后的多预算对比', '',
             '模型 `/home/liushenyu/opt-13b`，5-shot、FP16、batch size 1。修正后的四档预算共80组全部完成并核验。', '',
             '严格预算方法使用每 head 独立索引、`max(1,floor(ratio×可见前缀长度))`，不额外保留当前或最近20个 token。',
             'Quest 官方选择规则的 OPT 适配单独标注，保留最少3页、前两层Full，因此实际比例可能明显超过名义比例。', '',
             '这些是准确率用的全位置因果稀疏选择模拟，前向耗时不代表KV卸载或实际decode延迟。旧版recent-window/递归复用结果未混入此表。', '']
    lookup = {(r['budget_ratio'], r['task'], r['strategy']): r for r in rows}
    for ratio in dirs:
        lines += [f'## 名义预算 {ratio:.0%}', '',
                  '| 方法 | Winogrande | OpenBookQA acc / acc_norm | COPA | RTE |',
                  '|---|---:|---:|---:|---:|']
        for strategy in STRATEGIES:
            vals = []
            for task in TASKS:
                r = lookup[ratio, task, strategy]
                value = f"{r['acc']:.2%}"
                if task == 'openbookqa': value += f" / {r['acc_norm']:.2%}"
                vals.append(value)
            lines.append('| ' + LABELS[strategy] + ' | ' + ' | '.join(vals) + ' |')
        lines += ['', '| 方法 | Winogrande实际比例 | OpenBookQA实际比例 | COPA实际比例 | RTE实际比例 |',
                  '|---|---:|---:|---:|---:|']
        for strategy in STRATEGIES:
            lines.append('| ' + LABELS[strategy] + ' | ' + ' | '.join(f"{lookup[ratio,t,strategy]['actual_budget_ratio']:.2%}" for t in TASKS) + ' |')
        lines.append('')
    lines += ['Oracle 使用本次稀疏运行状态的真实QK排序，是理想选择参照，不保证下游准确率高于所有方法；它与含recent-20训练标签的Oracle定义不同。', '',
              'Previous-step 使用同层同head前一query的真实完整QK排序，并按当前k重新选择，属于理想化时序局部性参照。', '',
              '第5层复用：第1–5层各用Oracle，第6–40层按head编号复用第5层当前query的索引。跨层相同head编号不保证相同语义。', '',
              'Quest 严格预算版本允许末页部分token并取消浅层Full/最少3页例外；两种Quest均为OPT准确率适配，不是官方端到端速度复现。', '',
              'Neuro-Fetcher的recent-20规则不属于本次新增基线；其名义30%历史Winogrande结果的实际密度约45.94%，不能当作严格同预算比较。', '',
              f"核验 {len(rows)} 组、{report['validated_requests']:,} 条输出；报告时间（北京时间）：{stamp}。"]
    prefix.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(prefix.with_suffix('.md'))


if __name__ == '__main__':
    main()
