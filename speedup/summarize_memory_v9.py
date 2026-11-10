#!/usr/bin/env python3
"""Validate the new memory matrix and add an explicitly sourced paper supplement."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
GIB = 2 ** 30
LENGTHS = (512, 1024, 1536, 1920)
METHODS = ('flexgen', 'infinigen', 'h2o', 'quest_native', 'quest_capped')
CUSTOM = ('full_offload', 'neural_serial', 'neural_async', 'full_gpu')


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    historical = ROOT / 'speedup/results/reviewer_throughput_20261026'
    native = ROOT / 'speedup/results/native_flexgen_v8_20261027-v8a'
    status = read(args.memory_dir / 'suite_status.json')
    if not status['complete'] or status['failures']:
        raise RuntimeError('Memory suite is incomplete')
    measured = {}
    for path in sorted(args.memory_dir.glob('*/*.json')):
        d = read(path)
        key = (d['method'], d['prompt_len'])
        if key in measured:
            raise ValueError(f'Duplicate configuration: {key}')
        assert d['complete'] and d['batch_size'] == 4 and d['generation_len'] == 128
        m = d['memory']
        for phase in ('before_generation', 'prefill', 'decode'):
            s = m[phase]
            assert 0 < s['allocated_bytes'] <= s['peak_allocated_bytes'] <= s['peak_reserved_bytes']
            assert s['reserved_bytes'] >= s['allocated_bytes']
        for field in ('allocated', 'reserved'):
            assert m[f'peak_{field}_bytes'] == max(m[p][f'peak_{field}_bytes'] for p in ('prefill', 'decode'))
        assert all(math.isfinite(d[k]) and d[k] > 0 for k in ('decode_s', 'total_s', 'decode_tokens_per_s'))
        measured[key] = (d, path)
    assert set(measured) == {(m, p) for m in METHODS for p in LENGTHS}, 'Expected all 20 configurations'
    rows = []
    for p in LENGTHS:
        for method in METHODS:
            d, path = measured[method, p]
            m = d['memory']
            oldpath = None
            if method in ('infinigen', 'h2o'):
                suffix = '_repeat' if method == 'h2o' and p == 1024 else ''
                oldpath = native / method / f'{method}_p{p}_g128_b4{suffix}.json'
            elif method.startswith('quest'):
                stem = 'quest' if method == 'quest_native' else 'quest_capped'
                oldpath = native / method / f'{stem}_p{p}_g128_b4.json'
            old = read(oldpath) if oldpath else None
            rows.append(dict(method=method, prompt_len=p, batch=4, generation_len=128,
                memory_runs=1, peak_allocated_gib=m['peak_allocated_bytes']/GIB,
                peak_reserved_gib=m['peak_reserved_bytes']/GIB,
                prefill_peak_allocated_gib=m['prefill']['peak_allocated_bytes']/GIB,
                decode_peak_allocated_gib=m['decode']['peak_allocated_bytes']/GIB,
                cpu_peak_rss_gib=m['cpu_process_peak_rss_bytes']/GIB,
                memory_run_decode_tokens_per_s=d['decode_tokens_per_s'],
                historical_decode_tokens_per_s=old['decode_tokens_per_s'] if old else '',
                historical_decode_tokens_per_s_stdev='', historical_throughput_runs=1 if old else '',
                memory_source=str(path.resolve()), historical_throughput_source=str(oldpath.resolve()) if oldpath else '',
                runtime=d['runtime']))
        for method in CUSTOM:
            path = historical / f'p{p}-g128-b4-{method}.json'
            d = read(path)
            assert d['complete'] and d['repeats'] == 5
            s = d['summary']
            rows.append(dict(method=method, prompt_len=p, batch=4, generation_len=128,
                memory_runs=d['repeats'], peak_allocated_gib=s['peak_allocated_bytes']['mean']/GIB,
                peak_reserved_gib=s['peak_reserved_bytes']['mean']/GIB,
                prefill_peak_allocated_gib=mean(r['prefill_peak_allocated_bytes'] for r in d['runs'])/GIB,
                decode_peak_allocated_gib=mean(r['decode_peak_allocated_bytes'] for r in d['runs'])/GIB,
                cpu_peak_rss_gib=s['cpu_process_peak_rss_bytes']['mean']/GIB,
                memory_run_decode_tokens_per_s='',
                historical_decode_tokens_per_s=s['decode_tokens_per_s']['mean'],
                historical_decode_tokens_per_s_stdev=s['decode_tokens_per_s']['stdev'],
                historical_throughput_runs=d['repeats'], memory_source=str(path.resolve()),
                historical_throughput_source=str(path.resolve()), runtime='custom offload backend'))
    by_key = {(r['method'], r['prompt_len']): r for r in rows}
    comparisons = []
    for p in LENGTHS:
        for method in ('neural_serial', 'neural_async'):
            a = by_key[method, p]
            for baseline in ('full_offload', 'flexgen', 'infinigen', 'h2o', 'quest_native', 'quest_capped'):
                b = by_key[baseline, p]
                bt = b['historical_decode_tokens_per_s'] or b['memory_run_decode_tokens_per_s']
                comparisons.append(dict(prompt_len=p, method=method, baseline=baseline,
                    extra_peak_allocated_gib=a['peak_allocated_gib']-b['peak_allocated_gib'],
                    extra_peak_allocated_percent=100*(a['peak_allocated_gib']/b['peak_allocated_gib']-1),
                    decode_throughput_ratio=a['historical_decode_tokens_per_s']/bt,
                    throughput_basis='historical neural / new supplementary baseline' if baseline == 'flexgen' else 'historical neural / historical baseline',
                    matched_runtime=baseline == 'full_offload'))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / 'memory_and_throughput.csv', rows)
    write_csv(args.output_dir / 'tradeoffs.csv', comparisons)
    lines = ['# 显存补测与论文写作依据（2026-10-29）', '',
        '完整补测已通过：5 种方法 × 4 种输入长度，共 20 组。OPT-13B FP16，batch=4，生成 128 tokens；decode 吞吐统计 127 个生成步。GPU 2/3 各独立运行一个实验，分别限制 CPU 20–29/30–39，共 20 核。只新增文件。', '',
        '## 1. 完整显存结果', '',
        '每格为 **峰值 allocated / 峰值 reserved，GiB**（1 GiB=2^30 bytes）。Neural/Full 四行来自原有五次运行的均值，其余五行来自本次单次补测；不能把这两批数据说成同一天同一 runtime 的重复试验。', '',
        '| 方法 | 512 | 1024 | 1536 | 1920 |', '|---|---:|---:|---:|---:|']
    for method in METHODS + CUSTOM:
        vals = [by_key[method, p] for p in LENGTHS]
        lines.append('| ' + method + ' | ' + ' | '.join(f"{r['peak_allocated_gib']:.3f} / {r['peak_reserved_gib']:.3f}" for r in vals) + ' |')
    lines += ['', '## 2. 支撑核心论点的同后端比较', '',
        '下面全部使用已有五次运行均值，相对于同一 custom backend 的 Full CPU KV offload。full_offload 是自定义 Full 参照，不应在论文中改名为原生 FlexGen。', '',
        '| 输入长度 | 方法 | decode tok/s | 相对 Full 加速 | allocated 增量 GiB | allocated 增幅 |',
        '|---:|---|---:|---:|---:|---:|']
    for c in comparisons:
        if c['baseline'] == 'full_offload':
            a = by_key[c['method'], c['prompt_len']]
            lines.append(f"| {c['prompt_len']} | {c['method']} | {a['historical_decode_tokens_per_s']:.2f} | {c['decode_throughput_ratio']:.2f}× | {c['extra_peak_allocated_gib']:.3f} | {c['extra_peak_allocated_percent']:.1f}% |")
    lines += ['', '## 3. 与原生 FlexGen 的显存比较', '',
        'FlexGen 这一列是新增的 original FlexGen 前后端，GPU 存放全部权重、CPU 存放 KV、overlap=false。Neural 使用项目 custom backend；这里展示实现的整体资源代价，不能单独解释为 predictor 的开销。', '',
        '| 输入长度 | FlexGen allocated GiB | Neural serial 增量 GiB / % | Neural async 增量 GiB / % |',
        '|---:|---:|---:|---:|']
    for p in LENGTHS:
        cs = [next(c for c in comparisons if c['prompt_len'] == p and c['method'] == m and c['baseline'] == 'flexgen') for m in ('neural_serial', 'neural_async')]
        lines.append(f"| {p} | {by_key['flexgen',p]['peak_allocated_gib']:.3f} | " + ' | '.join(f"{c['extra_peak_allocated_gib']:.3f} / {c['extra_peak_allocated_percent']:.1f}%" for c in cs) + ' |')
    lines += ['', '## 4. 吞吐来源与适用范围', '',
        '- `memory_and_throughput.csv` 分列保存历史吞吐和本次补测吞吐。历史 InfiniGen/H2O/Quest 采用已接受的 v8 报告口径；H2O 1024 使用原报告的 isolated repeat，首次并发记录仍保留在原目录。新增 FlexGen 尚无原先的五次重复吞吐，其当前单次时间只作为补测记录。',
        '- 补测增加了阶段边界同步，并采用两卡并行、共享主机资源；新旧单次时间有波动，不能混为新的稳定加速比。`tradeoffs.csv` 明确标注是否同后端及各自吞吐来源。',
        '- Quest 为已接受的 Transformers OPT cached-decode 适配，完整 KV 驻留 GPU。native 保留 16-token 页、至少 3 页和前两层 Full；capped 为 ceil(30%) token 预算。它的 allocated 包含完整 GPU KV，不能描述成官方分页 offload runtime 的显存。',
        '- InfiniGen/H2O 沿用已接受 v8 实现和适配器。本次不改变算法、预算或 recent 规则。InfiniGen 的有效 cache 容量由 v8 适配器设为 prompt+gen+1，名义容量另行记录在 JSON，不能声称实际只分配了 30% 的 cache 容量。', '',
        '## 5. 测量定义', '',
        '- allocated 为 PyTorch 实际管理的张量分配峰值，reserved 为缓存分配器向 CUDA 申请的总内存峰值；二者均包括 GPU 常驻模型权重和运行缓冲。reserved 不等于活跃张量用量，也不等于 nvidia-smi 的设备总占用。',
        '- 本次在 warmup 后重置峰值，再分 prefill/decode 测量；reserved 会保留 warmup 的缓存块。上下文、CUDA context 和其他非 PyTorch 内存不在这些计数内。',
        '- 原生 FlexGen 系列的阶段边界在第一个 decode layer compute 前，包含少量初始 decode 加载；总峰值有效，细分阶段不用于严格跨 runtime 比较。Quest 在 prefill 返回后设置边界。',
        '- CPU RSS 为整个进程生命周期的峰值，含模型加载，不是纯 decode 的工作集。每组原始 JSON 保留 allocated/reserved 快照、配置、源文件哈希和运行时间；原生系列另保留输出 token 以便追溯。', '',
        '## 6. 写作建议', '',
        '可以把贡献表述为“在已有准确率/困惑度实验所支持的质量范围内，以额外 GPU 内存换取 CPU KV offload 的吞吐提升”，并同时报告质量、吞吐和峰值显存。质量结果来自独立测试，不能从显存测量反推所有生成任务都保持质量。', '',
        '串行版在同后端对照下额外 allocated 较小；异步版还包含两个按完整历史容量预留的 GPU staging 槽、预测器/embedding 和临时张量，其开销随长度上升。当前数据不支持把所有异步场景都称为“显存仅略微上升”，也不能把总差额全部算作 predictor 参数显存。', '',
        '已有五次运行的异步吞吐波动仍应保留（见 CSV 的标准差）。论文宜报告具体配置和数值，避免声称所有长度都优于 H2O/Quest，或异步永远优于串行。', '',
        f'原始补测目录：`{args.memory_dir.resolve()}`',
        f'原有同后端数据：`{historical}`',
        f'原有横向吞吐数据：`{native}`', '']
    with (args.output_dir / 'REPORT.md').open('x') as stream:
        stream.write('\n'.join(lines))
    print(f'Validated 20 new configurations; wrote {len(rows)} sourced rows and {len(comparisons)} comparisons to {args.output_dir}')


if __name__ == '__main__':
    main()
