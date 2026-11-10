#!/usr/bin/env python3
"""Expose the existing no-prefetch GPU-resident reference in a new report."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'speedup/results'
PREVIOUS = RESULTS / 'reviewer_memory_report_v9_20261029'
OUTPUT = RESULTS / 'reviewer_memory_report_v10_full_gpu_20261029'
LENGTHS = (512, 1024, 1536, 1920)
LABELS = {
    'full_gpu': 'Full GPU（完整 KV，无预取）',
    'full_offload': 'Full CPU KV offload（自定义后端）',
    'flexgen': '原生 FlexGen CPU KV offload',
    'infinigen': 'InfiniGen', 'h2o': 'H2O',
    'quest_native': 'Quest native-rule', 'quest_capped': 'Quest capped',
    'neural_serial': 'Neuro-Fetcher serial', 'neural_async': 'Neuro-Fetcher async',
}


def main():
    with (PREVIOUS / 'memory_and_throughput.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    by_key = {(r['method'], int(r['prompt_len'])): r for r in rows}
    assert len(by_key) == len(rows) == 36
    full = {}
    for p in LENGTHS:
        path = RESULTS / f'reviewer_throughput_20261026/p{p}-g128-b4-full_gpu.json'
        d = json.loads(path.read_text())
        assert d['complete'] and d['strategy'] == 'full_gpu'
        assert d['repeats'] == len(d['runs']) == 5
        assert d['batch_size'] == 4 and d['generation_len'] == 128
        assert d['predictor_path'] is None and d['embedding_path'] is None
        for run in d['runs']:
            assert run['full_gpu_kv_bytes'] > 0
            assert all(run[k] == 0 for k in ('cpu_kv_home_bytes', 'pinned_staging_bytes',
                                           'gpu_staging_bytes', 'h2d_kv_bytes'))
        assert float(by_key['full_gpu', p]['peak_allocated_gib']) == d['summary']['peak_allocated_bytes']['mean'] / 2**30
        full[p] = d
    lines = ['# 显存与吞吐总表：加入完整 GPU 驻留参照', '',
        '配置统一为 OPT-13B FP16、batch=4、生成 128 tokens；decode 吞吐计入 127 个 decode 步。', '',
        '## Full GPU：权重和完整 KV 常驻 GPU，无预取', '',
        '直接使用已完成的一次完整 warmup 加五次测量，未重跑。源码检查及原始运行记录均确认：不加载预测器/其独立 embedding，CPU KV home、CPU/GPU staging 和 H2D KV 字节数全部为 0；完整 KV 在 GPU 上读写，使用完整注意力。该项沿用项目的 custom backend/SDPA 实现。', '',
        '| 输入长度 | allocated 峰值 GiB | reserved 峰值 GiB | decode tok/s（均值 ± SD） | 端到端秒 |',
        '|---:|---:|---:|---:|---:|']
    for p, d in full.items():
        s = d['summary']
        lines.append(f"| {p} | {s['peak_allocated_bytes']['mean']/2**30:.3f} | {s['peak_reserved_bytes']['mean']/2**30:.3f} | {s['decode_tokens_per_s']['mean']:.2f} ± {s['decode_tokens_per_s']['stdev']:.2f} | {s['total_s']['mean']:.2f} |")
    lines += ['', '## 所有方法', '',
        '每格为 **峰值 allocated GiB / decode tok/s**。Full GPU、Full offload、Neural 来自原有五次测量均值；InfiniGen/H2O/Quest 的吞吐来自已接受的历史单次记录，显存来自新补测。原生 FlexGen 的吞吐为本次显存补测的单次时间，标注 †。', '',
        '| 方法 | 512 | 1024 | 1536 | 1920 |', '|---|---:|---:|---:|---:|']
    ordered = []
    for method, label in LABELS.items():
        cells = []
        for p in LENGTHS:
            r = dict(by_key[method, p])
            speed = r['historical_decode_tokens_per_s'] or r['memory_run_decode_tokens_per_s']
            cells.append(f"{float(r['peak_allocated_gib']):.3f} / {float(speed):.2f}" + ('†' if method == 'flexgen' else ''))
            ordered.append(dict(display_name=label, **r))
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 解释与写作口径', '',
        '- Full GPU 在当前能容纳模型和完整 KV 的配置下速度最快，可作为无 CPU KV 传输的性能参照。当前 Neuro-Fetcher 的预留缓冲和预测器使其显存高于这个 Full GPU 实现；现有数据不支持相对 Full GPU 的显存节省或吞吐提升。',
        '- Neuro-Fetcher 的吞吐收益应相对 CPU KV offload 参照报告，且应同时列出额外显存。完整 GPU 驻留参照有助于读者看到当前实现与无传输路径之间的差距。',
        '- Full offload 与 Full GPU 的 allocated 很接近，是因为现有 Full offload 实现为所有层预留了完整历史容量的 GPU staging；offload 这一名称不代表当前实现必然节省 GPU 内存。',
        '- 各方法运行路径不同，不能把整张表的差异全部归因于预取算法。Quest 当前适配保留完整 GPU KV，不是官方分页卸载 runtime。',
        '- allocated 是 PyTorch 张量分配峰值；reserved 包含分配器缓存，不等同于 nvidia-smi 总占用。原有测量定义、阶段边界、H2O 1024 repeat 选择及波动说明沿用 v9 报告。', '',
        f'详细测量说明：`{PREVIOUS / "REPORT.md"}`',
        f'原始 Full GPU 数据：`{RESULTS / "reviewer_throughput_20261026"}`', '',
        '本版本只新增汇总脚本、报告和 CSV，未修改原有代码或结果。', '']
    OUTPUT.mkdir(exist_ok=False)
    with (OUTPUT / 'memory_and_throughput.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    with (OUTPUT / 'REPORT.md').open('x') as stream:
        stream.write('\n'.join(lines))
    print(f'Validated 4 Full GPU configurations (5 runs each); wrote {OUTPUT}')


if __name__ == '__main__':
    main()
