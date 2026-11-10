#!/usr/bin/env python3
"""Validate paired results and write an additive paper implementation note."""
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'speedup/results/layerwise_paired_v17_20261102'


def main():
    data = json.loads((OUT / 'summary.json').read_text())
    assert data['complete'] and data['all_validation_bitwise_equal']
    assert len(data['pairs']) == 6
    records = [json.loads(p.read_text()) for p in sorted(OUT.glob('pair*.json'))]
    assert len(records) == 12
    signature = records[0]['validation']
    for record in records:
        assert record['validation'] == signature
        assert len(signature['index_hashes']) == 127
    old, new = (data['summary'][label] for label in ('bulk_v12', 'layer_v16'))
    gib = 2 ** 30
    change = 100 * (data['ratio_of_mean_throughputs'] - 1)
    savings = statistics.mean(p['allocated_saved_gib'] for p in data['pairs'])
    old_stage = old['gpu_staging_bytes']['mean'] / gib
    new_stage = new['gpu_staging_bytes']['mean'] / gib
    lines = [
        '# 按层 KV staging：配对复测与论文修改依据', '',
        '日期：2026-11-02。只新增文件，未修改 V12/V16 源码、权重或历史结果。', '',
        '## 当前结论', '',
        '从代码和运行结果均确认 V16 实现逐层 KV 搬运，并在两个一层 GPU 缓冲间交替。'
        '它保留跨解码步预测和 CPU gather，但当前步 H2D 在该步开始时启动，'
        '后续层的搬运与层计算流水重叠。它没有把所有层的 KV 常驻 GPU，也没有改变选择算法。', '',
        f'同进程、同GPU交错六对复测中，V16 在 {data["number_of_pairs_layer_faster"]}/6 对中更快，'
        f'吞吐均值比为 {data["ratio_of_mean_throughputs"]:.4f}，即 {change:+.2f}%。'
        '不能把此前不同时段的 14.43 与 17.73 token/s 相除得到的约23%差距'
        '全部归因于按层调度。此次支持的是按层缓冲与调度实现整体的实测差异，'
        '未进一步分离缓冲容量、同步和并发竞争各自的因果贡献。', '',
        '## 控制条件', '',
        '- 单张 GPU 3、固定 CPU 30–39，共10核；infinigen 环境；'
        'cudaMallocAsync。检查每次运行前后没有其它进程占用 GPU 3。',
        '- OPT-13B FP16、batch 4、输入1920、输出128，其中127步 decode；'
        '同一个模型、同一份共享 embedding、同一个 FP32 单层4-head predictor。',
        '- 各版本一次完整 warmup，随后六个相邻配对：AB、BA、AB、BA、AB、BA。'
        '每个版本共六次正式测量；切换时恢复模型原始 forward；无 profiler。',
        '- 所有正式运行完整输出、prefill/decode logits 位级哈希、'
        '127步索引哈希以及 H2D 字节数全部相同。', '',
        '| 版本 | Decode token/s，均值±样本SD | allocated峰值均值 GiB | reserved峰值均值 GiB | GPU staging GiB |',
        '|---|---:|---:|---:|---:|',
    ]
    for label, row in (('V12 全层缓冲', old), ('V16 按层缓冲', new)):
        t = row['decode_tokens_per_s']
        lines.append(
            f'| {label} | {t["mean"]:.3f} ± {t["stdev"]:.3f} | '
            f'{row["peak_allocated_bytes"]["mean"]/gib:.3f} | '
            f'{row["peak_reserved_bytes"]["mean"]/gib:.3f} | '
            f'{row["gpu_staging_bytes"]["mean"]/gib:.3f} |')
    lines += ['', '| 配对 | 顺序 | V12 token/s | V16 token/s | V16/V12 |',
              '|---:|---|---:|---:|---:|']
    for pair in data['pairs']:
        order = ' → '.join(pair['order'])
        lines.append(
            f'| {pair["pair"]} | {order} | {pair["bulk_tps"]:.3f} | '
            f'{pair["layer_tps"]:.3f} | {pair["layer_divided_by_bulk"]:.4f} |')
    lines += ['', f'配对 allocated 平均减少 {savings:.3f} GiB。GPU staging 从 '
              f'{old_stage:.3f} 降至 {new_stage:.3f} GiB，容量减少 '
              f'{100*(1-new_stage/old_stage):.1f}%。'
              '这是容量和张量分配层面的明确收益；shared host的运行时间仍可能波动。',
              '', '本轮 allocated 是共用模型且反复创建 engine 的同进程测量，'
              '与旧独立基准中的 29.728/25.949 GiB 峰值有小量差异。'
              '论文不能把不同轮次的显存与吞吐拼成同一次测量。'
              'reserved 在 cudaMallocAsync 下与默认分配器的含义也不完全相同。', '',
              '## 论文中是否需要写', '',
              '**采用按层版的最终资源与吞吐结果，就需要交代此实现。**'
              '以系统实现细节和内存消融呈现即可，无需增加新的核心贡献。'
              '如不采用按层版本结果，则无需把这次优化写成主方法。', '',
              '1. **方法/实现（Section 3.2）**：分别描述跨步 predictor/CPU packing 与'
              '层内 GPU KV staging。原生 recent-20 和预测器设置不变。'
              '“Ping/Pong”需说明是 CPU 的按步 pinned staging，还是 GPU 的按层缓冲，'
              '两种角色不能混称。',
              '2. **Figure 6**：原图按整解码步交换 GPU buffer 的示意应更新，'
              '或新增内层按层流水示意。两块GPU缓冲各容纳一层的选中 KV。'
              '不要继续写 on-chip SRAM；实际是 GPU device memory。',
              '3. **实验**：放 V12/V16 小消融表，报告同轮的吞吐、allocated 和缓冲容量。'
              '叙述重点是降低缓冲显存且该配置下吞吐未受损；'
              '不将旧跨时段23%差距作为受控实验结论，也不外推到其它长度/批次。',
              '4. **评审回复**：此结果直接支持 R1 第3条要求解释 Ping/Pong 物理缓冲显存，'
              '以及 R2 Major 4 关于缺少 memory results 的意见。'
              '它补充系统实现和资源证据，不替代 predictor/transfer/LLM 分项成本与'
              'serial/async 对照。', '',
              '原评审（第1页）：“Please also report predictor and transfer costs during '
              'co-execution and clarify the physical memory used by the Ping/Pong buffers, '
              'since the stated overlap condition alone does not establish zero overhead.”', '',
              '### 可用于论文的英文表述', '',
              '> We use layer-streamed GPU KV staging with two reusable device buffers, '
              'each sized for one layer of selected KV entries. The buffers alternate '
              'across layers, with CUDA events enforcing transfer and consumer dependencies. '
              'Prediction and CPU packing for the following decoding step proceed '
              'asynchronously, while the current step transfers KV entries incrementally '
              'as its layers execute. This design preserves the FP32 predictor and '
              'the selected indices and avoids retaining two all-layer staging copies '
              'on the GPU.', '',
              f'> In a balanced six-pair comparison on OPT-13B (batch size 4, '
              f'1920 input tokens, 128 generated tokens), layer-streamed staging '
              f'reduced GPU staging capacity from {old_stage:.3f} to {new_stage:.3f} GiB '
              f'and changed mean decode throughput from '
              f'{old["decode_tokens_per_s"]["mean"]:.2f} to '
              f'{new["decode_tokens_per_s"]["mean"]:.2f} tokens/s. '
              'Generated outputs, logits, selected-index hashes, and transferred KV bytes '
              'were identical across the compared runs. These results characterize '
              'this workload and implementation rather than a universal scheduling speedup.', '',
              '## 可复现资料', '',
              f'- 配对 runner：{ROOT / "speedup/benchmark_layerwise_paired_v17.py"}',
              f'- 完整汇总与每次原始结果：{OUT}',
              f'- 此报告生成器：{ROOT / "speedup/report_layerwise_paired_v17.py"}',
              '- manifest.json 包含配置、CPU affinity、prompt/checkpoint/source 哈希；'
              '每次结果含GPU温度/频率/进程及主机负载快照。',
              '- 原始论文 Figure 6 在 Manuscript.pdf 第7页；'
              '评审缓冲显存问题在 reviews.pdf 第1页，缺少memory结果在第2页。', '']
    report = '\n'.join(lines)
    for path in (OUT / 'REPORT.md', ROOT / 'LAYERWISE_PAPER_NOTE_20261102.md'):
        with path.open('x') as stream:
            stream.write(report)
    print(report)


if __name__ == '__main__':
    main()
