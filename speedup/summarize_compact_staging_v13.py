#!/usr/bin/env python3
"""Validate all optimized outputs and write a separately versioned report."""
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'speedup/results'
WORK = RESULTS / 'compact_staging_v11_20261029'
OUTPUT = WORK / 'report_v13'
GIB = 2**30


def read(path):
    return json.loads(path.read_text())


def main():
    assert read(WORK/'benchmark_v13/suite_status.json')['complete']
    assert read(WORK/'benchmark/suite_status.json')['complete']
    for name in ('tiny_correctness.json', 'opt13b_correctness.json'):
        assert read(WORK/name)['passed']
    rows = []
    for p in (512, 1024, 1536, 1920):
        for strategy in ('neural_serial', 'neural_async'):
            name = f'p{p}-g128-b4-{strategy}.json'
            oldpath = RESULTS/'reviewer_throughput_20261026'/name
            old = read(oldpath)
            for version, folder in (('v11', 'benchmark'), ('v13', 'benchmark_v13')):
                path = WORK/folder/name
                d = read(path)
                assert d['complete'] and d['repeats'] == len(d['runs']) == 5
                assert d['predictor_sha256'] == old['predictor_sha256']
                assert d['embedding_sha256'] == old['embedding_sha256']
                for a, b in zip(d['runs'], old['runs']):
                    assert a['output_sha256'] == b['output_sha256'], (path, 'output mismatch')
                    assert [x['index_hash'] for x in a['plans']] == [x['index_hash'] for x in b['plans']], (path, 'index mismatch')
                    assert a['h2d_kv_bytes'] == b['h2d_kv_bytes']
                    assert not a['staging_growths'], (path, 'unexpected tie growth, inspect')
                if version == 'v13':
                    assert d['embedding_shared_with_model'] and d['predictor_embedding_additional_bytes'] == 0
                    assert d['allocator_config'] == 'backend:cudaMallocAsync'
                s = d['summary']
                rows.append(dict(version=version, strategy=strategy, prompt_len=p, batch=4,
                    generation_len=128, repeats=5,
                    allocated_peak_gib=s['peak_allocated_bytes']['mean']/GIB,
                    reserved_peak_gib=s['peak_reserved_bytes']['mean']/GIB,
                    nvml_sampled_device_peak_gib=d.get('nvml_memory',{}).get('sampled_device_peak_bytes',0)/GIB if version=='v13' else '',
                    decode_tokens_per_s=s['decode_tokens_per_s']['mean'],
                    decode_tokens_per_s_stdev=s['decode_tokens_per_s']['stdev'],
                    total_s=s['total_s']['mean'],
                    gpu_staging_gib=s['gpu_staging_bytes']['mean']/GIB,
                    model_gib=d['model_tensor_bytes']/GIB,
                    predictor_gib=d['predictor_tensor_bytes']/GIB,
                    extra_embedding_gib=d.get('predictor_embedding_additional_bytes', d['predictor_embedding_bytes'])/GIB,
                    old_allocated_peak_gib=old['summary']['peak_allocated_bytes']['mean']/GIB,
                    old_decode_tokens_per_s=old['summary']['decode_tokens_per_s']['mean'],
                    output_hashes_and_index_hashes_match_old=True,
                    request_setup_s=s.get('request_setup_s',{}).get('mean',''),
                    request_total_including_setup_s=s.get('request_total_including_setup_s',{}).get('mean',''),
                    source=str(path.resolve()), old_source=str(oldpath.resolve())))
    final = [r for r in rows if r['version']=='v13']
    OUTPUT.mkdir(exist_ok=False)
    with (OUTPUT/'memory_throughput.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    old = read(RESULTS/'reviewer_throughput_20261026/p1920-g128-b4-neural_async.json')
    a = next(r for r in final if r['prompt_len']==1920 and r['strategy']=='neural_async')
    old_stage = old['runs'][0]['gpu_staging_bytes']/GIB
    lines = ['# Neuro-Fetcher 显存定位与精确缓冲优化（2026-10-30）', '',
        '全部原文件保留。新增 v11/v12/v13 文件和结果，未改预测器精度、checkpoint、索引预算、threshold ties、recent-20 或注意力算法。', '',
        '## 38.83 GiB 的来源', '',
        '以 OPT-13B FP16、batch=4、输入 1920、生成 128 为例，原异步实现的峰值 allocated 为 38.833 GiB：', '',
        '| 组成 | GiB | 依据 |', '|---|---:|---|',
        f"| 模型参数及 buffers | {a['model_gib']:.3f} | 实际模型张量求和 |",
        f"| 两个完整历史 GPU staging 槽 | {old_stage:.3f} | 原始运行记录 gpu_staging_bytes |",
        f"| FP32 预测器参数及 buffers | {a['predictor_gib']:.3f} | 实际张量求和，含位置编码 |",
        '| 独立 FP16 predictor embedding | 0.479 | 实际源张量求和 |',
        f"| 其它峰值张量及分配差额 | {old['summary']['peak_allocated_bytes']['mean']/GIB-a['model_gib']-old_stage-a['predictor_gib']-.47943115234375:.3f} | 总峰值减固定存储；含临时工作区，不是独立模块实测峰值 |", '',
        '旧槽形状为 [40层, K/V, batch4, 2048容量, 40头, 128维]，每槽 6.25 GiB。虽然实际搬运约 30%，仍按完整历史容量预留两份。这是主要可避免开销。', '',
        '## 优化内容', '',
        '- 每槽初始容量改为 min(max_length, ceil(max_length×0.3)+20)，本配置为 635 tokens；两个槽共 3.876 GiB，直接减少 8.624 GiB。若 threshold ties 超出容量，等待旧槽使用完成后扩容，保留所有原始索引。本次所有实测均未触发扩容。',
        '- predictor embedding 与主模型 FP16 token embedding 逐元素完全相同，验证后共享同一 GPU 张量，额外占用从 0.479 GiB 变为 0。预测器仍按原路径把 lookup 结果转为 FP32。',
        '- 仅缩槽仍出现较高 reserved。诊断发现默认分配器在逐步增长的变长计算中积累缓存/碎片；最终版本使用 cudaMallocAsync，并在请求开始、prefill 完成处释放未使用缓存。没有逐 token 清理缓存。',
        '- 请求开始的缓存释放在原 timing 之前，但耗时另记 request_setup_s 和 request_total_including_setup_s；prefill 后清理包含在 prefill/端到端计时中。', '',
        '## 完整工作负载结果', '',
        '四种长度 × 串行/异步 × 五次测量，每组先做一次完整 warmup。GPU2/3 各一个实验，CPU 各十核。', '',
        '| 输入 | 方法 | 原 allocated GiB | 新 allocated GiB | 新 reserved GiB | 整卡 NVML 峰值 GiB | decode tok/s ± SD |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in final:
        lines.append(f"| {r['prompt_len']} | {r['strategy']} | {r['old_allocated_peak_gib']:.3f} | {r['allocated_peak_gib']:.3f} | {r['reserved_peak_gib']:.3f} | {r['nvml_sampled_device_peak_gib']:.3f} | {r['decode_tokens_per_s']:.2f} ± {r['decode_tokens_per_s_stdev']:.2f} |")
    lines += ['', '## 与 GPU 驻留及 FlexGen 的关系', '',
        f"1920 输入下，最终异步版 allocated 为 {a['allocated_peak_gib']:.3f} GiB；原生 FlexGen 为 27.310 GiB，Full GPU 为 31.112 GiB（均采用 allocated 定义）。优化后的 allocated 已进入目标区间。完整驻留仍然更快；这里的优化解决资源浪费，没有把 CPU KV 传输成本消除。", '',
        'NVML 是每 0.1 秒采样的整卡 used，包含 CUDA context、非 PyTorch 内存以及缓存；采样从加载模型到全部重复结束，覆盖 warmup，因此不能与只计 PyTorch 张量的 27.31/31.11 GiB 直接相减。表格如实保留 allocated、reserved、NVML 三种定义，不以较低的 allocated 代替真实整卡占用。cudaMallocAsync 的部分 PyTorch 分配器统计含义与默认分配器不同，NVML 用于独立交叉核对。', '',
        '## 正确性与吞吐解释', '',
        '- 小模型：已有正确性检查全部通过；强制全部分数并列会触发扩容，新旧索引与 logits 完全一致。',
        '- OPT-13B、batch4、输入1920、生成8：v11 与原版本的串行/异步索引及 logits 位级一致。',
        '- 完整128输出：v11/v13 全部四种长度、两种调度、五次重复，输出哈希、每步索引哈希以及 H2D 字节数均与原有记录一致；v13 还执行运行时 embedding 相等检查。',
        '- 新旧吞吐测量发生在不同时间，主机共享资源及 GPU 并行使耗时波动。不能用本次均值差直接宣称缩槽带来稳定提速/减速。CSV 同时保留旧均值、新均值、标准差及来源。',
        '- 原始 38.83 GiB 数据无需删除：可以将原实现视为初版缓冲配置，将优化版资源/吞吐结果成对报告。不要把旧吞吐和新显存包装成同一次运行结果。', '',
        '## 新增文件', '',
        f'- v11 紧凑缓冲：`{ROOT / "speedup/reviewer_offload_backend_v11.py"}`',
        f'- 最终后端：`{ROOT / "speedup/reviewer_offload_backend_v12.py"}`',
        f'- 最终 benchmark：`{ROOT / "speedup/benchmark_reviewer_offload_v13.py"}`',
        f'- 批量入口：`{ROOT / "speedup/run_compact_staging_suite_v13.py"}`',
        f'- 全部结果与诊断：`{WORK}`', '']
    with (OUTPUT/'REPORT.md').open('x') as stream:stream.write('\n'.join(lines))
    print(f'Validated 16 configuration records, 80 measured runs, all old output/index hashes match; report: {OUTPUT}')


if __name__ == '__main__':
    main()
