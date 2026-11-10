# 实验与论文修订资料更新

上传日期：2026-11-10。本次整理截至2026-11-03已完成的实验与文档；上传过程没有运行新实验。

本更新保留GitHub仓库原有文件，添加评审补充实验代码、原始运行记录及写作资料。原文件与本地快照间的换行差异不作为源码更新提交。

## 显存与吞吐量

主模型OPT-13B FP16，batch 4，输入1920 tokens，生成128 tokens（127步decode）。下表显存为PyTorch峰值allocated，单位GiB；吞吐为decode tokens/s。各行显存与吞吐来自各自同一运行组。

| 方法 | Decode tok/s | allocated GiB | 测量数 |
|---|---:|---:|---:|
| InfiniGen | 9.163 | 25.518 | 1 |
| 原生FlexGen CPU-KV | 5.589 ± 0.200 | 27.310 | 5 |
| Neuro-Fetcher V16按层缓冲 | 15.273 ± 0.077 | 26.068 | 6 |
| Full GPU | 74.567 ± 0.028 | 31.112 | 5 |

相较InfiniGen，最新实现allocated显存多约2.16%、测得decode吞吐比为1.67；相较本次FlexGen配置，allocated少约4.55%、吞吐比为2.73。横向实验的输入文本、runtime、warmup、日期及重复协议不同，比值属于已测实现的观察结果。Full GPU在完整KV可驻留时仍明显更快。

同模型、同prompt的六对V12/V16复测显示：两块GPU staging从覆盖全40层改为各覆盖一层，容量从3.876降至0.097 GiB；allocated均值下降3.779 GiB；吞吐从14.827变为15.273 tok/s，约提高3.01%。输出、logits、选中索引哈希及H2D字节数一致。

## 阅读入口

- [显存、吞吐量与按层缓冲的完整写作依据](MEMORY_THROUGHPUT_PAPER_SUMMARY_20261103.md)：配置、内存拆分、统计定义、审稿意见映射和英文段落。
- [按层配对实验说明](LAYERWISE_PAPER_NOTE_20261102.md)及[原始汇总](speedup/results/layerwise_paired_v17_20261102/summary.json)。
- [机器可读资源/吞吐汇总](speedup/results/memory_throughput_summary_20261103/facts.json)、[CSV](speedup/results/memory_throughput_summary_20261103/all_runs.csv)及[原始来源hash](speedup/results/memory_throughput_summary_20261103/provenance.json)。
- [论文修订交接](PAPER_REVISION_HANDOFF_20261028.md)、[质量与系统结果](PAPER_REVISION_RESULTS_20261028.md)、[审稿意见映射](PAPER_REVISION_REVIEWER_MAP_20261028.md)。
- [OPT-6.7B家族内迁移说明](OPT_67B_TRANSFER_PAPER_NOTE_20261028.md)。
- [严格预算准确率控制](accuracy/lm_eval/strict_budget_sweep_summary_20261018.md)及[实现说明](accuracy/lm_eval/REVIEWER_CONTROLS.md)。

原写作依据中的绝对路径用于服务器本地定位；从GitHub阅读时，可通过本页相对链接或仓库文件树打开对应资料。部分文档保留历史阶段性结论，资源与吞吐的最新口径见2026-11-03汇总。

## 代码与复算

- [V16按层后端](speedup/reviewer_offload_backend_v16.py)和[benchmark](speedup/benchmark_reviewer_offload_v16.py)。
- [V17配对runner](speedup/benchmark_layerwise_paired_v17.py)和[报告生成器](speedup/report_layerwise_paired_v17.py)。
- [V18只读汇总脚本](speedup/summarize_memory_throughput_v18.py)：读取已有JSON，不导入torch、不初始化CUDA；输出目录必须尚不存在。

在已安装依赖的`infinigen`环境中，可于仓库根目录复算已有数据：

```bash
conda run -n infinigen python speedup/summarize_memory_throughput_v18.py \
  --output-dir speedup/results/memory_throughput_recheck_20261110
```

运行模型实验前仍需自行准备模型/预测器权重，并按实际机器检查脚本中的路径。上传内容不包含模型checkpoint。
