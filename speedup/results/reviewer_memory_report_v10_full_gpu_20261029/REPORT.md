# 显存与吞吐总表：加入完整 GPU 驻留参照

配置统一为 OPT-13B FP16、batch=4、生成 128 tokens；decode 吞吐计入 127 个 decode 步。

## Full GPU：权重和完整 KV 常驻 GPU，无预取

直接使用已完成的一次完整 warmup 加五次测量，未重跑。源码检查及原始运行记录均确认：不加载预测器/其独立 embedding，CPU KV home、CPU/GPU staging 和 H2D KV 字节数全部为 0；完整 KV 在 GPU 上读写，使用完整注意力。该项沿用项目的 custom backend/SDPA 实现。

| 输入长度 | allocated 峰值 GiB | reserved 峰值 GiB | decode tok/s（均值 ± SD） | 端到端秒 |
|---:|---:|---:|---:|---:|
| 512 | 26.142 | 26.320 | 107.84 ± 0.06 | 5.10 |
| 1024 | 27.945 | 28.254 | 93.97 ± 0.04 | 6.13 |
| 1536 | 29.751 | 30.201 | 81.68 ± 0.03 | 7.39 |
| 1920 | 31.112 | 31.699 | 74.57 ± 0.03 | 8.38 |

## 所有方法

每格为 **峰值 allocated GiB / decode tok/s**。Full GPU、Full offload、Neural 来自原有五次测量均值；InfiniGen/H2O/Quest 的吞吐来自已接受的历史单次记录，显存来自新补测。原生 FlexGen 的吞吐为本次显存补测的单次时间，标注 †。

| 方法 | 512 | 1024 | 1536 | 1920 |
|---|---:|---:|---:|---:|
| Full GPU（完整 KV，无预取） | 26.142 / 107.84 | 27.945 / 93.97 | 29.751 / 81.68 | 31.112 / 74.57 |
| Full CPU KV offload（自定义后端） | 26.142 / 22.14 | 27.945 / 12.90 | 29.751 / 9.23 | 31.112 / 7.57 |
| 原生 FlexGen CPU KV offload | 24.762 / 16.90† | 25.410 / 9.64† | 26.374 / 7.17† | 27.310 / 5.93† |
| InfiniGen | 24.721 / 23.57 | 25.010 / 14.10 | 25.300 / 9.85 | 25.518 / 7.23 |
| H2O | 24.762 / 37.00 | 25.410 / 28.39 | 26.374 / 21.04 | 27.308 / 13.75 |
| Quest native-rule | 29.634 / 78.79 | 34.540 / 68.76 | 39.444 / 57.39 | 43.203 / 50.68 |
| Quest capped | 29.635 / 75.31 | 34.541 / 65.86 | 39.445 / 55.13 | 43.206 / 48.97 |
| Neuro-Fetcher serial | 27.403 / 34.95 | 29.284 / 23.13 | 31.164 / 16.95 | 32.575 / 14.24 |
| Neuro-Fetcher async | 29.364 / 47.82 | 32.807 / 22.93 | 36.250 / 16.96 | 38.833 / 16.54 |

## 解释与写作口径

- Full GPU 在当前能容纳模型和完整 KV 的配置下速度最快，可作为无 CPU KV 传输的性能参照。当前 Neuro-Fetcher 的预留缓冲和预测器使其显存高于这个 Full GPU 实现；现有数据不支持相对 Full GPU 的显存节省或吞吐提升。
- Neuro-Fetcher 的吞吐收益应相对 CPU KV offload 参照报告，且应同时列出额外显存。完整 GPU 驻留参照有助于读者看到当前实现与无传输路径之间的差距。
- Full offload 与 Full GPU 的 allocated 很接近，是因为现有 Full offload 实现为所有层预留了完整历史容量的 GPU staging；offload 这一名称不代表当前实现必然节省 GPU 内存。
- 各方法运行路径不同，不能把整张表的差异全部归因于预取算法。Quest 当前适配保留完整 GPU KV，不是官方分页卸载 runtime。
- allocated 是 PyTorch 张量分配峰值；reserved 包含分配器缓存，不等同于 nvidia-smi 总占用。原有测量定义、阶段边界、H2O 1024 repeat 选择及波动说明沿用 v9 报告。

详细测量说明：`/home/liushenyu/infinigen/speedup/results/reviewer_memory_report_v9_20261029/REPORT.md`
原始 Full GPU 数据：`/home/liushenyu/infinigen/speedup/results/reviewer_throughput_20261026`

本版本只新增汇总脚本、报告和 CSV，未修改原有代码或结果。
