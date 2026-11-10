# 按层 KV staging：OPT-13B 实测（2026-10-31）

改动前独立 Git 快照：refs/snapshots/current-20261031-before-layerwise（42769f54e7919c4df0f7265e3dcb0e89fd2a952d）。原分支、原始源码和历史结果未改。

配置：OPT-13B FP16；batch 4；输入1920；生成128，其中127步 decode；学习式阈值 30% 加原有 recent-20。预测器仍为 FP32 的单层、4-head Transformer。Neural 每项1次完整 warmup + 5次测量。

| 方法 | decode token/s，均值±SD | PyTorch allocated峰值 GiB | reserved峰值 GiB | GPU staging GiB |
|---|---:|---:|---:|---:|
| 原紧凑版 V12 | 14.43 ± 2.33 | 29.728 | 30.575 | 3.876 |
| 按层版 V16 | 17.73 ± 0.16 | 25.949 | 27.050 | 0.097 |
| 原生 FlexGen | 5.59 ± 0.20 | 27.310 | 28.488 | 不适用 |

按层版较 V12 降低 allocated 3.779 GiB，且低于原生 FlexGen 1.361 GiB；与 FlexGen 的跨实现 decode 吞吐均值比为 3.17×。FlexGen 是五次独立进程、每次一输出 token warmup；Neural 是一进程中一次完整 warmup 后五次测量，不能把比值说成同后端算法单因素加速。

V16 NVML 整卡采样峰值为 28.604 GiB，V12 为 31.791 GiB；该口径包含 CUDA 上下文及其它设备占用，不可和 FlexGen 的 PyTorch allocated 数字直接相减。

## 正确性与适用范围

- 小模型：包含全索引并列触发动态扩容、完整历史缓存 OPT 对照和串异步一致性，均通过。
- OPT-13B 强制8输出 token：V16 对 V12 两种调度的索引和 logits 位级一致。
- 五次完整128输出：每次输出哈希、127步逐步索引哈希、选择密度和 H2D KV 字节与 V12 完全一致；没有触发扩容。
- V16 只改 GPU KV staging 的容量与调度：两个一层缓冲按层交替；CPU KV home 和两套 pinned CPU staging 未变，FP32 predictor、模型权重及注意力计算均未变。
- 按层 DMA 事件跨越层计算间隙，原始记录中的 dma_ms 不能当作纯 PCIe 传输时间。
- V12/V16 在不同时间测量。V16 的吞吐提升是观察结果；若要单独归因于调度，需进行交错配对重测。

原始记录：/home/liushenyu/infinigen/speedup/results/layerwise_v16_20261031/p1920-g128-b4-neural_async.json；正确性：/home/liushenyu/infinigen/speedup/results/layerwise_v16_20261031/tiny_correctness.json 和 /home/liushenyu/infinigen/speedup/results/layerwise_v16_20261031/opt13b_correctness.json；旧版与 FlexGen 数据路径见 comparison.json。
