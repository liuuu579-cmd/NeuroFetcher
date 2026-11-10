# 显存优化结论及改稿补充（2026-10-30）

本说明优先于同目录 report_v13/REPORT.md 中将 v13 串行/异步统称“最终版本”的表述。v13 是异步版可用候选；v13 串行版出现严重性能退化，不推荐用于论文主结果。所有旧文件、测试及失败/退化记录保留。

## 结论

以 OPT-13B FP16、batch4、1920 输入、128 输出为例，异步版 allocated 从 **38.833 GiB 降到 29.728 GiB**，减少 **9.104 GiB（23.4%）**。与原生 FlexGen 27.310 GiB、Full GPU 31.112 GiB 使用相同 allocated 定义比较，已进入用户提出的区间。v13 reserved 均值为 30.575 GiB，NVML 整卡采样峰值为 31.791 GiB；NVML 不应与 allocated 混比。

当前全部 GPU 驻留仍更快。优化后的显存优势相对 Full GPU 只有约 1.38 GiB，不能据此夸大成显著扩大可运行规模。当前工作在 GPU 容量受限、必须卸载 KV 的条件下才有吞吐优化的实际用途；还应将系统整体资源代价和收益如实写明。

## 原来 38.83 GiB 用在哪里

| 组成 | GiB |
|---|---:|
| OPT-13B 模型参数及 buffers | 23.941 |
| 两个按完整历史容量分配的 GPU Ping/Pong 槽 | 12.500 |
| FP32 预测器参数及 buffers | 0.564 |
| 独立 FP16 predictor embedding | 0.479 |
| 其它临时张量及分配差额 | 约 1.348 |

最后一项是总峰值扣除固定存储的余项，并非独立组件峰值。主要浪费是只选约 30% token，却为两个槽各预留完整 2048-token 容量，每槽 6.25 GiB。

## 实际改动

1. 每槽初始容量改为 ceil(2048×0.3)+20=635 tokens；两个槽共 3.876 GiB，节省 8.624 GiB。recent-20、阈值并列和原索引选择全部保留；实际超限会安全扩容。本次所有完整工作负载都未扩容。全部并列的强制测试验证了扩容路径。
2. predictor embedding 与 OPT embedding 逐元素相同，运行时验证后共享，额外节省 0.479 GiB。未量化或更改 predictor 的 FP32 计算。
3. 异步版采用 cudaMallocAsync，避免默认分配器的变长计算缓存膨胀；请求开始和 prefill 完成处释放未使用缓存，没有每 token 清缓存。设置及其耗时都有记录。

## 异步版 v13 完整结果

每组一次完整 warmup、五次测量。reserved 为五次运行计数的峰值均值；NVML 为加载、warmup 和全部测量期间每 0.1 秒采样的整卡最高值，包含 CUDA context 等非 PyTorch 占用，并可能漏掉短瞬时峰值。

| 输入长度 | allocated GiB | reserved GiB | NVML 整卡 GiB | decode tok/s ± SD |
|---:|---:|---:|---:|---:|
| 512 | 26.270 | 26.725 | 27.963 | 49.51 ± 1.60 |
| 1024 | 27.529 | 28.406 | 29.877 | 28.80 ± 6.25 |
| 1536 | 28.789 | 29.594 | 30.916 | 15.03 ± 3.65 |
| 1920 | 29.728 | 30.575 | 31.791 | 14.43 ± 2.33 |

四种长度每组五次运行，输出哈希、所有 decode 步的索引哈希和 H2D KV 字节数都与原结果一致。小模型及 OPT-13B 的新旧 v11 logits 位级一致。后续共享 embedding 使用运行时逐元素相等检查，完整生成的输出和索引同样验证通过。

1920 输入的平均 decode 吞吐为 14.43±2.33 tok/s，旧异步结果为 16.54±3.98 tok/s；分属不同时间的测量，不宣称二者吞吐等效或稳定提速。相对于历史同后端 Full offload 的 7.57 tok/s，均值比约 1.91；需要保留其历史测量来源。新结果包含相应的 prefill 和端到端时间，不能将旧吞吐直接与新显存拼为同一次实验。新端到端均值 42.72 秒，计入请求设置清理为 42.84 秒，单独记录的设置均值约 0.124 秒。

## 串行版与不成功的诊断

- v11（缩槽、不共享 embedding、默认分配器）：1920 输入 allocated 28.263 GiB，decode 12.54±0.49 tok/s，但 reserved 43.506 GiB，仍有缓存占用问题。
- v13（共享 embedding、cudaMallocAsync）：四种长度均完成，输出正确，但串行 decode 只有约 1.3 tok/s。这是明确的性能退化，不能推荐为最终串行方案。此前“卡死”的判断不准确，进程自行完成；停止命令因进程已退出未终止任何进程。
- v14（共享 embedding、默认分配器、29.5 GiB 分配器预算）：仅针对1920输入做五次诊断。allocated 27.783 GiB，reserved 29.471 GiB，decode 9.22±0.06 tok/s；输出和索引仍与原结果一致。整卡生命周期 NVML 峰值 33.925 GiB，包含初始化过程，不宣称整卡低于31 GiB。该限制导致明显的性能代价，保留作诊断，未推广到其它长度。
- max_split_size_mb:128 未解决 reserved 膨胀；仅阶段 empty_cache 也未解决。这些原始记录保存在 allocator_diagnostic。

## 复现与文件

使用 infinigen 环境，单卡实验，GPU2/3各十核。不应直接用 v13 双线路 suite 作为推荐的串行实现：其中串行慢路径是已知问题。异步成功结果来自 benchmark_reviewer_offload_v13.py、reviewer_offload_backend_v12.py，环境设置 PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync。启动新任务前检查 GPU 空闲，每次使用新的输出文件路径。

详细组件、逐版本结果：report_v13/REPORT.md、report_v13/memory_throughput.csv。
异步最终原始数据：benchmark_v13/p*-g128-b4-neural_async.json。
串行预算诊断：benchmark_v14/p1920-g128-b4-neural_serial.json。
原有源码、checkpoint、论文交接文档和历史结果均未修改；所有实现均为新增版本。
