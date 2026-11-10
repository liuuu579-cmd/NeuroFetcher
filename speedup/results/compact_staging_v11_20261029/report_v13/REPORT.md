# Neuro-Fetcher 显存定位与精确缓冲优化（2026-10-30）

全部原文件保留。新增 v11/v12/v13 文件和结果，未改预测器精度、checkpoint、索引预算、threshold ties、recent-20 或注意力算法。

## 38.83 GiB 的来源

以 OPT-13B FP16、batch=4、输入 1920、生成 128 为例，原异步实现的峰值 allocated 为 38.833 GiB：

| 组成 | GiB | 依据 |
|---|---:|---|
| 模型参数及 buffers | 23.941 | 实际模型张量求和 |
| 两个完整历史 GPU staging 槽 | 12.500 | 原始运行记录 gpu_staging_bytes |
| FP32 预测器参数及 buffers | 0.564 | 实际张量求和，含位置编码 |
| 独立 FP16 predictor embedding | 0.479 | 实际源张量求和 |
| 其它峰值张量及分配差额 | 1.348 | 总峰值减固定存储；含临时工作区，不是独立模块实测峰值 |

旧槽形状为 [40层, K/V, batch4, 2048容量, 40头, 128维]，每槽 6.25 GiB。虽然实际搬运约 30%，仍按完整历史容量预留两份。这是主要可避免开销。

## 优化内容

- 每槽初始容量改为 min(max_length, ceil(max_length×0.3)+20)，本配置为 635 tokens；两个槽共 3.876 GiB，直接减少 8.624 GiB。若 threshold ties 超出容量，等待旧槽使用完成后扩容，保留所有原始索引。本次所有实测均未触发扩容。
- predictor embedding 与主模型 FP16 token embedding 逐元素完全相同，验证后共享同一 GPU 张量，额外占用从 0.479 GiB 变为 0。预测器仍按原路径把 lookup 结果转为 FP32。
- 仅缩槽仍出现较高 reserved。诊断发现默认分配器在逐步增长的变长计算中积累缓存/碎片；最终版本使用 cudaMallocAsync，并在请求开始、prefill 完成处释放未使用缓存。没有逐 token 清理缓存。
- 请求开始的缓存释放在原 timing 之前，但耗时另记 request_setup_s 和 request_total_including_setup_s；prefill 后清理包含在 prefill/端到端计时中。

## 完整工作负载结果

四种长度 × 串行/异步 × 五次测量，每组先做一次完整 warmup。GPU2/3 各一个实验，CPU 各十核。

| 输入 | 方法 | 原 allocated GiB | 新 allocated GiB | 新 reserved GiB | 整卡 NVML 峰值 GiB | decode tok/s ± SD |
|---:|---|---:|---:|---:|---:|---:|
| 512 | neural_serial | 27.403 | 25.616 | 26.050 | 27.203 | 1.40 ± 0.00 |
| 512 | neural_async | 29.364 | 26.270 | 26.725 | 27.963 | 49.51 ± 1.60 |
| 1024 | neural_serial | 29.284 | 26.404 | 27.113 | 28.459 | 1.33 ± 0.00 |
| 1024 | neural_async | 32.807 | 27.529 | 28.406 | 29.877 | 28.80 ± 6.25 |
| 1536 | neural_serial | 31.164 | 27.193 | 28.131 | 29.467 | 1.31 ± 0.02 |
| 1536 | neural_async | 36.250 | 28.789 | 29.594 | 30.916 | 15.03 ± 3.65 |
| 1920 | neural_serial | 32.575 | 27.782 | 28.812 | 30.280 | 1.27 ± 0.00 |
| 1920 | neural_async | 38.833 | 29.728 | 30.575 | 31.791 | 14.43 ± 2.33 |

## 与 GPU 驻留及 FlexGen 的关系

1920 输入下，最终异步版 allocated 为 29.728 GiB；原生 FlexGen 为 27.310 GiB，Full GPU 为 31.112 GiB（均采用 allocated 定义）。优化后的 allocated 已进入目标区间。完整驻留仍然更快；这里的优化解决资源浪费，没有把 CPU KV 传输成本消除。

NVML 是每 0.1 秒采样的整卡 used，包含 CUDA context、非 PyTorch 内存以及缓存；采样从加载模型到全部重复结束，覆盖 warmup，因此不能与只计 PyTorch 张量的 27.31/31.11 GiB 直接相减。表格如实保留 allocated、reserved、NVML 三种定义，不以较低的 allocated 代替真实整卡占用。cudaMallocAsync 的部分 PyTorch 分配器统计含义与默认分配器不同，NVML 用于独立交叉核对。

## 正确性与吞吐解释

- 小模型：已有正确性检查全部通过；强制全部分数并列会触发扩容，新旧索引与 logits 完全一致。
- OPT-13B、batch4、输入1920、生成8：v11 与原版本的串行/异步索引及 logits 位级一致。
- 完整128输出：v11/v13 全部四种长度、两种调度、五次重复，输出哈希、每步索引哈希以及 H2D 字节数均与原有记录一致；v13 还执行运行时 embedding 相等检查。
- 新旧吞吐测量发生在不同时间，主机共享资源及 GPU 并行使耗时波动。不能用本次均值差直接宣称缩槽带来稳定提速/减速。CSV 同时保留旧均值、新均值、标准差及来源。
- 原始 38.83 GiB 数据无需删除：可以将原实现视为初版缓冲配置，将优化版资源/吞吐结果成对报告。不要把旧吞吐和新显存包装成同一次运行结果。

## 新增文件

- v11 紧凑缓冲：`/home/liushenyu/infinigen/speedup/reviewer_offload_backend_v11.py`
- 最终后端：`/home/liushenyu/infinigen/speedup/reviewer_offload_backend_v12.py`
- 最终 benchmark：`/home/liushenyu/infinigen/speedup/benchmark_reviewer_offload_v13.py`
- 批量入口：`/home/liushenyu/infinigen/speedup/run_compact_staging_suite_v13.py`
- 全部结果与诊断：`/home/liushenyu/infinigen/speedup/results/compact_staging_v11_20261029`
