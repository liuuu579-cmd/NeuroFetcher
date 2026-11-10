# 显存补测与论文写作依据（2026-10-29）

完整补测已通过：5 种方法 × 4 种输入长度，共 20 组。OPT-13B FP16，batch=4，生成 128 tokens；decode 吞吐统计 127 个生成步。GPU 2/3 各独立运行一个实验，分别限制 CPU 20–29/30–39，共 20 核。只新增文件。

## 1. 完整显存结果

每格为 **峰值 allocated / 峰值 reserved，GiB**（1 GiB=2^30 bytes）。Neural/Full 四行来自原有五次运行的均值，其余五行来自本次单次补测；不能把这两批数据说成同一天同一 runtime 的重复试验。

| 方法 | 512 | 1024 | 1536 | 1920 |
|---|---:|---:|---:|---:|
| flexgen | 24.762 / 24.992 | 25.410 / 25.848 | 26.374 / 27.143 | 27.310 / 28.488 |
| infinigen | 24.721 / 25.193 | 25.010 / 25.758 | 25.300 / 26.400 | 25.518 / 26.820 |
| h2o | 24.762 / 24.947 | 25.410 / 25.812 | 26.374 / 27.041 | 27.308 / 28.346 |
| quest_native | 29.634 / 36.836 | 34.540 / 35.525 | 39.444 / 40.811 | 43.203 / 44.281 |
| quest_capped | 29.635 / 36.838 | 34.541 / 35.525 | 39.445 / 40.812 | 43.206 / 44.281 |
| full_offload | 26.142 / 26.344 | 27.945 / 28.299 | 29.751 / 30.266 | 31.112 / 31.783 |
| neural_serial | 27.403 / 31.730 | 29.284 / 37.557 | 31.164 / 43.396 | 32.575 / 35.646 |
| neural_async | 29.364 / 33.684 | 32.807 / 41.072 | 36.250 / 39.521 | 38.833 / 46.650 |
| full_gpu | 26.142 / 26.320 | 27.945 / 28.254 | 29.751 / 30.201 | 31.112 / 31.699 |

## 2. 支撑核心论点的同后端比较

下面全部使用已有五次运行均值，相对于同一 custom backend 的 Full CPU KV offload。full_offload 是自定义 Full 参照，不应在论文中改名为原生 FlexGen。

| 输入长度 | 方法 | decode tok/s | 相对 Full 加速 | allocated 增量 GiB | allocated 增幅 |
|---:|---|---:|---:|---:|---:|
| 512 | neural_serial | 34.95 | 1.58× | 1.261 | 4.8% |
| 512 | neural_async | 47.82 | 2.16× | 3.222 | 12.3% |
| 1024 | neural_serial | 23.13 | 1.79× | 1.339 | 4.8% |
| 1024 | neural_async | 22.93 | 1.78× | 4.862 | 17.4% |
| 1536 | neural_serial | 16.95 | 1.84× | 1.413 | 4.7% |
| 1536 | neural_async | 16.96 | 1.84× | 6.499 | 21.8% |
| 1920 | neural_serial | 14.24 | 1.88× | 1.463 | 4.7% |
| 1920 | neural_async | 16.54 | 2.19× | 7.721 | 24.8% |

## 3. 与原生 FlexGen 的显存比较

FlexGen 这一列是新增的 original FlexGen 前后端，GPU 存放全部权重、CPU 存放 KV、overlap=false。Neural 使用项目 custom backend；这里展示实现的整体资源代价，不能单独解释为 predictor 的开销。

| 输入长度 | FlexGen allocated GiB | Neural serial 增量 GiB / % | Neural async 增量 GiB / % |
|---:|---:|---:|---:|
| 512 | 24.762 | 2.641 / 10.7% | 4.602 / 18.6% |
| 1024 | 25.410 | 3.874 / 15.2% | 7.397 / 29.1% |
| 1536 | 26.374 | 4.790 / 18.2% | 9.876 / 37.4% |
| 1920 | 27.310 | 5.265 / 19.3% | 11.523 / 42.2% |

## 4. 吞吐来源与适用范围

- `memory_and_throughput.csv` 分列保存历史吞吐和本次补测吞吐。历史 InfiniGen/H2O/Quest 采用已接受的 v8 报告口径；H2O 1024 使用原报告的 isolated repeat，首次并发记录仍保留在原目录。新增 FlexGen 尚无原先的五次重复吞吐，其当前单次时间只作为补测记录。
- 补测增加了阶段边界同步，并采用两卡并行、共享主机资源；新旧单次时间有波动，不能混为新的稳定加速比。`tradeoffs.csv` 明确标注是否同后端及各自吞吐来源。
- Quest 为已接受的 Transformers OPT cached-decode 适配，完整 KV 驻留 GPU。native 保留 16-token 页、至少 3 页和前两层 Full；capped 为 ceil(30%) token 预算。它的 allocated 包含完整 GPU KV，不能描述成官方分页 offload runtime 的显存。
- InfiniGen/H2O 沿用已接受 v8 实现和适配器。本次不改变算法、预算或 recent 规则。InfiniGen 的有效 cache 容量由 v8 适配器设为 prompt+gen+1，名义容量另行记录在 JSON，不能声称实际只分配了 30% 的 cache 容量。

## 5. 测量定义

- allocated 为 PyTorch 实际管理的张量分配峰值，reserved 为缓存分配器向 CUDA 申请的总内存峰值；二者均包括 GPU 常驻模型权重和运行缓冲。reserved 不等于活跃张量用量，也不等于 nvidia-smi 的设备总占用。
- 本次在 warmup 后重置峰值，再分 prefill/decode 测量；reserved 会保留 warmup 的缓存块。上下文、CUDA context 和其他非 PyTorch 内存不在这些计数内。
- 原生 FlexGen 系列的阶段边界在第一个 decode layer compute 前，包含少量初始 decode 加载；总峰值有效，细分阶段不用于严格跨 runtime 比较。Quest 在 prefill 返回后设置边界。
- CPU RSS 为整个进程生命周期的峰值，含模型加载，不是纯 decode 的工作集。每组原始 JSON 保留 allocated/reserved 快照、配置、源文件哈希和运行时间；原生系列另保留输出 token 以便追溯。

## 6. 写作建议

可以把贡献表述为“在已有准确率/困惑度实验所支持的质量范围内，以额外 GPU 内存换取 CPU KV offload 的吞吐提升”，并同时报告质量、吞吐和峰值显存。质量结果来自独立测试，不能从显存测量反推所有生成任务都保持质量。

串行版在同后端对照下额外 allocated 较小；异步版还包含两个按完整历史容量预留的 GPU staging 槽、预测器/embedding 和临时张量，其开销随长度上升。当前数据不支持把所有异步场景都称为“显存仅略微上升”，也不能把总差额全部算作 predictor 参数显存。

已有五次运行的异步吞吐波动仍应保留（见 CSV 的标准差）。论文宜报告具体配置和数值，避免声称所有长度都优于 H2O/Quest，或异步永远优于串行。

原始补测目录：`/home/liushenyu/infinigen/speedup/results/reviewer_memory_v9_20261029`
原有同后端数据：`/home/liushenyu/infinigen/speedup/results/reviewer_throughput_20261026`
原有横向吞吐数据：`/home/liushenyu/infinigen/speedup/results/native_flexgen_v8_20261027-v8a`
