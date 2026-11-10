# Neuro-Fetcher：显存、吞吐量与按层缓冲的论文修订依据

整理日期：2026-11-03。本文汇总已有实验，不启动新实验。原源码、权重、历史结果和既有交接文档全部保留。

## 1. 目前可以采用的结论

**我们已经得到一个显存接近 InfiniGen、低于本次原生 FlexGen CPU-KV 配置，同时解码吞吐更高的实现。**“多一点内存换吞吐”对 InfiniGen 成立；对 FlexGen，最新结果更好：显存更少，吞吐更高。

以下采用最新六对受控复测中的按层版 Neuro-Fetcher（V16，由 V17 runner 测量），与已有原生基线比较。共同工作负载形状为 OPT-13B FP16、batch=4、输入1920 tokens、生成128 tokens。

| 对照 | Neuro-Fetcher 的峰值 allocated 差值 | 相对显存变化 | Decode 吞吐均值比 | 可采用表述 |
|---|---:|---:|---:|---|
| InfiniGen | +0.550 GiB | +2.16% | 1.667× | 约多用2.2%的GPU张量显存，测得约1.67倍解码吞吐 |
| 原生 FlexGen CPU-KV | −1.242 GiB | −4.55% | 2.733× | 峰值GPU张量显存更低，测得约2.73倍解码吞吐 |
| Full GPU | −5.044 GiB | −16.21% | 0.205× | 节省显存，但完整GPU驻留明显更快 |

这里的“显存”指 **PyTorch peak allocated GPU memory**，包含模型、预测器及工作区；不代表CPU内存，也不等于 `nvidia-smi` 的整卡占用。单位为 GiB（2³⁰ bytes）。

上述横向比值是已测实现的结果，输入长度/批次相同，但基线的输入文本、运行后端、日期、warmup及重复方式不同。论文应披露这些条件，不能把比值写成同后端、单一算法因素的因果加速。

论文价值可以概括为：**在已有准确率/困惑度实验所刻画的质量范围内，学习式选择与异步执行提供了较好的CPU KV offload显存—吞吐折中；按层GPU缓冲进一步降低了系统开销。**质量结论仍引用准确率/困惑度结果，不能仅靠本次吞吐与哈希一致性代替。

## 2. 最新主表：显存与吞吐来自各自同一组运行

| 方法 | 正式测量数 | Decode tok/s，均值±样本SD | allocated峰值均值 GiB | reserved峰值均值 GiB | Prefill s | Decode s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|
| InfiniGen，原生机制 | 1 | 9.163，单次 | 25.518 | 26.820 | 1.816 | 55.441 | 57.257 |
| FlexGen，权重GPU/KV CPU | 5 | 5.589 ± 0.200 | 27.310 | 28.488 | 2.438 | 90.991 | 93.429 |
| **Neuro-Fetcher V16，按层GPU缓冲** | **6** | **15.273 ± 0.077** | **26.068** | **27.141** | **5.438** | **33.262** | **38.700** |
| Full GPU，完整KV驻留 | 5 | 74.567 ± 0.028 | 31.112 | 31.699 | 1.565 | 6.813 | 8.377 |

各时间列为逐次时间的均值；吞吐列为逐次吞吐的均值，不能用“508 / 平均decode秒数”替代它。

InfiniGen这一行采用**显存补测同次运行的9.163 tok/s**。旧原生吞吐表中还存在7.232 tok/s的历史结果，保留原记录，但本主表不把旧吞吐和新显存拼接成同一次测量，也不通过选择更慢的历史基线放大比值。

端到端总时延的相应比值为：InfiniGen / V16 = 1.480，FlexGen / V16 = 2.414。**1.667×和2.733×是decode吞吐比，不是端到端时延加速比。**Neuro-Fetcher的prefill比这两项原生基线更慢，应如实保留分项时间。

### 测量范围与配置

- 生成128个token时，第一个输出由prefill得到，其后127步decode。Decode总token数为4×127=508；包含prefill的输出token数为4×128=512。
- Decode throughput = 508 / decode wall time；end-to-end output throughput = 512 / (prefill + decode)。`total_s`包含prefill与decode，不含checkpoint加载、warmup和单列的请求初始化。初始化另记为 `request_setup_s`，包含初始化的时间为 `request_total_including_setup_s`。
- GPU为NVIDIA RTX 6000 Ada；每项实验使用单张GPU。最新配对及FlexGen重复使用GPU3、CPU30–39，共10核；InfiniGen显存补测使用GPU2、CPU20–29。不同物理卡及共享主机条件是横向比较的运行条件。
- 主模型：`/home/liushenyu/opt-13b`；原生FlexGen/InfiniGen使用对应的 `/home/liushenyu/opt-13b-flexgen-np` 权重格式。不是OPT-6.7B迁移实验。
- Predictor checkpoint：`/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth`。最新吞吐实验使用 **1层Transformer encoder、4 heads、FFN 2048、FP32**，不是两层版。主模型为FP16。
- Predictor有5个输出；前5个OPT层使用对应输出，深层复用第5个输出。Predictor的token embedding与主模型共享，额外embedding显存为0。
- Neuro-Fetcher保留原生ceil/threshold选择、并列分数全部保留、recent-20 union。此长输入吞吐实验实测历史选择密度为 **30.0408%**；不能把较短分类输入中约45.94%的密度写到这里。最近20个token可能已在预测集合中，不会必然额外增加20项。
- InfiniGen遵循本次接受的原生适配：alpha=7、partial weight ratio=0.3、前两层Full、effective max_num_kv=2049（历史nominal=615）。不能据此宣称InfiniGen的实际保留率严格等于30%，或实际缓冲只有615项。
- FlexGen及InfiniGen的权重100%GPU、KV home 100%CPU，通用循环的 `overlap=false`。**InfiniGen仍执行自身的逐层prefetch stream**；不能解释成禁用了InfiniGen全部预取。
- 原生基线输入来自PG19，Neuro-Fetcher输入来自固定英文模板；token IDs不相同。原始JSON及manifest保留prompt hash。
- 最新配对中每个后端先完整warmup一次，再执行六对AB/BA交错运行；FlexGen为五次独立进程，各先做一次单输出token的原生warmup；InfiniGen为一次正式测量；Full GPU为一次完整warmup加五次正式测量。

## 3. 按层优化：已经实现，也完成了受控复测

旧V12让GPU上两块缓冲都容纳全部40层的选中KV；V16改成**两块各容纳一层KV的GPU缓冲**，按层交替使用。

CPU侧仍保留完整KV home和两套覆盖全部层的pinned staging，用于跨解码步预测、CPU gather/packing。当前步开始时启动该步KV的H2D，后续层搬运与当前层计算流水执行；CUDA events确保缓冲读取、复用和搬运之间的依赖。不是提前把全部40层KV都搬上GPU，也不是删除CPU staging。

| 同一进程、同模型、同prompt的六对复测 | V12：全层GPU缓冲 | V16：按层GPU缓冲 |
|---|---:|---:|
| Decode tok/s，均值±SD | 14.827 ± 0.147 | 15.273 ± 0.077 |
| allocated峰值均值 GiB | 29.847 | 26.068 |
| reserved峰值均值 GiB | 30.844 | 27.141 |
| 两块GPU staging容量合计 GiB | 3.876 | 0.097 |
| 历史选择密度 | 30.0408% | 30.0408% |
| 完整生成过程H2D KV bytes | 255,652,659,200 | 255,652,659,200 |

配对结果：V16在6/6对中更快；吞吐均值比 **1.0301（+3.01%）**，allocated均值下降 **3.779 GiB（12.66%）**。GPU staging容量下降 **97.5%**。

全部正式运行的完整输出、prefill logits、decode logits、127步逐步索引哈希及H2D字节数一致。本轮未触发缓冲扩容。另有小模型的并列分数扩容测试、原生cached OPT对照和OPT-13B强制token一致性检查。

该优化没有改模型权重、预测器精度、checkpoint、选中索引、recent-20或注意力计算。若threshold ties使选择数超过初始容量，会扩容以保留原算法的全部索引；0.097 GiB是此工作负载的缓冲容量，不能保证任意输入都固定不变。

**受控复测支持“显著减少缓冲显存，该配置下吞吐未损失且约提高3%”。**先前独立测试14.43→17.73 tok/s的约23%差距跨越不同时段，不能全归因于按层优化。3%也属于两种完整缓冲/调度实现的比较，尚未拆分容量、同步及资源竞争各自的因果贡献。

## 4. 显存用在哪里，为什么从38 GiB降到26 GiB

### 初版峰值的主要来源

原异步实现的allocated峰值约38.833 GiB。GPU上两个槽均按2048-token完整历史容量、全部40层预分配，每槽6.25 GiB，合计 **12.5 GiB**；实际只传输约30%，也没有减少这部分预留。另有约0.479 GiB重复embedding。

第一步紧凑化（V12）将单槽容量缩为名义预算加recent余量，本配置635 tokens，并在需要时扩容；两块全层GPU staging共3.876 GiB。验证embedding相同后共享，取消重复GPU副本。该版独立实测allocated降至29.728 GiB。

第二步按层化（V16）将两块全层GPU槽替换为两块单层槽，初始容量仍为635，GPU staging进一步降至 **104,038,400 bytes = 0.096893 GiB**。

### 当前主要固定存储

| 存储 | 大小 | 所在位置及解释 |
|---|---:|---|
| OPT-13B参数及buffers | 23.941 GiB | GPU，实际张量求和 |
| FP32 predictor参数及buffers | 0.564 GiB | GPU，包含位置编码buffer；不是仅参数大小 |
| Predictor embedding额外副本 | 0 GiB | 与主模型共享；被共享的embedding本身约0.479 GiB，已包含在主模型中 |
| 两块单层GPU KV staging | 0.097 GiB | GPU，当前容量635 tokens |
| 完整CPU KV home | 6.250 GiB | CPU pinned memory，按2048容量分配 |
| 两套全层CPU staging合计 | 3.876 GiB | CPU pinned memory，用于packing及异步步骤缓冲 |

前三项GPU固定张量加staging约24.603 GiB；峰值还包含激活、选择张量、临时工作区及并发中存活的张量。不能把剩余差额全部标成“预测器开销”，也不能把InfiniGen的25.518 GiB直接加0.564 GiB当作我们的理论峰值：两个后端的缓冲及工作区不同。

这里展示了CPU固定KV/packing缓冲容量，合计约10.126 GiB，**不是进程总RSS**。独立V16五次运行的进程peak RSS均值约14.34 GiB；RSS包含进程生命周期中的加载等开销，配对运行又共用同一进程，不能把其高水位当成纯decode内存。当前结论聚焦GPU显存，不能写成“总系统内存只增加2%”。

### 三种GPU内存统计必须分别保留

1. `peak_allocated`：PyTorch张量分配峰值；本文主表使用这一项。
2. `peak_reserved`：PyTorch管理的分配器内存；在cudaMallocAsync与默认分配器之间，统计含义可能有差异。不能以reserved替换allocated后仍沿用原差值。
3. NVML / `nvidia-smi`：整卡已用显存，包含CUDA context及非PyTorch分配；独立V16从模型加载到全部运行按0.1秒采样，整卡采样峰值 **28.604 GiB**。该数字不是最新配对测试的同次结果，也不是精确瞬时峰值。

不能将V16的NVML 28.604 GiB与InfiniGen的allocated 25.518 GiB相减。主表的“约2.2%额外显存”限于allocated定义，不能转述为整卡占用只多2.2%。

## 5. InfiniGen本来也是按层，论文如何区分

本次实际基线使用 `speedup/flexgen/infinigen/flex_opt_v7.py` 和 `pytorch_backend_v7.py`。当前层的部分投影推测下一层索引，然后通过独立stream预取下一注意力层KV。CPU-KV配置下，其预取staging在cache home设备（CPU）分配，GPU侧按层加载。

因此，**不能把“按层预取”本身写成相对InfiniGen的新贡献**，也不能称InfiniGen将全部40层KV GPU staging长期驻留。Neuro-Fetcher的区别是跨解码步的学习式预测与CPU packing，以及本实现的两块可复用单层GPU缓冲；对比应准确描述预测来源、时间位置和存储布局。

## 6. 历史独立结果：保留，但不与最新结果拼接

| 版本/运行组 | Decode tok/s | allocated GiB | GPU staging GiB | 定位 |
|---|---:|---:|---:|---|
| 初版全历史容量异步，2026-10-26 | 16.541 ± 3.977 | 38.833 | 12.500 | 说明初版显存瓶颈 |
| V12紧凑全层缓冲独立测试 | 14.427 ± 2.326 | 29.728 | 3.876 | 优化过程的历史记录 |
| V16按层缓冲独立测试，2026-10-31 | 17.728 ± 0.162 | 25.949 | 0.097 | 另一次完整测试；与其自身显存一起引用 |
| V12六对复测，2026-11-02 | 14.827 ± 0.147 | 29.847 | 3.876 | 最新消融的对照 |
| V16六对复测，2026-11-02 | 15.273 ± 0.077 | 26.068 | 0.097 | 本文主结果 |

最新配对共享模型并反复创建engine，allocated峰值与独立运行有小量差别；保留各自同组数据即可。**不能挑25.949 GiB搭配15.273 tok/s冒称同组结果，也不以17.728 tok/s代替受控复测中的15.273 tok/s来计算按层消融收益。**

本文件更新了此前“按层仍未实现”“相较FlexGen多用8.9%显存”的阶段性结论。旧文档原样保存，其数据依然属于相应旧版本；当前论文若采用按层版，应以本文件和2026-11-02配对记录为准。

## 7. 预测、传输成本与异步证据的边界

已有2026-10-26固定prefix、隔离组件测试可作成本背景：

| Context | Predictor GPU ms | CPU packing ms | H2D stream ms |
|---:|---:|---:|---:|
| 512 | 15.87 | 27.31 | 22.47 |
| 1024 | 34.62 | 104.48 | 42.61 |
| 1536 | 70.02 | 133.68 | 62.54 |
| 1920 | 82.07 | 242.56 | 77.54 |

这是旧后端的隔离成本，不是V16在并发执行中的新增延迟，不能简单相加得到每步耗时。本次V17计时没有profiler，原始每步记录仍包含predictor、计算、等待及DMA事件区间；并发下事件区间包含调度等待和竞争。**V16的 `dma_ms`跨越逐层传输及层计算间隙，尤其不能标成纯PCIe DMA服务时间。**

历史serial/async矩阵及组件记录仍保留。按层配对只对比V12/V16的异步实现，不构成新V16的serial/async消融。此前CUDA profiler因 `CUPTI_ERROR_NOT_INITIALIZED` 缺少设备活动，不应将CPU trace作为GPU完全重叠的直接证据。

论文可以报告测得收益、成本与依赖关系，但不要写 `zero overhead`、`all predictor costs are hidden` 或用单个不等式推导实际完全重叠。

## 8. 论文与审稿回复具体怎么改

| 位置/问题 | 使用本次新增内容 |
|---|---|
| Section 3.2 / Figure 6 | 分清CPU的按步pinned staging与GPU的按层双缓冲；更新旧整步GPU Ping/Pong示意，注明两块GPU缓冲各仅一层；用GPU device memory，不写on-chip SRAM |
| 实验设置 | 明确FP16 OPT-13B、FP32单层predictor、batch4、1920/128、127步decode、recent-20及实际密度；逐方法列runtime、warmup、重复与计时边界 |
| 显存—吞吐主结果 | 使用第2节同组数据，allocated/reserved分列；对InfiniGen多约2.2% allocated、对本次FlexGen更低；保留Full GPU参照 |
| 实现/消融 | 使用第3节V12/V16六对复测，主要论点为缓冲容量−97.5%、allocated−3.779 GiB、吞吐+3.01% |
| R1第3条 | 回答Ping/Pong物理内存到底放在哪里、占多少；说明predictor/packing/transfer被计时及并发事件区间的含义，避免声称完全隐藏 |
| R2 Major 4 | 补显存结果、硬件与backend条件、prefill/decode/total、测量协议，承认跨runtime比较的范围 |
| 质量与结论 | 引用既有准确率/PPL结果；输出哈希只证明当前调度改动在已测输入上保持计算结果，不等于完成所有质量数据集重评 |

评审原话（reviews.pdf第1页）：

> Please also report predictor and transfer costs during co-execution and clarify the physical memory used by the Ping/Pong buffers, since the stated overlap condition alone does not establish zero overhead.

以下英文可作为改稿底稿，按最终章节与表号替换：

**Implementation.**

> We use layer-streamed GPU KV staging with two reusable device buffers, each sized for one layer of selected KV entries. Prediction and CPU packing for the following decoding step proceed asynchronously. The current step transfers KV entries incrementally as its layers execute, with CUDA events enforcing transfer and consumer dependencies. The full KV home and step-level packing buffers remain in pinned CPU memory. This implementation retains the FP32 predictor, the original selection rule, and the recent-token union.

**Memory–throughput results.**

> For OPT-13B with batch size 4, 1,920 input tokens, and 128 generated tokens, the layer-streamed implementation achieved 15.27 ± 0.08 decode tokens/s with a mean peak PyTorch allocation of 26.07 GiB. Relative to the measured InfiniGen configuration (9.16 tokens/s, 25.52 GiB), it used 2.2% more allocated GPU memory and achieved a 1.67× decode throughput ratio. Relative to the measured native FlexGen CPU-KV configuration (5.59 ± 0.20 tokens/s, 27.31 GiB), it used 4.5% less allocated GPU memory and achieved a 2.73× throughput ratio. These comparisons use the same workload dimensions but different runtimes, prompt texts, warmup procedures, and measurement dates; they characterize the evaluated implementations rather than an isolated algorithmic speedup. Full GPU residency remained substantially faster when its cache fit in device memory.

**Staging ablation.**

> In a balanced six-pair comparison using a common model and prompt on one GPU, replacing two all-layer GPU staging buffers with two one-layer buffers reduced staging capacity from 3.876 to 0.097 GiB and mean peak allocated memory by 3.779 GiB. Mean decode throughput changed from 14.83 to 15.27 tokens/s. Outputs, prefill and decode logits, selected-index hashes, and transferred KV bytes were identical across the compared runs.

审稿回复可据此说明“新增了哪些表和方法解释”，在实际完成LaTeX修改后再写 `We have added ...`，不要把本汇总文档当成稿件已经修改。

## 9. 适用范围与后续交接

本次按层最终版本完整测量覆盖输入1920、batch4、生成128。此前512/1024/1536的结果属于旧后端；不能将整条旧曲线换成V16标签。已有InfiniGen/H2O/Quest横向结果继续保留；没有新做V16与全部方法在所有长度的统一套测，不把旧表重命名成新版本结果。

不需要将已接受基线重新审计或额外实验作为开始改论文的条件。后续收到LaTeX包后，以新版本目录修改稿件，原件保留。LLaMA/跨架构本轮仍不开展。

建议新对话先读本文，再读 [按层配对说明](/home/liushenyu/infinigen/LAYERWISE_PAPER_NOTE_20261102.md)，并结合原交接中的质量结果、审稿映射和OPT-6.7B迁移说明。本文补充并更新资源/吞吐口径，不覆盖其它章节的既有结论。

## 10. 原始证据与可复现汇总

| 资料 | 路径 |
|---|---|
| 最新配对汇总、配置和逐次结果 | [V17 summary.json](/home/liushenyu/infinigen/speedup/results/layerwise_paired_v17_20261102/summary.json)；同目录含manifest、12次正式运行、2次warmup |
| 按层实现 | [reviewer_offload_backend_v16.py](/home/liushenyu/infinigen/speedup/reviewer_offload_backend_v16.py) |
| 配对runner | [benchmark_layerwise_paired_v17.py](/home/liushenyu/infinigen/speedup/benchmark_layerwise_paired_v17.py) |
| InfiniGen显存与同次吞吐 | [infinigen_p1920_g128_b4.json](/home/liushenyu/infinigen/speedup/results/reviewer_memory_v9_20261029/infinigen/infinigen_p1920_g128_b4.json) |
| FlexGen五次重复 | [FlexGen summary.json](/home/liushenyu/infinigen/speedup/results/flexgen_repeats_v15_20261031/summary.json)；同目录五份原始JSON |
| Full GPU | [p1920-g128-b4-full_gpu.json](/home/liushenyu/infinigen/speedup/results/reviewer_throughput_20261026/p1920-g128-b4-full_gpu.json) |
| V16独立五次运行及NVML | [p1920-g128-b4-neural_async.json](/home/liushenyu/infinigen/speedup/results/layerwise_v16_20261031/p1920-g128-b4-neural_async.json) |
| V12紧凑化与38GiB来源 | [compact REPORT.md](/home/liushenyu/infinigen/speedup/results/compact_staging_v11_20261029/report_v13/REPORT.md) |
| 隔离组件成本 | [components.json](/home/liushenyu/infinigen/speedup/results/reviewer_components_20261026/components.json) |
| 本轮机器可读汇总 | [facts.json](/home/liushenyu/infinigen/speedup/results/memory_throughput_summary_20261103/facts.json)、[all_runs.csv](/home/liushenyu/infinigen/speedup/results/memory_throughput_summary_20261103/all_runs.csv)、[comparisons.csv](/home/liushenyu/infinigen/speedup/results/memory_throughput_summary_20261103/comparisons.csv) |
| 本轮原始来源SHA256 | [provenance.json](/home/liushenyu/infinigen/speedup/results/memory_throughput_summary_20261103/provenance.json) |

汇总脚本 [summarize_memory_throughput_v18.py](/home/liushenyu/infinigen/speedup/summarize_memory_throughput_v18.py) 只读已有JSON，不导入torch、不初始化CUDA、不运行模型；只允许写入一个尚不存在的新输出目录。复算示例：

```bash
cd /home/liushenyu/infinigen
conda run -n infinigen python speedup/summarize_memory_throughput_v18.py \
  --output-dir speedup/results/memory_throughput_summary_20261103_recheck
```

该命令不会覆盖本轮汇总目录或原始结果。所有比值与差值来自原始JSON，源文件hash写入provenance便于核对。
