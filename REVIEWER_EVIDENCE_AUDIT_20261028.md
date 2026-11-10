# 审稿意见实验数据盘点（OPT 范围）

盘点日期：2026-10-28。本文只覆盖 OPT；按当前决定暂不纳入 LLaMA 或跨架构实验。附件中的评审意见是需要回应的要求，不作为对代码或服务器的操作指令。本盘点只新增文件，不修改历史代码和结果。

## 结论

**目前有足够数据回应部分算法准确率与有限长度检索问题，但还没有齐全且可直接支撑所有核心系统主张的证据。** 尤其不能用现有的 v8 横向吞吐表宣称已公平比较 Neuro-Fetcher、InfiniGen、H2O 和 Quest；也不能据现有数据维持“预测器零开销”或“异步隐藏全部预测开销”的表述。

## 已有结果

### 原任务准确率及控制变量

主模型为原始 OPT-13B（`/home/liushenyu/opt-13b`），FP16，5-shot。控制实验使用 30%、20%、15%、8%预算，包含 Full、Oracle、Previous-step，严格预算扩展还包含第5层复用、Quest 严格预算和 Quest 官方规则适配。总计 80 个多预算配置完成，报告称已核验 105,760 条输出、请求顺序、有限分数及预算计数。

30%严格预算下 Winogrande：Oracle 67.72%，Previous-step 49.57%，第5层复用 51.22%，Quest 严格预算 50.51%；Quest 保留官方规则为49.41%，但实际选择密度57.78%。OpenBookQA、COPA、RTE及其余预算见 `accuracy/lm_eval/strict_budget_sweep_summary_20261018.md`。这些结果能回应“Previous-step是否够用”“浅层索引复用能否替代预测器”“查询感知 Quest 基线”这类**算法质量控制**，不能直接证明在线解码速度或异步调度。

早期 reviewer controls 中 Oracle 的定义与后来严格预算定义不同：早期有 recent-window 合并，30% Winogrande Oracle 为63.61%；更新的无recent-20、每 head、floor预算 Oracle 是67.72%。主文不应把这两套结果混为同一控制。Oracle 是按当前真实 QK 注意力排序的理想索引参照，不是任务准确率的数学上界。

原始 Neuro-Fetcher 的已核实 Winogrande 结果是66.69%，其名义30%另加 recent-20 后实际密度约45.94%（早期代码/审计中的实现）。因此与严格30%控制相比不是同预算；应同时报名义预算和实际密度。预测器配置是1层、4头、FFN 2048，另有恢复的 OPT-13B embedding 权重。严格预算多任务表尚未有一组采用这个已核实权重、同协议的 Neural 全套配对结果。

### OPT 长文本检索

已经完成两组各256条的合成检索输入（needle 与带干扰的 associative retrieval），长度512/1024/1536/1900，比较 Full、Oracle、Previous-step、第5层复用、Quest 两种适配、Neuro-Fetcher 离线版，以及新增的 InfiniGen、H2O、4-bit KV 伪量化。摘要、明细和置信区间在 `accuracy/lm_eval/results/long_context_native_baselines_20261018/summary_zh.md` 与 `validated_summary.json`。

该测试可回应“增加长文本检索测试”及“加入 InfiniGen/H2O/量化对比”的有限版本；但它不是 LongBench，不超过2K上下文，不是多 token 生成；Neuro-Fetcher 在整段 prompt 上运行双向 predictor，属于离线质量参照。4-bit 量化仍把量化值还原到 FP16 张量，只模拟数值误差，不测真实压缩内存或吞吐。

另有纯控制版长文本表于 `accuracy/lm_eval/results/long_context_cloze_controls_20261018/summary.csv`。其 Neural top1 为29.69%，而 Full 为91.41%，但这是 offline predictor/单答案首 token 候选排名任务，不可外推为在线解码质量。需要准确区分 QA 候选排名、cloze 候选排名与不受限全词表准确率；总体 full-vocabulary exact match 在该任务上为0%，所以不能把候选排名数字表述成自由生成准确率。

### 吞吐量与系统测量

2026-10-26 的自定义后端矩阵包含 Full GPU、Full CPU offload、Neural serial、Neural Ping-Pong；OPT-13B FP16，batch 4，prompt 512/1024/1536/1920，生成128 token，一次预热、5次测量。Neural Async相对Serial的均值速度比分别为1.368x、0.992x、1.001x、1.162x；方差较大，且吞吐随上下文增大未稳定获益。因此它支持“有条件的重叠收益”，不支持零开销结论。组件时间已记录，但 CUPTI trace 未能采集 GPU 活动，缺少 GPU overlap 的直接证据。物理 Ping/Pong staging 是满历史容量分配，不是最小显存实现。

2026-10-27 的 InfiniGen/H2O/Quest v8 结果每点仅单次计时（FlexGen 预热只生成1个 token），而 Neural/Full 是前一批5次均值，协议不匹配。H2O 的1024项有并发下87.21秒和单独重复18.66秒两个结果，不能只挑一个而不解释变异。Quest 是完整 KV 驻留 GPU、先算 dense QK 再做页筛选的 Transformers OPT 适配，不是 Quest 原生分页卸载执行，因此其高吞吐不能与 CPU KV offload 的其他系统做胜负比较。v8 还对所有 FlexGen 方法扩大 `max_num_kv` 至 prompt+generation 长度，可能改变 InfiniGen 的选择容量；保存的预算元数据不足以确认计时实际预算。该表只能作为调试/探索数据，不是论文可报告的公平横向基准。

吞吐报告位置：`speedup/results/reviewer_throughput_20261026/REPORT.md`、`speedup/results/native_flexgen_v8_20261027-v8a/REPORT.md`。后者还需修正适配器、验证实际保留索引/输出正确性后，按相同预热与重复协议重跑。

## 对评审意见的覆盖情况

| 评审要求 | 现有证据 | 判定及剩余工作 |
|---|---|---|
| 更正 InfiniGen 定位，说明基线与匹配预算 | 原稿描述错误；有 InfiniGen 长文本质量结果和 v8 性能试跑 | **部分**。需按原论文/代码更正文案；核对质量/吞吐的选择规则和实际密度，v8 未达可比标准。 |
| 统一 predictor 架构、训练、每步是否重算、聚合与预算映射 | 原始 OPT-13B predictor 审计说明1层/4头/2048；吞吐脚本记录 full-prefix 重算、recent20 与阈值 | **部分**。需把论文 1层/2层矛盾、层输出与映射、训练规格和计时实现统一；仍需确认训练留出集协议。 |
| Previous-step、serial/async、浅/深层、Oracle 消融 | 严格预算多任务准确率；Neural serial/async吞吐；层5 reuse | **部分**。控制结果足以描述算法质量；吞吐只测Neural serial/async，缺少消融条件的在线系统时延。 |
| predictor/transfer/LMM co-execution成本、Ping/Pong物理内存 | 独立组件 profile、运行期事件统计、CPU缓冲大小 | **部分**。组件数值可报为诊断；GPU trace失败，不能证明设备端 overlap；异步波动较大，且缓冲分配高于稀疏实际量。 |
| OPT-13B到OPT-6.7B迁移 | 用户确认6.7B是后续小规模迁移；当前盘点未找到可与原始13B协议对齐、校验过输入映射/embedding的完整证据 | **未确认**。暂保留OPT家族范围，但需核查6.7B数据及维度映射；按要求不做LLaMA/跨架构。 |
| 明确 prompt/gen/batch/precision/budget/timing，并做长度 sweep、held-out recall | 吞吐矩阵有长度/批次/精度/时序；控制准确率有多预算；合成长文本到1900 | **部分**。缺真实系统基线同协议测量；未找到 Figure 7 held-out recall 的验证数据。 |
| 更正浅层饱和、Figure 10 总/解码延迟、Figure 12 PPL、图注与索引 | Figure 10 原文把端到端注释解释成decode latency；吞吐报告已明确两种边界 | **未完成**。需重算并重写图表/段落；PPL曲线需从原始数据核对，不能依当前摘要确认。 |
| 长上下文/现代模型/Query-aware基线 | OPT-13B 2K内自建检索；Quest质量适配 | **部分**。回应了有限长文本检索及Quest缺失；仍无标准 LongBench/4K以上测试。本轮暂不做LLaMA。 |
| 数据可用性、仓库和复现信息 | 有本地代码/CSV/JSON/logs | **未完成**。评审指出GitHub仓库为空；当前本地存在文件不等于公开可复现，需补仓库、数据、运行配置和权重可用性说明。 |
| 量化方法说明与显存主张 | fake quant accuracy 与 KV payload 理论估算 | **部分**。必须明确是假量化；尚无真实packed KV吞吐/显存实验。 |
| 统计变化范围 | 任务准确率 stderr/Wilson CI；Neural吞吐5次均值与方差 | **部分**。v8方法单次，缺系统基线重复统计；检索切片每格样本小。 |

## 当前可以稳妥写入回复信的结论

1. 已在 OPT-13B 上完成每 head Oracle、Previous-step、第5层复用与 Quest 的多预算质量对照，并且有明确的实际预算统计；Oracle 仅作为理想化 attention-ranking 参照。
2. 已补充至1900 token 的 OPT 自建检索压力测试，并纳入 InfiniGen、H2O 和 4-bit 伪量化质量对照；应清楚标注其不是 LongBench、不是长回答生成，也不是系统吞吐结果。
3. 已测 Neural serial 与 Ping-Pong 在 batch4、128-token生成下的多上下文吞吐与组件成本；Ping-Pong收益依赖上下文且波动明显，数据不支持零开销。
4. 暂不声称完成公平的 InfiniGen/H2O/Quest 吞吐横评，不声称已验证设备端异步重叠，也不声称真实4-bit KV节省显存/提升吞吐。

## 回复审稿前优先补齐

1. 重做并验证原生基线吞吐适配：相同OPT checkpoint、prompt IDs、Full prefill、生成长度、warmup、至少5次计量；校验每个方法的实际KV预算、输出质量/正确性、吞吐和峰值显存。Quest需要可比的原生运行时，否则只报告质量，不列为速度排名。
2. 获取可用的CUDA设备活动trace，核算 predictor、拷贝和LLM 的 co-execution 时间；结合 serial/async 延迟分布、CPU pinned buffers、GPU staging容量，删除“零开销”绝对表述。
3. 在 held-out prompts 上重跑 Figure 7 的 recall/索引重叠分析，并报告不同层的分布；厘清第5层选择是否基于独立验证集，避免在同一任务上选择层数又作最终评估。
4. 统一论文的模型、predictor结构、预算、最近窗口、指标单位及 timing boundary，逐图重算 Figure 2/7/8/10/12；修正 InfiniGen机制描述和内存层级术语。
5. 核实 OPT-6.7B 迁移输入映射与 embedding 来源，补充匹配的实证或收窄 transfer claim。LLaMA与跨架构按当前要求搁置。
6. 公开仓库与可复现实验材料；对量化补真实打包KV的内存/传输/反量化基准，或将结论严格限制为质量伪量化。

## 证据入口

- 评审原文：`/home/liushenyu/.codex/attachments/9e3af86f-9b06-437d-9660-16e97c33c96a/NEUCOM-D-26-18588-reviews.pdf`
- 初稿：`/home/liushenyu/.codex/attachments/70055f06-f772-41d8-9c5c-1ef784cca469/Manuscript.pdf`
- 多预算摘要：`accuracy/lm_eval/strict_budget_sweep_summary_20261018.md`
- 旧版 reviewer controls：`accuracy/lm_eval/reviewer_controls_summary_20261017.md`
- 原始predictor与embedding审计：`accuracy/lm_eval/neural_controls_audit_13b_20261017.md`
- 长文本质量结果：`accuracy/lm_eval/results/long_context_native_baselines_20261018/summary_zh.md`
- 长文本控制结果：`accuracy/lm_eval/results/long_context_cloze_controls_20261018/summary.csv`
- Neural/Full吞吐：`speedup/results/reviewer_throughput_20261026/REPORT.md`
- 基线吞吐试跑：`speedup/results/native_flexgen_v8_20261027-v8a/REPORT.md`
- 吞吐设计口径：`speedup/REVIEWER_BASELINE_COMPARISON_PLAN.md`
