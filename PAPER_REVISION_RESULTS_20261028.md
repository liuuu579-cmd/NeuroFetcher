# 论文修订结果与指标口径

更新日期：2026-10-28。所有结果为当前工作区已有结果；用户确认可接受，改稿阶段不要求重跑。数字应与原 CSV/JSON 保持一致。

## 1. 主 reviewer controls：OPT-13B，多预算

模型 `/home/liushenyu/opt-13b`，5-shot，FP16，batch 1。严格预算方法以每个 head 单独选择索引，预算为 `max(1, floor(ratio × 可见前缀长度))`，不额外加入 recent token。共80个配置，105,760条输出；现有报告称已核验输出顺序、有限分数和预算统计。

每格格式：`Winogrande / OpenBookQA acc (acc_norm) / COPA / RTE`，均为百分数。

| 方法 | 30% | 20% | 15% | 8% |
|---|---|---|---|---|
| Oracle per-head | 67.72 / 42.40 (43.40) / 81 / 50.90 | 67.32 / 42.40 (44.80) / 81 / 49.82 | 67.17 / 42.20 (46.80) / 79 / 50.90 | 64.72 / 39.20 (43.60) / 75 / 55.96 |
| Previous-step per-head | 49.57 / 26.80 (35.00) / 57 / 49.46 | 50.51 / 27.00 (36.60) / 58 / 52.71 | 50.36 / 27.40 (35.40) / 53 / 51.26 | 48.93 / 25.80 (36.40) / 57 / 48.38 |
| Layer-5 reuse per-head | 51.22 / 26.40 (34.20) / 58 / 54.51 | 51.70 / 26.60 (35.40) / 52 / 52.35 | 51.78 / 25.40 (34.60) / 52 / 49.46 | 50.20 / 25.60 (33.40) / 52 / 54.51 |
| Quest strict-budget OPT adaptation | 50.51 / 26.40 (38.20) / 48 / 56.32 | 51.30 / 24.20 (36.20) / 48 / 53.79 | 50.99 / 23.60 (34.40) / 49 / 53.43 | 51.93 / 22.00 (37.00) / 53 / 53.43 |
| Quest official-rule OPT adaptation | 49.41 / 32.60 (41.00) / 63 / 56.32 | 49.41 / 32.60 (41.00) / 63 / 55.60 | 49.41 / 32.60 (41.00) / 63 / 56.32 | 49.41 / 32.60 (41.00) / 63 / 55.60 |

严格预算各方法实测总选择密度：30%配置分别是29.40% (Wino), 29.21% (OBQA), 29.13% (COPA), 29.82% (RTE)；20%为19.48%,19.33%,19.26%,19.84%；15%为14.39%,14.23%,14.15%,14.81%；8%为7.45%,7.33%,7.28%,7.82%。Quest官方规则保留至少3页、前两层Full，不是严格30%；30%名义下实际密度为57.78%,69.42%,75.09%,33.31%。

语义：Oracle 使用当前 query 的真实 QK attention 排序、逐 head top-k，是理想选择参照而非任务准确率上界。Previous-step 使用同层同 head 前一 query 的完整真实排序，在当前预算下重新选取。Layer-5 reuse 的前5层各用 Oracle，第6至40层按 head 编号复用第5层当前 query 的索引。Quest 的两种OPT实现都应标作 adaptation；官方规则与严格限额分别列出。

## 2. Neuro-Fetcher历史对照与旧 Oracle 口径

原始 Neuro-Fetcher 的 1-layer/4-head/FFN-2048 配置，在 OPT-13B、Winogrande、5-shot、名义30% + recent-20 下 accuracy 为66.6929755%；实测 retained entries 为45.94%。Full为68.2715075%。该Neural使用shared selection；新 Oracle 表为per-head且无recent-20。因此不要把两者标成完全同预算/同粒度。

恢复 OPT-13B predictor embedding 后 Neural 重现66.69%；未恢复 embedding 的诊断版本为51.46%，不得作为正式结果。checkpoint `/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth`，embedding `/home/liushenyu/test/opt_13b_embed_weights.pth`。

旧 controls 的 Oracle 在30%名义预算 + recent-window定义下为63.61% Winogrande；严格预算新 Oracle 为67.72%。这是口径变化，不是结果相互矛盾。

## 3. 长文本检索与质量基线

OPT-13B，FP16，batch 1；两项自建任务各128条，共256条；长度512/1024/1536/1900，目标深度25/50/75/90%，每格8条平衡答案。此为2K内压力测试，不是标准LongBench。每个策略评估 QA候选排名、cloze候选排名、cloze全词表首 token，并保留按 task/length 的结果与 Wilson 区间。

| 方法 | QA候选排名 | Cloze候选排名 | Cloze全词表首 token | Cloze实际选择密度 |
|---|---:|---:|---:|---:|
| Full | 73.83% | 91.41% | 32.81% | 100.00% |
| Oracle per-head | 78.52% | 94.14% | 39.45% | 29.94% |
| Previous-step per-head | 35.94% | 44.53% | 21.48% | 29.94% |
| Layer-5 reuse per-head | 20.70% | 32.03% | 1.95% | 29.94% |
| Quest strict adaptation | 28.91% | 37.89% | 5.86% | 29.94% |
| Quest official-rule adaptation | 72.66% | 85.55% | 48.83% | 32.64% |
| Neuro-Fetcher historical offline | 23.83% | 29.69% | 10.55% | 30.88% |
| InfiniGen native accuracy path | 77.73% | 94.14% | 41.41% | 31.68% |
| H2O native accuracy path | 32.42% | 35.55% | 8.98% | 52.48% |
| 4-bit KV fake quantization | 72.66% | 91.41% | 32.42% | 100.00% |

说明：Neural离线 predictor 可读整段 prompt，不是 causal online预测；fake quant量化/反量化后仍保留FP16张量，不证明真实内存和速度收益。H2O保留原生15% heavy + 15% recent规则，未额外加Neuro-Fetcher recent-20。InfiniGen用变换 OPT-13B checkpoint、partial weights 0.3、alpha7/capacity1/budget0.3，前两层Full。H2O与InfiniGen按整段容量运行，实际密度因浅层Full等与名义预算不同。

## 4. Neural/Full throughput与调度

OPT-13B FP16，batch4，prompt 512/1024/1536/1920，固定生成128 tokens；一次完整warmup，5次测量。decode token/s是均值 ± 样本标准差；total是含prefill端到端平均秒数。

| Prompt | Full GPU tok/s | Full CPU offload tok/s | Neural serial tok/s | Neural async tok/s | Async/serial | End-to-end秒数 Full/Offload/Serial/Async |
|---:|---:|---:|---:|---:|---:|---|
| 512 | 107.84±0.06 | 22.14±0.01 | 34.95±0.05 | 47.82±9.61 | 1.368x | 5.10 / 23.37 / 14.97 / 11.62 |
| 1024 | 93.97±0.04 | 12.90±0.00 | 23.13±0.02 | 22.93±10.52 | 0.992x | 6.13 / 41.69 / 24.25 / 29.59 |
| 1536 | 81.68±0.03 | 9.23±0.00 | 16.95±0.08 | 16.96±5.29 | 1.001x | 7.39 / 58.55 / 33.45 / 37.75 |
| 1920 | 74.57±0.03 | 7.57±0.00 | 14.24±0.02 | 16.54±3.98 | 1.162x | 8.38 / 71.58 / 40.16 / 38.86 |

isolated component profile：predictor 15.9/34.6/70.0/82.1 ms，H2D 22.5/42.6/62.5/77.5 ms，CPU packing约27至243 ms，对应上述四个上下文。共执行时事件含竞争与等待，不是纯kernel service time。CUDA profiler因`CUPTI_ERROR_NOT_INITIALIZED`未采到GPU活动；不要声称已经用trace直接证明GPU overlap。当前数据显示收益受上下文影响，不能支持无条件“零开销”。Ping/Pong物理GPU staging按完整历史容量分配，异步两槽；不是最小显存实现。

## 5. 已接受的原生基线吞吐横向结果

用户确认当前基线和运行结果可以接受，论文修改直接使用，不以重跑为条件。OPT-13B FP16，batch4，128生成，prompt512/1024/1536/1920。每格为 `decode tok/s / total seconds`：

| Prompt | Full GPU | Neural serial | Neural async | InfiniGen | H2O | Quest native rule | Quest capped |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 107.84 / 5.10 | 34.95 / 14.97 | 47.82 / 11.62 | 23.57 / 22.08 | 37.00 / 14.12 | 78.79 / 6.83 | 75.31 / 7.12 |
| 1024 | 93.97 / 6.13 | 23.13 / 24.25 | 22.93 / 29.59 | 14.10 / 37.00 | 28.39 / 18.66 | 68.76 / 8.16 | 65.86 / 8.49 |
| 1536 | 81.68 / 7.39 | 16.95 / 33.45 | 16.96 / 37.75 | 9.85 / 53.04 | 21.04 / 25.49 | 57.39 / 10.16 | 55.13 / 10.54 |
| 1920 | 74.57 / 8.38 | 14.24 / 40.16 | 16.54 / 38.86 | 7.23 / 72.12 | 13.75 / 38.75 | 50.68 / 11.79 | 48.97 / 12.17 |

配套说明必须清楚：FlexGen InfiniGen/H2O与Neural自定义 offload backend、Quest Transformers OPT cached-decode适配实现运行路径不同。Quest运行使用完整KV驻留GPU，不测CPU-to-GPU页卸载收益；将结果描述为按相同工作负载得到的native-rule/adaptation吞吐结果，不将它包装成Quest官方分页卸载runtime。InfiniGen使用alpha7、partial ratio0.3；H2O为heavy0.15/recent0.15并保留自身recent规则；不要给其它算法手动添加recent-20。Quest原生页规则为16-token page、至少3页、前两层Full；capped variant按ceil(0.30×visible length) token预算、无recent union。H2O 1024第一条并发记录87.21秒，隔离重复为18.66秒；表沿用既有报告选取的隔离重复，同时在正文/补充材料保留其波动记录。

## 6. 模型与Predictor版本口径

- 用户确认有一层和两层 Transformer predictor 两个真实版本；论文要明确哪张图/哪组实验采用哪一版本。
- 当前代码中与原始Neuro-Fetcher checkpoint相匹配、主 reviewer controls及新 throughput实现为一层 encoder、4 heads、FFN 2048、OPT-13B 5120维输入、输出5层分数。checkpoint为`predictor_topk_0.3_iwl_epoch_20.pth`。
- 初稿3.1.1写1层；4.1 implementation写2层。不要简单机械地把全文全改成一层或全改成两层；利用LaTeX包内图表/实验配置及对应checkpoint把版本映射完成后统一。
- 源文件 `accuracy/src/modeling_opt_neu.py` active branch为1层、4头、FFN2048；代码有注释掉的8层、16头、FFN4096旧结构。用户所说两层版本的具体模型/参数及实验映射必须以完整LaTeX附件和相应训练/实验配置核准，不得把注释旧结构称为两层版本。

## 7. 文件与来源

- 多预算表：`accuracy/lm_eval/strict_budget_sweep_summary_20261018.md` 与 `strict_budget_sweep_summary_20261018.csv`。
- 旧reviewer controls：`accuracy/lm_eval/reviewer_controls_summary_20261017.md`。
- predictor/embedding审计：`accuracy/lm_eval/neural_controls_audit_13b_20261017.md`。
- 长文本 native baseline：`accuracy/lm_eval/results/long_context_native_baselines_20261018/summary_zh.md`、`summary.csv`、`validated_summary.json`。
- 长文本 controls：`accuracy/lm_eval/results/long_context_cloze_controls_20261018/summary.csv`、`validated_summary.json`。
- Neural/Full throughput：`speedup/results/reviewer_throughput_20261026/REPORT.md`、`summary.csv`。
- Native baseline throughput：`speedup/results/native_flexgen_v8_20261027-v8a/REPORT.md`及各方法原始JSON/log。
- 初稿：`/home/liushenyu/.codex/attachments/70055f06-f772-41d8-9c5c-1ef784cca469/Manuscript.pdf`。
- 评审：`/home/liushenyu/.codex/attachments/9e3af86f-9b06-437d-9660-16e97c33c96a/NEUCOM-D-26-18588-reviews.pdf`。
