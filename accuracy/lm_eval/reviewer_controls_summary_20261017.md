本批 Reviewer Controls 已全部完成，最后一项为 RTE / Oracle / 8%，准确率文件写入时间为 2026-10-16T22:39:35.906552+08:00（北京时间）。

配置：OPT-13B，模型路径 `/home/liushenyu/opt-13b`，5-shot，FP16，最大长度 2048，batch size 1，recent_window=1；预算为 30%、20%、15%、8%。36 个配置均具有非空 JSONL、metrics.json 和 accuracy.json。

完整性检查：逐行解析 36 份 JSONL，共 47,592 条结果；请求及顺序与对应输入 JSONL 一致；每条 token 数、logprob 数与 per_request 指标一致；logprob 为有限数；指标中的请求数、token 总数、耗时总和与明细一致；未发现错误。此检查没有重新运行模型或评分器。

下表单位均为 %。OpenBookQA 同时保留 acc 与 acc_norm，其他任务为 acc。acc_norm 是评估器按候选答案长度归一化后的指标，请与论文原有指标保持一致。

| 方法 | 预算 | OpenBookQA acc | OpenBookQA acc_norm | Winogrande acc | COPA acc | RTE acc |
|---|---:|---:|---:|---:|---:|---:|
| Full attention | 100% | 41.20 | 43.80 | 68.27 | 81.00 | 51.26 |
| Previous-step Index Reuse | 30% | 25.20 | 35.00 | 50.43 | 60.00 | 51.62 |
| Previous-step Index Reuse | 20% | 26.00 | 36.60 | 51.62 | 51.00 | 51.99 |
| Previous-step Index Reuse | 15% | 26.40 | 37.60 | 52.17 | 55.00 | 51.26 |
| Previous-step Index Reuse | 8% | 24.40 | 36.60 | 51.14 | 55.00 | 52.71 |
| Oracle | 30% | 36.20 | 43.40 | 63.61 | 77.00 | 55.96 |
| Oracle | 20% | 33.60 | 40.20 | 60.54 | 72.00 | 55.96 |
| Oracle | 15% | 33.00 | 40.00 | 59.83 | 70.00 | 58.48 |
| Oracle | 8% | 24.40 | 34.00 | 53.43 | 56.00 | 53.43 |

Oracle 与 Previous-step 的准确率差值如下，单位为百分点；正值表示 Oracle 更高。

| 预算 | OpenBookQA acc | OpenBookQA acc_norm | Winogrande acc | COPA acc | RTE acc |
|---|---:|---:|---:|---:|---:|
| 30% | +11.00 | +8.40 | +13.18 | +17.00 | +4.33 |
| 20% | +7.60 | +3.60 | +8.92 | +21.00 | +3.97 |
| 15% | +6.60 | +2.40 | +7.66 | +15.00 | +7.22 |
| 8% | +0.00 | -2.60 | +2.29 | +1.00 | +0.72 |

20% 预算时，相对 Full 的变化如下，单位为百分点。

| 方法 | OpenBookQA acc | OpenBookQA acc_norm | Winogrande acc | COPA acc | RTE acc |
|---|---:|---:|---:|---:|---:|
| Previous-step Index Reuse | -15.20 | -7.20 | -16.65 | -30.00 | +0.72 |
| Oracle | -7.60 | -3.60 | -7.73 | -9.00 | +4.69 |

主要观察：Oracle 在 30%、20%、15% 预算下，在四个任务的 acc 上均高于 Previous-step。OpenBookQA 的 8% 预算下，两者 acc 相同，Oracle 的 acc_norm 反而低 2.60 个百分点。

Previous-step 在 OpenBookQA、Winogrande、COPA 上相对 Full 有明显下降。RTE 的 Full 为 51.26%，对照结果约 51%—58%，且没有随预算增大而持续改善的趋势；不能概括为所有任务都下降。

这里的 Oracle 使用当前查询的真实 attention score，在包含最近 token 的预算约束下选 top-k。它是理想索引选择对照，不能理解为下游任务准确率的严格数学上界。Previous-step 的结果对应当前实现：递归复用上一步已选的索引，并强制保留最近 token。

模型前向耗时如下，区间为同一任务四档预算的最小值至最大值，单位为分钟。

| 任务 | Full | Previous-step | Oracle |
|---|---:|---:|---:|
| OpenBookQA | 1.52 | 28.06—28.44 | 22.75—23.54 |
| Winogrande | 1.95 | 46.28—46.76 | 38.09—42.43 |
| COPA | 0.17 | 2.61—2.68 | 2.13—2.30 |
| RTE | 0.76 | 34.02—34.40 | 27.67—29.10 |

36 个配置记录的模型前向耗时累计 13 小时 46 分 14.0 秒（49574.015 秒）。这个总数不含模型加载、make_result 中的概率及 token 结果整理、文件写入和准确率评分，不代表整批实验的墙钟时长。

当前实现先计算完整 attention score，再通过逐 token 的 Python 循环选择索引、构建掩码；Full 走并行矩阵计算。因此这批耗时用于解释本次实验成本，不能作为稀疏 KV 系统的加速比结果。

原始 metrics.json 的 control 计数及 mean_previous_step_iok 只对应最后一条请求：runner 在每条请求开始时重置状态，并在全部请求结束后读取控制指标。IOK 表示相邻已选集合的重叠，也不能当作对当前 Oracle top-k 的命中率。此次汇总没有将这些字段当作全测试集统计。

本报告汇总的是新跑的 Full、Previous-step 和 Oracle；尚未在表中加入论文原方法及其他已有 baseline。与它们合表前应核对 checkpoint、shot 数、输入集合、预算定义及准确率指标。

各任务每个配置的推理请求数：OpenBookQA 2,000；Winogrande 2,534；COPA 200；RTE 554。这里的请求数包含每个题目的各候选答案，不能直接当作题目数。

CSV 中 acc、acc_norm、stderr 保留 0—1 的原始尺度；percent 列为百分比；delta_vs_full_pp 列为百分点。JSON 同时保留完整性检查概况及各条件汇总。

生成时间：2026-10-17T09:50:43.944004+08:00；读取环境：`/home/liushenyu/anaconda3/envs/infinigen/bin/python`。
