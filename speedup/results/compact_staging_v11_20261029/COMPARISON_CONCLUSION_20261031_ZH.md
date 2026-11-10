# 显存与 FlexGen 对比结论（2026-10-31）

配置：OPT-13B FP16、batch=4、输入1920、生成128 tokens，其中127步为 decode。显存比较采用 PyTorch peak allocated，不能与 NVML 整卡 used 混用。

| 方法 | Decode吞吐，均值±样本标准差 | allocated峰值 | reserved峰值 |
|---|---:|---:|---:|
| 原生 FlexGen，全部权重GPU、KV在CPU、overlap=false | 5.59±0.20 token/s | 27.310 GiB | 28.488 GiB |
| Neuro-Fetcher紧凑缓冲异步版，FP32 predictor | 14.43±2.33 token/s | 29.728 GiB | 30.575 GiB |

FlexGen五次独立进程测量，每次先做一输出token的原生warmup；Neuro-Fetcher为已有的同一进程一次完整warmup后五次测量。两批实验时间、运行路径和warmup不同，不能解释成同后端受控的算法因果实验。两者decode吞吐均使用batch×127/decode时间，跨实现的五次均值比为2.58。额外allocated约2.418 GiB，即8.9%。原生FlexGen此前单次为5.93 token/s，本轮重复均值为5.59。

## 为什么还高于 InfiniGen 加预测器

目前异步实现固定常驻：OPT模型约23.941 GiB、FP32 predictor含buffers约0.564 GiB、两个覆盖40层的KV staging槽合计3.876 GiB，固定存储已经约28.38 GiB，尚未计入临时激活。故25.5+0.56不能作为当前异步实现峰值的估计；两个实现的缓冲和工作区策略不同。

若继续做到26–27 GiB，应优先研究按层流式KV staging：两槽各只保留一层时，按相同635-token容量，理论缓冲约0.097 GiB，较现状少约3.78 GiB。由29.73简单减算可得到约26 GiB的方向性估计，但临时激活峰值和并发关系也会改变。该方案未实现、未测量，不能写作已达成的结果；逐层搬运可能使当前预取等待增加，必须重测吞吐和输出一致性。

## FP16 predictor诊断：不作为可直接替换版本

同一1920输入、batch4、完整128输出的单次自由生成诊断：FP32变为FP16后allocated由29.728降至29.033 GiB，但127个decode步骤中有126步索引哈希不同，4个样本中2个生成序列发生变化，logits也不一致。这是改变精度的实验，不是无损内存优化。后续轨迹差异同时包含先前输出分歧的影响，不能把所有索引差异单独归因于同一前缀的排序变化。未经独立准确率/困惑度评估，不替换已接受的FP32方案。

此处127个decode步骤的说明也纠正了同目录英文备忘中“128 decode steps”的计数笔误。FP16吞吐只有诊断单次结果，不作正式加速结论。

## 当前可采用结论

当前29.73 GiB方案相对于上述原生FlexGen配置，以额外约8.9%的allocated显存得到约2.58倍的测量均值吞吐，是有价值的折中。Full GPU仍更快，论点应限定在CPU KV offload场景。继续降到26–27 GiB是另一项调度实现工作，应保留当前已验证版本作为基准，避免只压显存却损失大部分吞吐。

数据来源：

- `speedup/results/flexgen_repeats_v15_20261031/summary.json`及五个原始JSON。
- `speedup/results/compact_staging_v11_20261029/benchmark_v13/p1920-g128-b4-neural_async.json`。
- `speedup/results/compact_staging_v11_20261029/predictor_half_v15/opt13b_p1920_g128.json`。

本轮仅新增脚本和记录；所有此前源码、checkpoint和结果均保留。
