# OPT-13B 新增对比：30%预算、无额外近期窗口

模型 `/home/liushenyu/opt-13b`（40层、40个 MHA heads），5-shot，四个数据集。所有新对比按 head 独立选索引，不加载神经预测器，不额外加入当前 token 或最近20个 token。

严格预算为 `max(1, floor(0.30 × 当前可见 token 数))`。最短前缀至少保留1项，因此长度1–3存在必要的取整例外；其余前缀向下取整。实际比例按全部请求、层、head 的选择项总数，除以完整因果注意力项总数统计。它是 attention/KV 选择密度，不是跨 head 去重后的传输字节数。

| 方法 | Winogrande acc | OpenBookQA acc / acc_norm | COPA acc | RTE acc |
| --- | ---: | ---: | ---: | ---: |
| Oracle（每 head） | 67.72% | 42.40% / 43.40% | 81.00% | 50.90% |
| Previous-step（每 head） | 49.57% | 26.80% / 35.00% | 57.00% | 49.46% |
| 第5层索引复用（每 head） | 51.22% | 26.40% / 34.20% | 58.00% | 54.51% |
| Quest：严格30%适配 | 50.51% | 26.40% / 38.20% | 48.00% | 56.32% |
| Quest：官方选择规则的 OPT 适配 | 49.41% | 32.60% / 41.00% | 63.00% | 56.32% |

| 方法 | Winogrande 实际比例 | OpenBookQA 实际比例 | COPA 实际比例 | RTE 实际比例 |
| --- | ---: | ---: | ---: | ---: |
| Oracle（每 head） | 29.40% | 29.21% | 29.13% | 29.82% |
| Previous-step（每 head） | 29.40% | 29.21% | 29.13% | 29.82% |
| 第5层索引复用（每 head） | 29.40% | 29.21% | 29.13% | 29.82% |
| Quest：严格30%适配 | 29.40% | 29.21% | 29.13% | 29.82% |
| Quest：官方选择规则的 OPT 适配 | 57.78% | 69.42% | 75.09% | 33.31% |

Oracle 使用当前运行状态的真实 QK 排序，各 head 独立 top-k；它是理想化选择参照，不保证任务 accuracy 高于 Full 或其他方法。
评审2第3条字面要求的 label-based Oracle 包含 heavy-hitter + recent-window。本次按用户指定取消其他方法的近期窗口，因而本 Oracle 应称为无近期窗口的严格预算参照，不能写成与训练标签完全一致的上界。之前的 recent-20 Oracle 结果仍单独保留在旧验证目录。

Previous-step 使用同一层、同一 head 的前一个 token/query 的真实 QK 排序，在前一步历史内按当前 k 重新选取。首 token 自举；不递归复制稀疏集合，也不额外保留当前 token。这需要理想化的前一步完整排序，用于准确率参照；不能仅凭此结果证明异步调度收益或可实现延迟。

第5层复用：第1–5层各自使用 Oracle，第6–40层复用当前 query 在第5层选出的索引，head 编号对应；每个请求独立重置。不同层相同编号的 head 不保证有相同语义。

Quest 官方规则的 OPT 适配：16-token 页、sign-aware min/max key 页上界、`min(可见页数, max(3, floor(token_budget / 16)))` 页，前两层 Full。保留这些规则导致实际比例偏离30%；本项不声称严格30%。参照官方准确率代码，未混入其优化运行时额外处理最新页的规则。

Quest 严格30%适配：按相同页上界排序，最后一页可只取部分 token，取消浅层 Full 和至少3页例外。页分数相同按较早 token 位置打破并列，不使用真实 QK 再排序。它是自行实现的预算适配，需与保留官方规则的一项分开标注。

两项 Quest 均为 OPT/PyTorch 的离线准确率适配：原仓库支持 Llama/Mistral，原准确率代码对 prefill 使用 Full、decode 使用选择；这里为与现有 lm_eval 口径一致，在整段输入的每个位置模拟因果 decode 选择，prompt 和答案位置都使用相应方法。原仓库为固定绝对 token_budget，这里改为每个前缀按30%生成预算。因此这不是原生 Quest 端到端部署或官方速度结果。当前未完成页只使用已可见 token 的 key min/max，避免未来信息泄漏。

Quest 参照：<https://github.com/mit-han-lab/Quest/blob/01c1623bf9395009520874e989e29f683203b357/evaluation/quest_attention.py>。

先前已复现的 Winogrande 参照：

- full: acc=0.6827150750，实际比例=100.00%。
- neural: acc=0.6669297553，实际比例=45.94%。

Neuro-Fetcher 的30% top-k 后额外并入最近20个 token，实际约45.94%，本次保留此原规则。它与严格30%对比不等预算；按用户要求同时报告名义比例和实际比例。这些对比不能单独将差距归因于学习：预算、近期窗口和 shared/per-head 索引粒度也不同；previous-step 的准确率也不能代替串行/异步实际延迟消融。Neural predictor 为1个 encoder layer、4 heads、FFN 2048；权重 `/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth`，embedding 在 checkpoint 加载后恢复。原 predictor 整段运行未传入因果 mask，可看整段输入；历史分数的复现不等于在线因果预测已验证。其他任务尚无使用这套已核实 predictor 的配对 neural 复跑结果，不从旧的不兼容配置拼接比较。

20组结果均逐条核验请求数量、顺序、内容和有限 log probability，并核对源码/输入哈希、checkpoint 无缺失或形状不匹配、预算统计。严格预算四种方法的预算不匹配行数均为0；Quest 保留官方规则的一项单独记录例外。12项确定性测试及 GPU 小规模验证通过。前向耗时仅记录为运行诊断，不用于延迟或加速结论。

新增文件入口：`run_strict_reviewer_controls_suite.sh`；使用 infinigen Python，最多 GPU0–3、每卡10个 CPU 核心、总40/64核心。原文件和历史结果保留。完整结果可跳过；源码或输入哈希改变、存在不完整输出时会拒绝静默覆盖。

```bash
cd /home/liushenyu/infinigen/accuracy/lm_eval
conda run -n infinigen bash run_strict_reviewer_controls_suite.sh
conda run -n infinigen python summarize_strict_reviewer_controls.py
```
