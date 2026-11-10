# 13B reviewer-control audit

The original Winogrande neural comparison uses `/home/liushenyu/opt-13b` and
`/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth`, as recorded in
`qqq_fullcache_30.py`. The active predictor definition in
`accuracy/src/modeling_opt_neu.py` has one Transformer encoder layer, four
attention heads, and FFN width 2048. The commented eight-layer/16-head/4096
definition belongs to a different checkpoint and is not used here.

The predictor embedding is an extra 13B embedding loaded by the model
constructor. The standard checkpoint loader initializes that missing parameter
again after construction. The new runner therefore restores
`/home/liushenyu/test/opt_13b_embed_weights.pth` after `from_pretrained`.
The diagnostic recorded `matches_before_reload=false` and
`matches_after_reload=true`; with restoration, neural Winogrande is
`0.6669297553`, matching the historical `0.6669` result. The old behavior is
still available with `--predictor-embedding-mode original` for diagnosis.

## Winogrande, 13B, 5-shot, nominal 30%, recency union 20

| condition | accuracy | actual retained entries |
| --- | ---: | ---: |
| Full | 0.682715 | 100.00% |
| Neural, restored embedding | 0.666930 | 45.94% |
| Oracle, per-head true attention | 0.684294 | 45.50% |
| Oracle, shared true attention | 0.664562 | 43.08% |
| Previous-step, shared true attention | 0.666140 | 43.41% |

The Oracle is an idealized true-attention selection reference, not a
guaranteed accuracy upper bound. It ranks the
current true attention probabilities and keeps the top-k positions, while the
model's output loss is not itself optimized. Per-head selection and shared
selection are different controls; the original neural method uses one shared
index set.

The nominal budget is `ceil(history_length * ratio)` followed by a union with
the configured recent window. Consequently, the measured density is above
30%, especially for short histories. These results use recency 20 to match
the original neural implementation. A run with recency 1 is a different
experiment and should not be compared as the same budget setting.

The previous-step control uses the preceding query's true-attention ranking,
then reselects at the current query's budget. It does not recursively reuse
its own previous selected set. This definition isolates the reviewer control
and should be stated explicitly in the manuscript.

Results are in `results/neural_controls_validation_20261017/`. The implementation
is additive: `modeling_opt_neural_controls.py`,
`neural_reviewer_controls.py`, and the validation runners do not modify the
existing model or evaluation files.

## 原始模型与旧结果的区别

13B 是原始实验，6.7B 属于后续迁移实验，不混入本次主对比。
`NUM_PREDICT_LAYERS=5` 指输出的 OPT 层分数组数，不是 predictor 深度；
主模型 OPT-13B 有 40 层。其他旧任务脚本引用的
`predictor_alpha_2_dyn_posw_epoch_20.pth` 确实是 8 层权重，不能直接用于当前 1 层结构。
新 runner 严格加载权重，不忽略或裁剪不匹配的参数。

旧 `modeling_opt_reviewer.py` 基于 `modeling_opt_orig.py`；新 controls 基于原 neural。
旧 Oracle 使用计入总预算的 recent-1，原 neural 则使用 top-k 后额外并入 recent-20。
这解释了旧 Oracle 0.636148 与 neural 0.6669 不具备直接可比性。
旧 previous-step 递归复用自己选出的集合，也不同于本次的上一 query 真实排序参照。
旧报告、原代码及历史输出均保留；本报告说明与 neural 对齐后的实验。

## 其他任务：13B、5-shot、Oracle 名义 30% + recent-20

| 任务 | Shared acc | Per-head acc | Shared / Per-head 实测比例 |
| --- | ---: | ---: | ---: |
| OpenBookQA | 0.426 | 0.414 | 48.47% / 50.88% |
| COPA | 0.780 | 0.800 | 51.69% / 53.86% |
| RTE | 0.498195 | 0.512635 | 32.26% / 33.56% |

OpenBookQA 的 acc_norm 分别为 0.452 / 0.446。
这三个任务尚未用恢复 embedding 的 neural 重做完整配对，不能据此声称超过历史 neural。
目前完成的是 30% 设置的定位验证，其余预算的旧结果不代表修正后的结果。

## 论文表述边界

- Neural 原实现共享索引；per-head Oracle 的选择粒度更灵活，需要独立标注。
- 实测比例统计所有请求、所有层和 heads 的保留注意力项，分母为完整因果注意力项。
  最近窗口与 top-k 的重叠，以及阈值并列，会改变实际保留数。
  同一名义比例不代表相同的实测比例，更不能把这些结果写成严格总 KV 预算 30%。
- Oracle 使用当前稀疏运行状态中的真实注意力，并非另一次 Full 运行的注意力。
  Shared 按各 head 的真实注意力概率均值排序；per-head 各自排序。
  真实注意力 top-k 不直接优化任务 accuracy，不保证超过 neural 或 Full。
  本次也没有对 0.6843 与 Full 0.6827 的差距作统计显著性结论。
- Previous-step 根据上一 query 的真实排序，在前一步可见历史内按当前 k 重新选取，
  再并入当前 recent-20；并非原封不动复制前一步 top-k 集合。
- 本 runner 支持无 padding、无 cache 的整段离线准确率验证，不是解码速度实验。
  原 predictor 的 Transformer encoder 未传入因果 mask，能读取整段输入；
  最终 KV mask 因果不代表 predictor 本身只看过去。这里保留该行为以复现历史分数，
  若论文声称在线因果预测，还需要单独核对这一点。

## 复现与验证

新增 runner 的 embedding 默认模式已改为 `restore`；诊断模式 `original` 仍可显式指定。
Winogrande 共 2534 个候选请求、1267 道题；恢复 embedding 的 neural 使用
`-embedding-restored` 文件名，0.5146 的诊断结果不作为正式基线。
八个确定性单元测试已通过，覆盖原计算一致性、预算与窗口、shared 排序、
previous-step、因果边界与全请求累计统计。

续跑 Winogrande 30% 验证（最多 GPU 0–3 四个任务，各 10 CPU 线程）：

```bash
cd /home/liushenyu/infinigen/accuracy/lm_eval
conda run -n infinigen bash run_neural_controls_validation.sh
```

完整配置按 metrics 和 accuracy 文件跳过。
旧 `run_reviewer_controls_suite.sh` 保留旧口径，不是本次与 neural 对齐的入口。

已逐条核验 12 个完成配置（含 1 个未恢复 embedding 的诊断配置）：
请求数量、顺序和内容与对应输入一致，所有非首 token 的 log probability 均有限。
汇总表为 `neural_controls_validation_summary_20261017.csv`。
新增 Python 文件编译检查和 shell 脚本语法检查均通过。
