# 代码索引

## 精度与预取实现

| 文件/目录 | 用途 |
| --- | --- |
| `accuracy/src/modeling_opt_neu.py` | 神经网络预取的 OPT 实现 |
| `accuracy/src/modeling_opt_inf.py` | InfiniGen OPT 实现；perplexity 入口会按策略选择它 |
| `accuracy/src/modeling_opt_ours_v1.py` | 隔层预取变体；当前 `run_lm_eval_harness.py` 的 InfiniGen 策略选择此文件 |
| `accuracy/src/modeling_opt_ours_v2.py` | 原笔记记录为轻量 Transformer 预测 Q，再与实际 K 计算分数的变体 |
| `accuracy/src/modeling_opt_ours_v3.py` | 原笔记记录为直接预测 attention 分数的变体 |
| `accuracy/src/modeling_opt_orig*.py` | 原有 OPT 参考实现及实验版本 |
| `accuracy/src/modeling_*_setup.py` | 模型预处理阶段使用的实现 |
| `accuracy/src/modeling_*_prune.py` | pruning 相关变体 |
| `accuracy/src/predictor_model*.py` | 预测器网络定义；不同目录的副本按原样保留 |
| `accuracy/lm_eval/nerual*.py` | 神经预取任务脚本；保留原有 `nerual` 拼写，避免破坏外部引用 |
| `accuracy/lm_eval/inf*.py` | InfiniGen 任务与参数实验脚本 |
| `accuracy/lm_eval/h2o*.py` | H2O 对照实验脚本 |
| `accuracy/lm_eval/evaluate_c4_accuracy.py` | 原位于 `accuracy/lm_eval/results/` 的源码，整理到评测源码目录 |

## 性能实现

`speedup/flexgen/infinigen/` 保留 `flex_opt` 与 `pytorch_backend` 的成对版本。

| 后缀 | 原工作区笔记记录的用途 |
| --- | --- |
| `_org` | 原始参考文件 |
| `_v1` | 修改后的过渡版本 |
| `_v2` | 串行预取与计算 |
| `_v3` | 并行预取与计算 |
| `_v4` | 并行预取代码的整理版本 |
| `_v5` | 内存管理与显存占用优化的过渡版本 |
| `_v6` | prefill 与 decode 显存占用优化 |
| `_v7` | 增加详细计时输出 |
| `_shallow` | shallow 预取计时实验 |

以上用途来自原工作区笔记，不代表本次已运行或验证每个版本。默认 `speedup/flexgen/flexgen/flex_opt.py` 和 `pytorch_backend.py` 的相对符号链接保留，仍指向 `../infinigen/`。原有 `*_shallow.py` 链接的目标也按工作区原样保留；切换变体前需确认脚本实际选中的文件。

## 运行前需检查

- 原工作区大量脚本包含绝对的模型、预测器和语料路径。按实际环境修改后运行；权重和数据未随代码上传。
- `accuracy/transformers/` 需要自行克隆并可编辑安装，版本为原工作区的 `v4.35.2`。
- 原始工作区已删除 `accuracy/lm_eval/ours.sh` 和 `accuracy/src/modeling_opt_ours.py`。旧图表脚本和旧评测入口仍有相关引用；归档保留这个状态，没有猜测应替换成哪个实验版本。
- `ours_v1.sh` 是历史脚本，使用 `--prefetch-strategy`，而当前统一入口使用 `--strategy`；运行前应对齐参数。可优先检查 `infinigen.sh` / `neural.sh`。
- 本次没有执行模型推理和 GPU 性能评测。静态语法检查不验证运行时导入、模型兼容性、外部数据或实验指标。
