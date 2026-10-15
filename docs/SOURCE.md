# 代码来源与整理记录

- 整理日期：2026-10-15。
- 源工作区：`infinigen`。
- 上游仓库：`https://github.com/snu-comparch/infinigen`。
- 原工作区基准提交：`3754f52`，同时包含工作区中已修改和未提交的源码。
- 第三方 Transformers checkout：`v4.35.2`；完整源码及其 Git 历史未纳入本仓库。
- 仓库目标：`https://github.com/liuuu579-cmd/NeuroFetcher`。

## 整理内容

保留 `accuracy/` 与 `speedup/` 的源码结构、脚本、实验配置、原始版权声明及许可证。保留所有实验版本，避免仅凭名称丢弃仍可能使用的实现。

以下源码从结果目录移到源码目录：

| 原相对路径 | 本仓库路径 |
| --- | --- |
| `accuracy/lm_eval/results/evaluate_c4_accuracy.py` | `accuracy/lm_eval/evaluate_c4_accuracy.py` |

本地 Transformers 中的两个修改文件都是绝对符号链接：`modeling_opt.py` 指向 `accuracy/src/modeling_opt_neu.py`，`modeling_opt_orig.py` 指向 `accuracy/src/modeling_opt_ours_setup.py`。两个目标源码都已包含在本仓库中，因此不复制绑定原机器路径的链接或重复保存目标文件。

新增中文 README、代码索引、Git 忽略规则和文本换行规则；保留上游 README 为 `docs/INFINIGEN.md`。Shell 脚本统一使用 LF 换行，消除原 CRLF 导致的 Bash 解析错误。算法代码不作功能性重构。

## 排除内容

模型 checkpoint（包括 OPT/H2O）、partial weights、skewing 数据、生成的 embedding、提示词和数据集文本、JSONL/CSV 评测输入输出、日志、Python 缓存、安装生成的 egg-info，以及完整 Transformers checkout。

这些文件仍留在原始工作区。该整理目录是独立代码副本，不会减少原工作区占用的磁盘空间。
