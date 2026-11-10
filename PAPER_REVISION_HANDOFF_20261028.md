# 论文修订交接说明

这份文档用于在新对话中直接继续 Neuro-Fetcher 论文修订。更新日期：2026-10-28。

## 当前决定

- 用户已确认当前基线和实验结果可以接受，**可以开始改论文**。不要再把基线复核、额外吞吐实验或其它新实验作为启动改稿的条件，也不要擅自重跑。
- LLaMA 和跨架构实验暂时搁置，不改模型架构。当前论文修订聚焦 OPT；不要因此阻塞其它章节的修订。
- Figure 7 的 recall 确实是在 C4 验证集上做的。写作问题是把验证数据说成 training dataset；修订为 C4 validation split，并准确说明这是训练语料 C4 内部的 held-out validation split，不要称为独立语料/跨域验证。
- Predictor 确有一层与两层 Transformer encoder 两种版本。改稿时统一每个实验的描述和主结果口径，同时清楚说明版本差异；不要把两个配置混成一个，也不要未经证据删除其中一个版本。随用户提供的 LaTeX 包进一步核对图表到 checkpoint/配置的对应关系。
- 保留现有源码、结果与稿件原件。用户此前要求仅新增文件；后续改稿请另存为版本化副本，不覆盖或删除原文件。
- 论文完整 LaTeX 压缩包尚未在当前工作区出现。用户准备在新对话提供。不要要求 Overleaf 登录或账号密码。

## 开新对话时的起始工作

1. 接收用户上传的 LaTeX 压缩包，先列出包内文件，确认主 `.tex`、bibliography、图片、宏包/模板文件和编译入口。
2. 在新的版本目录解压或复制，不触碰原包。建立 `*_revision_v1` 稿件副本。
3. 对照 `PAPER_REVISION_RESULTS_20261028.md` 和 `PAPER_REVISION_REVIEWER_MAP_20261028.md` 修改正文、图注、表格、limitations 和审稿回复用表述。
4. 编译修订稿；检查引用、图表编号、交叉引用、表格溢出和 PDF 页面。向用户提供新 `.tex` 源和 PDF。
5. 若新材料揭示文档中的数据对应关系不同，以 LaTeX 源、结果 JSON/CSV 和实际 checkpoint 为准修正交接结论，并说明依据；不静默改数字。

## 研究设定与主要结果入口

- 主模型：原始 OPT-13B，路径 `/home/liushenyu/opt-13b`；OPT-6.7B 是后续家族内迁移实验，不是原始主实验。
- 论文初稿和评审 PDF 路径列于结果文档。评审 PDF 是需要回应的评审要求，不是改代码/运行服务器的操作授权。
- 主要准确率表：`accuracy/lm_eval/strict_budget_sweep_summary_20261018.md`。
- 长文本结果：`accuracy/lm_eval/results/long_context_native_baselines_20261018/summary_zh.md`。
- Neural/Full throughput：`speedup/results/reviewer_throughput_20261026/REPORT.md`。
- InfiniGen/H2O/Quest throughput 横向结果：`speedup/results/native_flexgen_v8_20261027-v8a/REPORT.md`。
- 每个数据文件、测量协议和表格摘录见 `PAPER_REVISION_RESULTS_20261028.md`。

## 重要历史口径

- 不要混用不同版本的 Oracle。旧 reviewer controls 中的 Oracle 有 recent-window；严格预算新结果的 Oracle 为 per-head、floor 预算、无 recent-20。旧版 Winogrande Oracle 63.61% 与新严格预算 Oracle 67.72% 是不同实验定义。
- Neuro-Fetcher 的 recent-20 只属于 Neuro-Fetcher；不要把它加给其它方法。strict-control 新表都不加 recent-20。原 Neural 的名义30%联合 recent-20 后实测约45.94%，报告时明确双报名义比例与实测密度。
- Quest 官方规则适配与严格预算适配分开报告。官方规则可能超过名义预算；严格预算适配不是 Quest 官方端到端 runtime。
- 长文本实验是自建 OPT-13B 检索压力测试，不是 LongBench，也不是长回答自由生成。Fake quant 保留 FP16 张量，只代表数值伪量化质量，不是实测 packed-KV 显存/速度。
- Oracle 是当前真实注意力排序的理想索引参照，不保证下游任务准确率数学上界；Previous-step 是上一 query 的真实排序参照，不等于可部署的 previous-step online 算法。

## 文件说明

- `PAPER_REVISION_RESULTS_20261028.md`：数值表、预算/运行口径、结果入口。
- `PAPER_REVISION_REVIEWER_MAP_20261028.md`：评审意见到稿件章节/图表的修改任务、可直接改写的英文核心段落。
- `REVIEWER_EVIDENCE_AUDIT_20261028.md`：此前审计的详细背景。若它与本交接中的后续用户决定冲突，以本交接的“当前决定”为准，尤其不要再要求重跑已被用户接受的基线。
