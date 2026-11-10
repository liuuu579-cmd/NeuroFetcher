# OPT-6.7B Predictor 输入迁移：论文表述

## 实现事实

OPT-13B predictor 的输入维度为5120；OPT-6.7B 的模型 hidden size 为4096，不能把 OPT-6.7B 的 hidden states 直接输入原 predictor。当前实现为 predictor 单独保留 OPT-13B 的 token embedding table。给定输入 token IDs (x_1,\ldots,x_T)，通过该表查询

\[
\mathbf{e}^{(13B)}_t = E_{13B}[x_t] \in \mathbb{R}^{5120},
\]

然后将这些向量送入 OPT-13B predictor。与此同时，OPT-6.7B 语言模型仍使用自己的 embedding、hidden states 和 Transformer 层完成正常的文本生成。也就是说，两条输入路径共用 token IDs，但各自使用所属模型的表示空间。

该方法**没有**把 OPT-6.7B 的4096维 hidden state线性投影到5120维；它复用的是 OPT-13B 词嵌入表，不是 learned cross-model projection。当前代码中的 OPT `project_in` 和 InfiniGen `skewing_matrix` 都不是这项适配。

## 可直接用于论文的英文段落

> **Predictor input for OPT-6.7B.** The predictor was trained with 5120-dimensional OPT-13B token embeddings, whereas OPT-6.7B has a 4096-dimensional hidden representation. To preserve the predictor's input space, we use a frozen OPT-13B embedding table as a separate input source for the predictor. For each input token ID, the corresponding 5120-dimensional OPT-13B embedding is looked up and passed to the predictor. In parallel, the OPT-6.7B language model uses its own embeddings and hidden states for language-model inference. Thus, the predictor does not consume OPT-6.7B hidden states, and no learned 4096-to-5120 projection is applied. This evaluation tests transfer across the two OPT checkpoints using a shared tokenizer and the OPT-13B embedding space.

## 建议的简短版本

> We adapt the OPT-13B predictor to OPT-6.7B by looking up each input token ID in a frozen OPT-13B embedding table and feeding the resulting 5120-dimensional vectors to the predictor. OPT-6.7B continues to generate with its own model representations; no projection of its 4096-dimensional hidden states is used.

## 术语建议

- 推荐：`OPT-13B embedding lookup for predictor inputs`、`shared-tokenizer embedding-space transfer`。
- 可用但需限定：`intra-family transfer across OPT-13B and OPT-6.7B`。
- 避免：`projection matrix`、`projecting OPT-6.7B hidden states`、`dimension-mapping layer`。
- 不要由单一 OPT-6.7B 实验推导“适用于所有 OPT 尺寸”；只陈述实际测试的 checkpoint 与配置。

## 源码依据

- [`modeling_opt_neu.py`](/home/liushenyu/infinigen/accuracy/src/modeling_opt_neu.py:1014)：初始化5120维 predictor 和独立 OPT-13B embedding table。
- [`modeling_opt_neu.py`](/home/liushenyu/infinigen/accuracy/src/modeling_opt_neu.py:1164)：按输入 token IDs 查询 predictor embedding。
- [`modeling_opt_neu.py`](/home/liushenyu/infinigen/accuracy/src/modeling_opt_neu.py:831)：把查询到的表示输入 predictor。
- [`extract_13b_embeds.py`](/home/liushenyu/infinigen/accuracy/lm_eval/extract_13b_embeds.py:1)：导出 OPT-13B embedding 权重。
- 完整核查及容易混淆的两种矩阵说明见 [`OPT_67B_PREDICTOR_TRANSFER_AUDIT_20261028.md`](/home/liushenyu/infinigen/OPT_67B_PREDICTOR_TRANSFER_AUDIT_20261028.md)。

## 结果描述边界

这项适配说明 predictor 输入从哪里来，不单独证明所有较小 OPT 模型都具有相同迁移效果。论文中应把使用的 OPT-13B/OPT-6.7B checkpoint、predictor checkpoint、embedding table 和评估任务对应说明；如果某个结果实际使用了另一 predictor 或 embedding 配置，应分别标注。
