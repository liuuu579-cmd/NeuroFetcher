# OPT-6.7B Predictor 迁移实现核查

核查日期：2026-10-28。本文依据当前工作区可见代码、脚本、checkpoint 文件名和结果记录整理，供论文修订使用。重点核对“是否用投影矩阵把 OPT-6.7B hidden state 映射到 OPT-13B predictor 输入空间”。

## 结论

在当前仓库可见的 Neural/OPT-6.7B 路径中，**没有找到这样的 learned projection matrix 或 4096→5120 hidden-state 映射代码**。能够直接确认的实现是：Neuro-Fetcher predictor 旁挂一张 OPT-13B 的 embedding table，按输入 token IDs 查询得到5120维 predictor 输入，再将这组输入交给 predictor。这样绕过了直接拿 OPT-6.7B 4096维 embedding 送入固定5120维 predictor 的维数不匹配。

仓库中确实生成和使用了名为 `skewing_matrix` 的矩阵，但这是 InfiniGen 预处理所用的每层 Q/K 空间变换矩阵，不是 6.7B→13B predictor 表示映射。另一个 OPT 自带的 `project_in` 是 embedding dimension 到该 OPT 自己 transformer hidden size 的结构投影；对常见 OPT-6.7B config，两者都是4096维，因而不提供4096→5120适配。

因此，若记忆中的“投影矩阵”是用于解决 predictor 输入维度迁移，**目前代码证据不支持该表述**。可能混淆了 InfiniGen 的 Q/K skewing matrix、OPT 自带的 `project_in`，或“用 OPT-13B embedding table映射同 token IDs”这条路径。若另有训练脚本、实验分支或未同步文件实现了 learned projection，需要从 LaTeX附件/备份中再核对后才能写进论文。

## 代码证据

### Neural 路径使用 OPT-13B embedding lookup

在 `accuracy/src/modeling_opt_neu.py`：

- 约1017–1034行初始化 predictor，输入维度固定5120，active结构是一层 encoder、4 heads、FFN2048，输出5层预测分数。
- 约1035–1046行初始化 `nn.Embedding(config.vocab_size, 5120)` 并从 `/home/liushenyu/test/opt_13b_embed_weights.pth` 载入冻结的 embedding 权重。
- 约1164–1168行使用输入 `input_ids` 查询 `predictor_embed_13b`，输出5120维 `predictor_inputs`。
- 约834–836行把这组5120维输入送入 predictor。

相关代码：

- [`modeling_opt_neu.py`](/home/liushenyu/infinigen/accuracy/src/modeling_opt_neu.py:1014)
- [`modeling_opt_neu.py` forward](/home/liushenyu/infinigen/accuracy/src/modeling_opt_neu.py:1164)
- [`modeling_opt_neu.py` predictor call](/home/liushenyu/infinigen/accuracy/src/modeling_opt_neu.py:831)
- [`extract_13b_embeds.py`](/home/liushenyu/infinigen/accuracy/lm_eval/extract_13b_embeds.py:1)：从 OPT-13B 导出 embedding 权重到上述 checkpoint。

这一实现根据 token ID 复用 OPT-13B 的词向量表示；它不是线性投影，不读取 OPT-6.7B 的 embedding/hidden state来拟合转换矩阵，也不等于跨模型学习到一个映射。

### OPT 自带 `project_in`

在 [`modeling_opt_neu.py`](/home/liushenyu/infinigen/accuracy/src/modeling_opt_neu.py:695) decoder 初始化中，OPT 按 `word_embed_proj_dim != hidden_size` 才创建 `project_in`；forward 约882–883行将其用于该模型自己的 token embedding。

在 OPT-6.7B常见配置中 `word_embed_proj_dim` 与 `hidden_size` 都是4096，因此该模块为 `None`。即使某个 OPT 尺寸需要此模块，它也只把该 checkpoint 的 embedding 维度转为同一模型的 transformer hidden size，不是映射到 OPT-13B predictor 的5120维空间。

### InfiniGen 的 `skewing_matrix`

[`accuracy/setup/gen_opt_model.py`](/home/liushenyu/infinigen/accuracy/setup/gen_opt_model.py:62) 的预处理捕获样本上的 Query/Key projection输出，约74–96行通过每个 attention head 的 Q/K SVD构造矩阵 `A`，并右乘到 query/key 投影权重；约98–105行保存变换后的模型。

相关初始化/应用在 [`accuracy/setup/gen_partial_weight.py`](/home/liushenyu/infinigen/accuracy/setup/gen_partial_weight.py:37) 以及 InfiniGen模型实现。此矩阵用于重排/变换 Q/K 通道，便于partial projection或partial weight selection；它不是 predictor embedding adapter，也不把OPT-6.7B激活变为OPT-13B激活。

### 6.7B 启动配置

旧 [`accuracy/lm_eval/nerual.py`](/home/liushenyu/infinigen/accuracy/lm_eval/nerual.py:18) 对传入模型路径调用 `neural.sh` 并传入 predictor checkpoint；[`accuracy/lm_eval/figure13.sh`](/home/liushenyu/infinigen/accuracy/lm_eval/figure13.sh:13) 则显示 Figure 13 的 Full/InfiniGen 6.7B评估，但该脚本不是清晰的 6.7B Neural迁移证据。

当前 runner 默认把基础 checkpoint 与评估模型路径分别作为参数，但在 Neural model forward 中 predictor 输入由独立13B embedding table产生。历史脚本传入 `/home/liushenyu/opt-6.7b` 时，若确实走到 `modeling_opt_neu.py` Neural 分支，执行的也是 token-ID→13B embedding lookup 逻辑；然而旧结果没有完整的每次运行 metrics manifest，因此当前无法仅凭文件名证明某一张已发表的6.7B曲线对应哪份 predictor权重、embedding导出版本和是否成功加载。

## 找到/未找到的迁移资产

当前 `/home/liushenyu/test` 顶层能找到：

- `opt_13b_embed_weights.pth`，约515 MB，是 OPT-13B embedding权重。
- 多份 predictor checkpoint，包括 `predictor_alpha_2_dyn_posw_epoch_20.pth`、`predictor_topk_0.3_iwl_epoch_20.pth` 等。
- 未找到名称带 projection/projector/mapping/alignment/OPT-6.7B 的权重文件。

在当前仓库源码中搜索 OPT-6.7B Neural路径时，未找到 4096→5120 的 `nn.Linear` adapter、显式矩阵乘、独立 projection checkpoint加载逻辑。这里的“未找到”仅对本次可见的工作区与 `/home/liushenyu/test` 顶层文件有效，不排除其它目录、服务器备份或未上传的源码包里存在额外实现。

## 建议用于论文的准确表述

若要描述**代码已证实的实现**：

> For OPT-6.7B evaluation, the predictor retained its OPT-13B input dimensionality. Rather than projecting OPT-6.7B hidden states into the predictor space, the implementation looked up the same input token IDs in a frozen OPT-13B embedding table and used those 5120-dimensional vectors as predictor inputs. The OPT-6.7B language model itself continued to use its own checkpoint embeddings and hidden states for generation.

这段话只有在逐项核实论文中的6.7B结果确实由这条代码路径生成后，才应作为结果方法描述。如果当前修订范围不需要声称 6.7B predictor transfer，可把主张缩窄为“the 6.7B experiments use the OPT-13B token embedding table as predictor input”并在结果/局限处限定具体checkpoint，不要称 learned projection。

如果后续找到真实 projection matrix，需要再补全：矩阵来源与训练目标、shape（预期4096×5120或其转置）、作用在 token embedding 还是 contextual hidden state、是否含bias/normalization、checkpoint路径/hash、在6.7B结果中实际加载的日志证据，以及与直接使用13B embedding lookup是否为不同实验版本。

## 论文修订时避免的说法

- 不要写“we project the OPT-6.7B hidden states to 5120 dimensions”——当前代码未见该操作。
- 不要把 OPT `project_in` 称为 cross-model predictor adapter。
- 不要把 InfiniGen `skewing_matrix` 称为 6.7B predictor embedding projection。
- 不要仅据此宣称 predictor 能普遍 zero-shot transfer 到所有 OPT 尺寸；需限于实际测试checkpoint，并确认对应实验资产。
- 不要把 6.7B 的 Full/InfiniGen Figure 13结果当作 Neural predictor transfer 结果；需逐项核对图中每个系列和运行脚本。

## 新对话可继续查证的点

1. 解压完整 LaTeX包后，检查 Figure 13源图、legend、数据表和生成脚本，识别OPT-6.7B Neural曲线实际对应哪份checkpoint。
2. 若用户提供某个“投影矩阵”或旧代码目录，检查其shape、加载点、训练来源，并与本文件中三种机制区分。
3. 只有找到运行时加载证据后，才把 learned projection 记入论文；否则以 OPT-13B token embedding lookup 描述实现。
