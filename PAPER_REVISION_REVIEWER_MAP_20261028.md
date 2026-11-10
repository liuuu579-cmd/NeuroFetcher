# Reviewer意见到论文修改映射

用于在拿到完整LaTeX包后直接逐节修稿。英文段落是建议底稿，不应绕过LaTeX源码中的图表数据核对后直接盲贴。以论文实际采用的 predictor checkpoint、用户确认的实验结果和本目录的结果表为准。

## 先统一全篇术语

- `GPU memory` / `GPU HBM`：替代把显存称作 `on-chip SRAM` 的表述。Ping/Pong是GPU memory buffers，不是片上SRAM。
- `prefetch budget`、`retained-cache density`、`transferred bytes`分别表示配置比例、实际保留项比例、实际搬运字节，不能互换。
- `end-to-end latency`包括prefill与decode；`decode latency`只计生成阶段。Figure 10的60.6 s与21.0 s按评审指出应解释为总时延，修正文字/图注后不要再将其标为decode-only speedup。
- `oracle attention-ranking control`：避免不加限定地称数学 `upper bound`。它使用真实当前 attention 排序，但没有直接优化下游任务loss。
- `previous-step index reuse`须明确索引/排序定义：严格预算控制中使用同层、同head上一query的真实排序，并在当前预算下重选；不能把它说成原样复制全部previous-step selected set，也不能以此单独证明系统异步有效。
- Recent-20只用于Neuro-Fetcher原生定义；其它基线遵循自身机制。逐表列名义budget与实测density。

## 推荐英文替换文字

### Introduction：InfiniGen定位（替换“fixed/content-agnostic/always recent-k”描述）

> InfiniGen is a predictive KV-cache management method rather than a fixed recent-token prefetcher. It estimates the next layer's attention-related access pattern from the current decoding state using partial projections, and selectively prefetches the corresponding KV entries. Neuro-Fetcher is complementary in that it learns a content-conditioned mapping from token representations to cache-selection scores. We therefore compare the methods using their respective selection mechanisms and report both the configured budget and the measured retained-cache density.

### Section 3.1：Predictor architecture 与两个真实版本

拿到LaTeX后先按结果矩阵厘清两个版本对应关系，再以这类句型统一：

> We evaluated two predictor configurations in the development of Neuro-Fetcher. The one-layer configuration uses [verified number of heads], a feed-forward dimension of [verified value], and predicts scores for the first five OPT layers. The two-layer configuration is a separate model variant and is reported only for the experiments that use its corresponding checkpoint. Unless otherwise stated, each result is associated with the predictor checkpoint and configuration listed in Table [X].

若结果核验确认主实验均为一层，表中明确 `main configuration = 1 encoder layer, 4 heads, FFN 2048`；两层版本作为开发/ablation配置单列，不把不存在的两层checkpoint映射写入主结果。若图2或系统实验确实使用两层，则图/系统配置明确标出，正文解释差异来源，避免“全论文单一模型”假象。

同节补入计算流程（按代码实际行为校准）：

> At decoding step (t), the predictor processes the available prefix representations and produces scores for the five predicted shallow layers. The selected indices are determined by the configured threshold and the recent-token rule. The prefix is recomputed at each step in the evaluated implementation; therefore, predictor cost grows with context length and is included in the measured inference time.

若serial/async使用online-prefix predictor而accuracy评估保留历史双向离线版本，要清楚写成两种 evaluation protocol，不能把离线质量评估说成在线因果预测。

### Section 3.2：Previous-step与Ping-Pong边界

> The previous-step control tests whether temporal reuse can replace learned prediction. For each layer and attention head, it uses the preceding query's attention ranking and selects the number of entries required by the current budget. This is an oracle-assisted quality control because it requires the previous query's full ranking; it is not itself a deployable prefetcher. We separately compare serial and Ping-Pong execution to measure the system-level effect of scheduling.

将 `completely hides`、`virtually zero-overhead` 改成有条件结论：

> Ping-Pong scheduling overlaps predictor execution and data movement with LLM computation. The achievable benefit depends on context length and on whether the predictor and transfer stages fit within the available compute window. We report measured serial and asynchronous latency separately and do not assume that the overhead is fully hidden in every workload.

### Section 4.1：实验设置

> Unless otherwise specified, the main model is the original OPT-13B checkpoint. We report the predictor configuration, checkpoint, precision, prompt length, generation length, batch size, nominal cache budget, measured retained-cache density, and timing boundary for each experiment. Accuracy controls use five-shot evaluation on OpenBookQA, Winogrande, COPA, and RTE. Throughput tests report prefill, decode, and end-to-end measurements separately.

将系统实际运行时的硬件描述写成GPU memory/HBM；确认主论文GPU型号后统一为准确型号。把不同runtime/backend作为实现条件透明说明，不要称所有方法运行于相同backend，除非源数据确实如此。

### Figure 2：浅层收益与成本权衡

初稿“accuracy saturates at k=5”被评审指出与曲线不符。建议改为：

> Accuracy continues to improve beyond five predicted layers, while each additional prediction layer increases predictor work. We therefore use five layers as a quality-cost operating point in the evaluated configuration, rather than claiming that accuracy has saturated at this depth. The reuse experiment in Section [X] separately measures the effect of reusing the fifth-layer indices in deeper layers.

图注避免使用 `saturates`。报告准确率、吞吐/延迟归一化方式和误差/重复统计口径。若图中确有另一层数版本，注明checkpoint/架构。

### Figure 7：C4验证数据

> Figure 7 reports heavy-hitter recall on the held-out validation split of C4. C4 is also the corpus used to train the predictor, so this is an in-corpus validation result rather than an out-of-domain generalization test. We compute recall by comparing selected indices against the ground-truth heavy-hitter set at each OPT layer and report the result for each evaluated cache budget.

用户已确认图7数据来自C4验证集。删掉 `on the training dataset` 及暗示held-out不存在的表述；保留“验证split来自训练语料C4”的准确边界。Figure 7的图例/复用来源层编号统一为0-index或1-index，正文注明映射。

### Figure 8：预算公平与新增质量控制

在原Figure 8（Neuro-Fetcher、InfiniGen、H2O、Full、quantization）基础上补入或配套表格列出Oracle per-head、Previous-step per-head、Layer-5 reuse、Quest strict adaptation和Quest official-rule adaptation。预算标签用“名义配置 + 实测density”，解释recent-20只属于Neuro-Fetcher，以及Quest官方规则因最少页数/浅层Full会超预算。

结果讨论须基于完整表格：30%严格预算下Oracle在Winogrande为67.72%，Previous-step49.57%，Layer-5 reuse51.22%，Quest strict50.51%；不要宣称Oracle在每个任务一定高于Full或各方法。原Neural Winogrande66.69%名义30%但实测45.94%，不能将其作为严格同budget胜出结论。对旧Figure 8中“30/25%”混合标签逐项查原始配置并改成实际口径。

量化若目前仍是fake quant：称 `4-bit fake quantization quality control`；只描述PPL/accuracy，不写成实测内存或速度压缩。

### Figure 9与12：Perplexity/跨域表述

Figure 9写清 C4 validation、chunk size 256、prefill/decode或连续chunk的评分方式，并只根据曲线和原始日志描述差异。Figure 12针对WikiText-2和PG-19，若不同曲线相近则客观写明观察到的数值/误差范围；删除“InfiniGen出现明显spike”这类与图线不符的强断言，除非原始数值支持。

### Figure 10/11：时延、吞吐与batch scaling

Figure 10将60.6 s与21.0 s明确标为图中相应设置下的total/end-to-end latency（以原图原始点为准），同时单列prefill和decode；不得把端到端的比值写成decode speedup。新增结果中的 Neural/Full serial/async矩阵见`PAPER_REVISION_RESULTS_20261028.md`。

> Across the evaluated prompt lengths, Ping-Pong scheduling improves throughput at some lengths and is comparable to or slower than serial execution at others. We therefore characterize its benefit as workload-dependent. The measured predictor and transfer costs, end-to-end latency, and buffer capacity are reported alongside throughput.

Figure 11报告实际batch size、输入输出长度、总时延/吞吐单位和每种方法的运行后端。新增InfiniGen/H2O/Quest横向数据可放入主文或补充材料，按现有方法原生机制对比，不新增recent-20给其他方法。保留H2O 1024隔离重复18.66s及并发运行87.21s的运行差异背景。

### Generalization与结论

OPT-13B到OPT-6.7B的迁移单独描述并限制为已测试的OPT checkpoint/配置；说明使用的输入表示/embedding映射来自哪里。LLaMA与跨架构主张本轮暂不开展；不要扩展为对所有家族普遍有效。将结论里的`zero-overhead`、`up to 2.8x`、`without compromising quality`、`all smaller models`改成严格对应测量范围、budget和统计的结果。

## Reviewer回复要点

- R1-1/R2-M1：修正InfiniGen机制定位；明确每个baseline配置、runtime、budget实际密度。
- R1-2/R1-3/R2-M1：统一两种predictor版本与各图checkpoint映射；增加previous-step、layer5 reuse、oracle质量控制；给出serial/async以及预测、传输、LLM时间和物理buffer容量。
- R1-5/R2-M4/M5：完整列出prompt/gen/batch/precision/budget/timing；补充现有context sweep和OPT长文本检索；明确长文本是合成2K内任务而非LongBench。
- R1-6/R2-M1：校正Figure 10端到端/解码时延边界；承认Ping-Pong收益随工作负载变化，不再声称无条件零开销。
- R2-M3：明确Oracle是attention-ranking参照，Previous-step是temporal locality control；与Neural的索引粒度和recent-20规则差异单独披露。
- R2-M5：补Query-aware Quest质量对照和长文本压力测试，区分Quest官方规则与严格预算OPT适配。
- R2-M6及LLaMA相关：本轮不改架构、不启动跨架构实验；OPT-6.7B结果限于已测试配置，不能外推。
- R1-8/R1-10：在拿到LaTeX包后确认数据/代码availability、仓库链接、checkpoint及运行脚本说明，修正空仓库问题的陈述/补充材料。

## 不要做的事

- 不要要求先重跑已被用户接受的baseline，也不要把之前审计文档里的疑问重新当作阻塞。
- 不要把C4验证集写成训练集，也不要把C4内部验证说成out-of-domain。
- 不要把1层与2层predictor版本混作同一个配置；具体映射待LaTeX包和checkpoint核对。
- 不要把Fake quant写成真实packed KV结果；不要把Quest adaptation冒称官方卸载runtime。
- 不要新增对非Neural方法的recent-20；不做LLaMA/跨架构模型实验。
- 不要覆盖或删除原LaTeX源、图、表、参考文献或旧结果；输出带版本号的新稿件目录。
