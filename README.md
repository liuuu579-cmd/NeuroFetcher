# NeuroFetcher

从本地 InfiniGen 实验工作区整理的代码仓库，包含神经网络 KV cache 预取、InfiniGen/H2O/量化评估，以及基于 FlexGen 的性能实验代码。现有实验变体按原目录和文件名保留，便于对照和继续开发。

## 目录

| 路径 | 内容 |
| --- | --- |
| `accuracy/src/` | OPT/LLaMA 模型实现、InfiniGen 与神经预取变体、预测器结构 |
| `accuracy/setup/` | 模型转换、partial weights、skewing matrix 生成脚本 |
| `accuracy/lm_eval/` | 任务生成、准确率评估、预取策略启动脚本 |
| `accuracy/perplexity/` | perplexity、语言建模及其他评估实验 |
| `accuracy/scripts/` | 按原论文图表组织的评测批处理脚本 |
| `speedup/flexgen/` | FlexGen 原始、H2O、InfiniGen、shallow 及版本化后端 |
| `speedup/infinigen/` | KV 选择、partial weights 和 skewing 控制器 |
| `speedup/scripts/` | 性能对比和消融实验脚本 |
| `speedup/uvm/` | UVM/CUDA 内存分配及 attention 实验 |
| `docs/` | 代码索引、上游说明和归档来源 |

详细入口与变体说明见 [代码索引](docs/CODE_MAP.md)。

## 环境准备

原项目使用 Python 3.9、PyTorch 2.0.1。当前工作区的 Transformers checkout 为 `v4.35.2`；评测脚本会将本仓库的模型实现链接到这个可编辑安装中，因此需按下面的相对路径安装源码依赖。

```bash
git clone https://github.com/liuuu579-cmd/NeuroFetcher.git
cd NeuroFetcher
conda create -n neurofetcher python=3.9
conda activate neurofetcher
python -m pip install -r requirements.txt
git clone --depth 1 --branch v4.35.2 https://github.com/huggingface/transformers.git accuracy/transformers
python -m pip install -e accuracy/transformers
python -m pip install -e speedup/infinigen -e speedup/flexgen
```

模型权重、预测器权重、数据集和提示词文本需要另外准备。执行实验前，请检查脚本中的模型/数据路径；已有实验脚本保留了原工作区路径和参数。

## 评测入口

在 `accuracy/lm_eval/` 下：

- `generate_task_data.py`：生成 lm-eval 任务输入。
- `run_lm_eval_harness.py`：统一评测入口，支持 InfiniGen、neural、H2O、量化和 full-cache 相关参数。
- `neural.sh`、`infinigen.sh`、`h2o.sh`、`quant.sh`、`full_cache.sh`：对应策略的启动脚本。
- `evaluate_task_result.py`、`evaluate_c4_accuracy.py`：结果评估工具。

`accuracy/perplexity/opt.py` 是 OPT perplexity 等实验的入口；性能实验入口在 `speedup/scripts/`。

评测时请从相应脚本目录启动。部分脚本使用当前目录中的预测器模块，必要时将 `accuracy/src` 加入 `PYTHONPATH`。不同策略会修改本地 Transformers 的模型链接，不要让多个策略共用一个 checkout 同时运行。

## 归档范围

提交内容是源码、脚本、实验配置、依赖与文档。`.gitignore` 排除了模型 checkpoint、partial weights、skewing 数据、语料/提示词文本、JSONL/CSV 结果、缓存、构建产物及第三方 Transformers checkout。

所有原有代码变体保留用于研究对照；本次整理不重构推理算法。原工作区中已删除的 `accuracy/lm_eval/ours.sh` 和 `accuracy/src/modeling_opt_ours.py` 未从历史版本恢复，因此部分旧论文复现脚本仍引用不存在的历史入口，详见 [代码索引与已知限制](docs/CODE_MAP.md)。

本次进行静态语法、代码清单和 Git 内容检查；没有运行需要模型、数据和 GPU 的完整评测，归档不代表所有实验已验证可复现。

## 来源与许可

本仓库基于 [InfiniGen](https://github.com/snu-comparch/infinigen) 的本地修改版本，同时包含基于 FlexGen、H2O 和 Hugging Face Transformers 的代码。上游说明与论文引用见 [InfiniGen 文档](docs/INFINIGEN.md)，归档来源见 [SOURCE.md](docs/SOURCE.md)。原版权声明和 Apache-2.0 许可保留，见 [LICENSE](LICENSE) 和 [FlexGen LICENSE](speedup/flexgen/LICENSE)。
