# 新增长文本原生基线

仅新增文件，不改原模型源码、符号链接、入口或已有结果。使用 `infinigen` 环境的绝对 Python 路径。

新增 InfiniGen、H2O、4-bit KV 量化，分别跑同一组256条 QA 和填空输入。

| 方法 | 模型 / 原生实现 | 设置 |
|---|---|---|
| InfiniGen | `../setup/opt-model/opt-13b` / `../src/modeling_opt_inf.py` | 配套0.3 partial weights、alpha=7、capacity=1、budget=0.3、前两层Full |
| H2O | `/home/liushenyu/opt-13b` / `utils_lm_eval/modify_opt.py` | heavy=0.15、recent=0.15、前两层Full |
| KV量化 | `/home/liushenyu/opt-13b` / `../src/modeling_opt_orig.py` | 4-bit、group=64、前两层Full、其余层KV量化/反量化 |

InfiniGen和H2O保留整段长度容量、原生预热和recent边界规则，分别报告全位置加权密度和末位置密度，不能标成严格逐前缀30%。不额外添加recent-20。量化保留所有token，实际仍使用FP16存储；不据此报告实测显存/传输或速度改善。

新增加载器在启用方法前核对每种提示、任务、长度组合的首条Full结果。历史控制使用FP16 softmax，原生方法上转FP32；InfiniGen还有FP16权重变换。校验记录候选log-prob最大差和真实答案rank，容差分别为0.1（H2O/量化）和0.25（InfiniGen），并要求rank相同。该检查不是逐位等价证明。InfiniGen checkpoint残留的80个Q/K bias键是变换中已折叠的原生结构差异，其他不匹配一律拒绝。

## 运行

```bash
cd /home/liushenyu/infinigen/accuracy/lm_eval
# 当前空闲卡验证：每种提示8条，覆盖全部任务和长度
conda run -n infinigen bash run_long_context_native_baselines.sh --smoke
# 完整运行并审计汇总
conda run -n infinigen bash run_long_context_native_baselines.sh
# 等待空闲卡，smoke全部通过后自动完整运行
conda run -n infinigen bash queue_long_context_native_baselines.sh
```

队列只选择GPU 1–3中没有计算进程的卡，避开已有GPU 0任务。每个worker固定10个CPU核；最多30核。GPU占用变化时不结束其他进程。目录锁避免重复运行；已有完整且哈希一致的结果跳过，未完成输出拒绝覆盖。

输出：`results/long_context_native_baselines_20261018/`。独立smoke目录后缀为`_smoke`。完成后生成包括原有七种方法的`summary_zh.md`、`summary.csv`、`by_task_length.csv`和`validated_summary.json`。

## 2026-10-18验证状态

- Python/Shell语法检查通过。
- CPU小模型H2O和量化前向通过；InfiniGen原生mask因果性、预热、末步容量、密度检查通过（CPU FP32）。
- 原有QA/填空共14组结果独立审计通过。
- 早期GPU试跑被过严的数值/加载审计提前拦住；已核对原生FP32 softmax、InfiniGen折叠bias和Q/K变换，调整新增审计规则。
- 修正后的完整GPU smoke尚未通过；19:54检查时GPU 0–3均有他人进程，队列PID 3644846处于等待状态。不代表新增方法已产出实验结果。
- 队列后续若smoke失败会退出，保留详细日志，不启动全量；全量推理成功后才调用独立汇总审计。

本任务为OPT-13B的2K内、整段所有位置应用方法的检索压力测试，不是标准LongBench、Full prefill + sparse decode或系统速度测试。
