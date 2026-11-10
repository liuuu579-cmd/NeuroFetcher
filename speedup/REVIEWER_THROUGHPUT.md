The independent reviewer throughput backend uses the original OPT-13B weights at
`/home/liushenyu/opt-13b` and never changes the old FlexGen symlinks. Its first
matrix isolates scheduling and KV offload: Full GPU, Full CPU offload, learned
serial selection, and learned Ping-Pong selection. InfiniGen, H2O, Quest,
previous-step, and real packed KV quantization are not implemented in this matrix.

The predictor checkpoint has one encoder layer, four heads, FFN dimension 2048,
five outputs, and FP32 parameters. The separately retained source embedding table
is FP16 and is cast to FP32 for prediction. Prediction recomputes the entire
already cached prefix at each decode step. The first five outputs select layers
1–5, and output 5 is reused for deeper layers. Selection uses ceil(ratio * history)
and preserves ties, then unions recent-20. Indices differ by sample and are shared
across attention heads, matching the learned method. The current input token has
its KV computed locally and is included in attention, separately from history.

Serial and asynchronous versions use identical prefix histories and selections.
The async worker predicts and packs the following step while the current decode
runs. Its newest history KV is carried from GPU, because that row may not yet be
published in pinned CPU memory. Carry is patched before the next attention. A
small-model check compares this path with native cached OPT at 100% history, as
well as serial/async forced-input logits and free-generation equality.

All methods use Full causal SDPA prefill. OPT Q projections are unscaled before
SDPA, whose default applies 1/sqrt(head_dim) exactly once. CPU KV home and staging
are real pinned FP16 storage; DMA moves actual gathered KV. CPU packing,
predictor computation, selections, copies and pipeline startup/drain are included
in decode wall time. Allocating per-run home and staging buffers is excluded and
must be described as setup, while prefill includes populating CPU home.

Physical GPU staging is allocated for every layer at full history capacity to
handle threshold ties safely. Serial uses one such slot; async uses two. This
is intentionally reported in bytes and is **not a memory-minimal implementation**.
A sparse selection budget does not imply sparse GPU allocation in this version.
Results from a later compact-buffer backend must be labeled as a different version.

Prefill generates the first output token. Decode processes gen_len-1 input
tokens and produces the remaining outputs, so decode throughput is
batch*(gen_len-1)/decode_s. End-to-end throughput is batch*gen_len/total_s.
Generation has fixed length and does not stop at EOS. Each method uses one full
warmup and five measured repetitions. Individual runs and standard deviations
are retained, along with peak GPU allocation/reservation, process high-water RSS,
physical pinned buffers, actual selection ratio, and transferred KV bytes.
GPU event intervals during co-execution are stream elapsed times, not isolated
kernel service times. Selection-wait intervals include host starvation before
the stream receives its dependency; they are not pure DMA wait.

Start a single-GPU sequential matrix only when an authorized GPU is idle:

```bash
cd /home/liushenyu/infinigen
conda run -n infinigen python -u speedup/run_reviewer_throughput_suite.py \
  --gpu 2 --cpu-cores 20-29 \
  --output-dir /home/liushenyu/infinigen/speedup/results/reviewer_throughput_RUN_ID
```

The queue first reruns tiny and OPT-13B correctness gates. It checks GPU memory
before every job and stops if the card becomes busy. It never kills jobs or
overwrites existing result directories. Contexts 512/1024/1536/1920, generation
128 and batch 4 are the first matrix. Other batch sizes, algorithmic baselines,
isolated predictor/transfer profiling and a CUDA overlap trace remain separate
required follow-up work before writing the complete reviewer response.
