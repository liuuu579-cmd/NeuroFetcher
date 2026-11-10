# Reviewer Throughput Experiment

This first matched-backend matrix measures Full GPU, Full CPU offload, learned
serial offload, and learned Ping-Pong offload on original OPT-13B FP16. All
methods use the same causal SDPA prefill, prompt lengths 512/1024/1536/1920,
generation length 128, batch 4, one warmup, and five measured runs. Only GPU 2
and CPU cores 20-29 were used. Results are decode token/s mean ± sample SD;
decode excludes prefill.

| Prompt tokens | Full GPU | Full CPU offload | Learned serial | Learned Ping-Pong | Async / serial |
|---:|---:|---:|---:|---:|---:|
| 512 | 107.84 ± 0.06 | 22.14 ± 0.01 | 34.95 ± 0.05 | 47.82 ± 9.61 | 1.368x |
| 1024 | 93.97 ± 0.04 | 12.90 ± 0.00 | 23.13 ± 0.02 | 22.93 ± 10.52 | 0.992x |
| 1536 | 81.68 ± 0.03 | 9.23 ± 0.00 | 16.95 ± 0.08 | 16.96 ± 5.29 | 1.001x |
| 1920 | 74.57 ± 0.03 | 7.57 ± 0.00 | 14.24 ± 0.02 | 16.54 ± 3.98 | 1.162x |

Measured mean end-to-end seconds (including prefill):

| Prompt tokens | Full GPU | Full CPU offload | Learned serial | Learned Ping-Pong |
|---:|---:|---:|---:|---:|
| 512 | 5.10 | 23.37 | 14.97 | 11.62 |
| 1024 | 6.13 | 41.69 | 24.25 | 29.59 |
| 1536 | 7.39 | 58.55 | 33.45 | 37.75 |
| 1920 | 8.38 | 71.58 | 40.16 | 38.86 |

Actual historical KV selection was 30.04-30.12% for learned methods, including
the native recent-20 union. Full methods selected 100%. The 512 run selected
30.12%. At batch 4 and prompt 512, each learned run allocated 2.0 GB pinned CPU
KV home and 2.0 GB pinned CPU staging; GPU staging was 2.0 GB serial and 4.0 GB
Ping-Pong. At prompt 1920, the full-history staging capacity was roughly 6 GB
per slot. This backend therefore transfers a sparse subset but reserves slots
for the maximum history to handle threshold ties. Peak GPU allocation at prompt
1920 averaged 31.1 GiB Full GPU, 32.6 GiB learned serial and 38.8 GiB Ping-Pong.

Correctness gates passed before the matrix: a 7-layer test model matched native
cached OPT within 0.00074 maximum absolute logit error for Full offload and
all-history async; serial and async logits were bit-identical under forced
inputs and free greedy outputs matched. Original OPT-13B, batch 4, prompt 512,
and 8 generated tokens also passed; serial/async logits were identical. Full
GPU vs Full offload at 8 generated tokens differed by at most 0.0743 absolute
logit, consistent with FP16 SDPA numerical differences. The OPT13B smoke used a
different prompt length from the throughput conditions and was a correctness
gate, not a timed repetition.

The asynchronous speedup is not consistent. At prompt 512 it averaged 1.368x,
but its sample SD was 9.61 token/s and one repetition was much slower than the
other four. At 1024 and 1536 it was effectively tied with serial, with much
larger run-to-run variation. At 1920 it averaged 1.162x with 3.98 token/s SD.
This supports a conditional overlap benefit, not a zero-overhead claim. Full
GPU remains much faster in this single-GPU experiment because it does not
transfer the KV cache from CPU.

An isolated, fixed-prefix component profile found predictor GPU time increasing
from 15.9 ms at context 512 to 34.6, 70.0 and 82.1 ms at 1024, 1536 and 1920.
H2D event time increased from 22.5 to 42.6, 62.5 and 77.5 ms. CPU packing rose
from 27 ms to 243 ms. These isolated costs do not add directly to a decode step:
the pipeline overlaps them and stages only selected rows. In the timed
co-execution runs, predictor/DMA event intervals and LLM compute intervals are
retained per step in each JSON result. Those stream intervals can include
contention and must not be interpreted as isolated service time.

CUDA profiler traces were requested, but PyTorch reported
`CUPTI_ERROR_NOT_INITIALIZED`; device activities are missing. The generated
traces can be used to inspect CPU submission flow only. They do not verify GPU
overlap. A privileged Nsight/CUPTI trace is still required to make that claim.

This experiment does **not** yet compare InfiniGen, H2O, Quest, previous-step
reuse, layer-5 reuse, query-aware selection, or actual packed KV quantization.
The throughput method conditions here isolate backend and scheduling. Oracle
requires future attention scores and is a quality upper bound, not a deployable
throughput baseline unless the oracle scoring work is explicitly charged. Real
quantization must include packing, transfer, and unpacking costs; prior fake
quantized accuracy controls are not speed results.

Artifacts: `summary.csv`, per-condition `p*-g*-b*.json`, queue logs,
`opt13b_correctness/smoke_metrics.json`, and the component microbenchmark and
trace files in `../reviewer_components_20261026/`. `suite.exitcode` contains 0.
