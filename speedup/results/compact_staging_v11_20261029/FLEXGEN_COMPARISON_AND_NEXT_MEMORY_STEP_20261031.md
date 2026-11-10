# FlexGen throughput and further memory reduction

## Matched workload at 1920 prompt tokens

OPT-13B FP16, batch 4, 128 generated tokens, measuring 127 decode steps:

| Method | Decode throughput | Peak PyTorch allocated | Peak PyTorch reserved |
|---|---:|---:|---:|
| Native FlexGen CPU-KV | 5.59 ± 0.20 tok/s | 27.310 GiB | 28.488 GiB |
| Neuro-Fetcher async, compact exact staging | 14.43 ± 2.33 tok/s | 29.728 GiB | 30.575 GiB |

FlexGen used five independent fresh-process runs, each with one generated-token warmup and one measured 128-token generation. Neuro-Fetcher values are the existing five measured repetitions after one full warmup. The ratio of means is 2.58x. This comparison uses identical model, prompt length, batch and output length. It should be reported as a measured throughput comparison, with the differing warmup/repetition harness disclosed.

The earlier single FlexGen memory run was 5.93 tok/s; the new five-run mean is 5.59 tok/s. The five raw runs and summary are in `speedup/results/flexgen_repeats_v15_20261031`.

Given the repeat comparison, 29.73 GiB is defensible for the current goal of speeding up CPU KV offload: it is about 2.42 GiB above FlexGen but delivers 2.58x its measured decode throughput. It is below the 31.112 GiB allocated peak of the full-GPU reference, though full-GPU decoding remains much faster. Do not claim the proposed method is the fastest when all KV fits on GPU.

## FP16 predictor test

A same-prompt, full 128-token experiment changed the predictor weights and computations from FP32 to FP16 while keeping the OPT model, KV format, budget, and scheduling fixed. Predictor parameter storage fell from 0.469 GiB to 0.235 GiB; total peak allocated fell from 29.728 GiB to 29.033 GiB. However, selected indices differed on 126 of 128 decode steps, generated sequences changed for two of four batch rows, and logits were not bitwise equal. This is not an equivalent implementation and must not replace the accepted FP32 predictor without a separate accuracy/perplexity evaluation. Raw records are in `speedup/results/compact_staging_v11_20261029/predictor_half_v15/opt13b_p1920_g128.json`.

## Potential exact optimization

The optimized async implementation still reserves two GPU staging buffers, each shaped for all 40 layers and up to 635 KV tokens per layer. Together they occupy 3.876 GiB. A layer-streamed pipeline could keep two one-layer buffers and move the selected KV for the next layer while the current layer computes. At the same capacity, those buffers would occupy about 0.097 GiB, a theoretical reduction of roughly 3.78 GiB. Applied to the current peak, a first-order estimate is around 26 GiB.

This is an estimate, not a result. It requires changing the staging schedule and CUDA-event dependencies. Per-layer copies may introduce stalls, so throughput could fall. Any implementation should retain FP32 predictor inference and exact indices, pass output/logit parity, and be compared against both the current 14.43 tok/s result and the 5.59 tok/s FlexGen mean before being adopted.

## Recommendation

The current compact async version already has a substantial measured throughput advantage over FlexGen at a modestly higher PyTorch allocated peak. That is a solid result to use now. Layer-streamed staging is the next experiment only if lowering the GPU memory peak by another few GiB is important enough to justify implementation and performance validation. No existing source or result was modified for these measurements.
