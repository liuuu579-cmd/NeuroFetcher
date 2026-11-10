# Reviewer Baseline Throughput Comparison

## Run allocation

Each process is bound to one GPU. The GPUs are not combined for a single model:

| Lane | GPU | Methods, sequential within lane | CPU affinity |
|---|---:|---|---|
| A | 2 | InfiniGen native, then Neuro-Fetcher reference | cores 20-29 |
| B | 3 | H2O native, Quest native, Quest capped, then Neuro-Fetcher reference | cores 30-39 |

The lanes may run concurrently after method-specific adapters pass their
correctness smoke tests. If either GPU becomes busy before a new process starts,
that lane waits or stops; it never terminates another user's process. Each
experiment loads a full independent OPT-13B replica on its own GPU.

## Shared workload and reporting

- OPT-13B, FP16 weights and KV, batch 4, prompt lengths 512/1024/1536/1920,
  fixed 128 generated tokens, greedy decode, one full warmup and five measured
  repeats.
- Same prompt token IDs, generation policy, tokenizer, Full causal prefill,
  CPU/GPU thread limits, timing boundaries, and output validation within each
  paired lane. Add batch 1 after batch 4 to report single-request latency.
- Report prefill, decode, and total latency separately; decode and end-to-end
  tokens/s; run-level mean and standard deviation; peak GPU allocation and
  reservation, process peak RSS, pinned CPU/GPU buffer sizes, selected-cache
  density, and measured H2D bytes.
- Keep each method's original selection and scheduling semantics. Do not add a
  recent-20 union to InfiniGen, H2O, or Quest. Neuro-Fetcher uses its historical
  recent-20 union and the learned 30% threshold rule. Report actual density and
  actual bytes, since nominal 30% has different meanings across algorithms.
- Quest is reported both with its native page-size/minimum-page rule and with
  the previously agreed token-budget-capped adaptation. Label these separately.

## Algorithm-specific correctness gates

- InfiniGen uses the existing OPT-13B transformed checkpoint and partial
  weights, alpha 7, capacity 1.0, prefetch budget 0.3, partial weight ratio
  0.3, and the original two Full layers. Validate loaded transformed weights
  and a fixed-input next-token/logit comparison before timing.
- H2O uses the existing native heavy-hitter accumulation and heavy=0.15 plus
  recent=0.15 configuration, preserving its own recent boundary and the
  existing two Full layers. Confirm cache counts and selected indices against
  the accuracy implementation for a deterministic short decode.
- Quest uses page size 16, the official minimum three pages and first two Full
  layers for the native condition. The capped condition enforces no more than
  ceil(0.30 * visible history) selected tokens and has no added recent tokens.
  Validate page bounds and selected indices against the existing strict
  controls implementation.
- Neuro-Fetcher reference uses the already tested online-prefix predictor,
  trained five-output checkpoint, native recent-20 union, and 30% threshold.

These algorithms cannot all be made to use one identical selection call: H2O
needs accumulated attention, Quest needs current-query page scoring, and
InfiniGen predicts a later query from transformed partial weights. Their
selection logic must remain native. The comparable system contract is the same
model, precision, workload, prefill, timing/measurement protocol, and reported
memory/transfer costs. Any method whose native runtime cannot satisfy that
contract must be labeled as a runtime-specific result, not presented as a
matched-backend winner.

The first 2026-10-26 experiment established the Neuro-Fetcher/Full reference
protocol, but it did not measure these three baselines. No baseline throughput
result should be reported until its adapter and correctness gate are complete.
