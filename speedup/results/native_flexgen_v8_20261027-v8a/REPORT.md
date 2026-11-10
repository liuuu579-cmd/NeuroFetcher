# Reviewer Throughput Comparison

All conditions use OPT-13B FP16, batch 4, 128 generated tokens, and prompt
lengths 512/1024/1536/1920. The throughput columns count the 127 decode tokens;
end-to-end latency includes prefill and all 128 output tokens.

## Results

Decode throughput is in output tokens/s; total is end-to-end seconds.
FlexGen/Quest rows are single measured runs after one warmup. The Neural and
Full rows are means from the earlier five-run matched-workload experiment in
`reviewer_throughput_20261026/summary.csv`.

| Prompt | Full GPU | Neural serial | Neural async | InfiniGen | H2O | Quest native | Quest capped |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 107.84 / 5.10 | 34.95 / 14.97 | 47.82 / 11.62 | 23.57 / 22.08 | 37.00 / 14.12 | 78.79 / 6.83 | 75.31 / 7.12 |
| 1024 | 93.97 / 6.13 | 23.13 / 24.25 | 22.93 / 29.59 | 14.10 / 37.00 | 28.39 / 18.66 | 68.76 / 8.16 | 65.86 / 8.49 |
| 1536 | 81.68 / 7.39 | 16.95 / 33.45 | 16.96 / 37.75 | 9.85 / 53.04 | 21.04 / 25.49 | 57.39 / 10.16 | 55.13 / 10.54 |
| 1920 | 74.57 / 8.38 | 14.24 / 40.16 | 16.54 / 38.86 | 7.23 / 72.12 | 13.75 / 38.75 | 50.68 / 11.79 | 48.97 / 12.17 |

Each cell is `decode tok/s / total seconds`. The Full CPU-offload reference from
the earlier experiment was 22.14 / 23.37, 12.90 / 41.69, 9.23 / 58.55, and
7.57 / 71.58 at the four prompt lengths.

## Runtime Notes

- InfiniGen and H2O use their separate FlexGen frontends, full OPT weights on
  GPU, and KV home on CPU. InfiniGen uses alpha 7 and partial-weight ratio 0.3.
  H2O uses heavy ratio 0.15, its native recent-token handling, and full attention
  in the first two layers.
- H2O's original cache-read path requested one more decode row than it copied.
  The new v8 launcher fixes only that cache-length adapter while retaining the
  H2O selection/update implementation. H2O 1024 first measured 87.21 seconds
  while the other lane ran concurrently; an isolated repeat was 18.66 seconds.
  The table uses the repeat and both raw records are retained.
- Quest native and capped use page size 16 and Quest's sign-aware page bounds.
  Native uses the minimum-three-page rule and full first two layers. Capped
  selects `ceil(0.30 * visible_length)` tokens with no recent union and no
  first-two-layer exception. Measured capped selection density is 30.0% across
  lengths; native density is 30.9%-32.8%.
- Quest runs through a Transformers OPT cached-decode adaptation with full KV
  resident on GPU. It is not the official Quest paging runtime and does not
  measure CPU-to-GPU page-transfer savings. Its latency should not be ranked as
  a matched-runtime result against the CPU-KV-offload FlexGen rows.
- Neural serial/async and Full references come from the five-run custom
  offload-backend experiment. They are included as the project reference, with
  their separate runtime retained in the earlier report.

Raw per-condition JSON and logs are stored beside this report. No existing
source or result file was overwritten.
