# Memory and throughput source tables (2026-11-03)

OPT-13B FP16; batch 4; prompt 1920; generation 128 (127 decode steps).
GPU memory is PyTorch peak allocated/reserved, in GiB. SD is sample SD; n=1 has no SD.

| Method | Run set | n | Decode tok/s mean +/- SD | Allocated GiB | Reserved GiB | Prefill s | Decode s | Total s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Neuro-Fetcher V12 | paired_v17_20261102 | 6 | 14.827 +/- 0.147 | 29.847 | 30.844 | 5.417 | 34.265 | 39.682 |
| Neuro-Fetcher V16 | paired_v17_20261102 | 6 | 15.273 +/- 0.077 | 26.068 | 27.141 | 5.438 | 33.262 | 38.700 |
| InfiniGen | memory_v9_20261029 | 1 | 9.163 (single run) | 25.518 | 26.820 | 1.816 | 55.441 | 57.257 |
| FlexGen CPU-KV | repeats_v15_20261031 | 5 | 5.589 +/- 0.200 | 27.310 | 28.488 | 2.438 | 90.991 | 93.429 |
| Full GPU | original_20261026 | 5 | 74.567 +/- 0.028 | 31.112 | 31.699 | 1.565 | 6.813 | 8.377 |
| Neuro-Fetcher V16 | independent_20261031 | 5 | 17.728 +/- 0.162 | 25.949 | 27.050 | 5.255 | 28.657 | 33.913 |
| Neuro-Fetcher V12 | independent_20261030 | 5 | 14.427 +/- 2.326 | 29.728 | 30.575 | 6.752 | 35.963 | 42.716 |
| Neuro-Fetcher original | original_20261026 | 5 | 16.541 +/- 3.977 | 38.833 | 46.650 | 6.244 | 32.617 | 38.861 |

Cross-runtime rows differ in prompt token IDs, runtime, warmup, dates and repetition protocol.
Use latest paired V16 as the primary result and retain independent V16 as corroboration.
Do not combine one run set's memory with another run set's throughput.
