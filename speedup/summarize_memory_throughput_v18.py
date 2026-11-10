#!/usr/bin/env python3
"""Read existing results and create a new, CPU-only paper-summary directory.

Only the new output directory is written. Existing outputs are never replaced.
No torch import, CUDA initialization, model loading, or benchmark is performed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
GIB = 2 ** 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory: {}".format(output))
    sources = {}

    def read(relative):
        path = ROOT / relative
        raw = path.read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        assert data.get("complete", True), relative
        return data

    paired_dir = "speedup/results/layerwise_paired_v17_20261102"
    paired = read(paired_dir + "/summary.json")
    assert paired["complete"] and paired["all_validation_bitwise_equal"]
    assert paired["number_of_pairs_layer_faster"] == 6
    cfg = paired["manifest"]
    assert (cfg["batch"], cfg["prompt_len"], cfg["gen_len"]) == (4, 1920, 128)
    pair_runs = {"bulk_v12": [], "layer_v16": []}
    reference_validation = None
    for path in sorted((ROOT / paired_dir).glob("pair*.json")):
        data = read(str(path.relative_to(ROOT)))
        if reference_validation is None:
            reference_validation = data["validation"]
        assert data["validation"] == reference_validation, str(path)
        pair_runs[data["backend"]].append(data["result"])
    assert all(len(runs) == 6 for runs in pair_runs.values())

    rows = []

    def aggregate(method, run_set, runs, files, protocol):
        metrics = ("decode_tokens_per_s", "prefill_s", "decode_s", "total_s",
                   "peak_allocated_bytes", "peak_reserved_bytes")
        row = {"method": method, "run_set": run_set, "n": len(runs),
               "protocol": protocol, "sources": files}
        for key in metrics:
            values = [r[key] for r in runs]
            row[key + "_mean"] = mean(values)
            row[key + "_sd"] = stdev(values) if len(values) > 1 else None
        row["peak_allocated_gib_mean"] = row["peak_allocated_bytes_mean"] / GIB
        row["peak_reserved_gib_mean"] = row["peak_reserved_bytes_mean"] / GIB
        rows.append(row)
        return row

    for backend, method in (("bulk_v12", "Neuro-Fetcher V12"),
                            ("layer_v16", "Neuro-Fetcher V16")):
        aggregate(method, "paired_v17_20261102", pair_runs[backend],
                  [p for p in sources if p.endswith(backend + ".json")],
                  "one common-model process; one full warmup/backend; six balanced adjacent pairs")
        for key, entry in paired["summary"][backend].items():
            if key + "_mean" in rows[-1]:
                assert abs(rows[-1][key + "_mean"] - entry["mean"]) < 1e-5

    inf_path = "speedup/results/reviewer_memory_v9_20261029/infinigen/infinigen_p1920_g128_b4.json"
    inf = read(inf_path)
    aggregate("InfiniGen", "memory_v9_20261029", [{**inf, **inf["memory"]}],
              [inf_path], "one measured generation; one-output-token native warmup")
    flex_dir = "speedup/results/flexgen_repeats_v15_20261031"
    flex_paths = [str(p.relative_to(ROOT)) for p in sorted(
        (ROOT / flex_dir).glob("flexgen_p1920_g128_b4_repeat*.json"))]
    assert len(flex_paths) == 5
    flex = [read(p) for p in flex_paths]
    aggregate("FlexGen CPU-KV", "repeats_v15_20261031",
              [{**d, **d["memory"]} for d in flex], flex_paths,
              "five fresh processes; one-output-token native warmup/process")

    run_files = [
        ("Full GPU", "original_20261026", "speedup/results/reviewer_throughput_20261026/p1920-g128-b4-full_gpu.json"),
        ("Neuro-Fetcher V16", "independent_20261031", "speedup/results/layerwise_v16_20261031/p1920-g128-b4-neural_async.json"),
        ("Neuro-Fetcher V12", "independent_20261030", "speedup/results/compact_staging_v11_20261029/benchmark_v13/p1920-g128-b4-neural_async.json"),
        ("Neuro-Fetcher original", "original_20261026", "speedup/results/reviewer_throughput_20261026/p1920-g128-b4-neural_async.json"),
    ]
    run_documents = {}
    for method, run_set, path in run_files:
        document = read(path)
        run_documents[run_set + ":" + method] = document
        aggregate(method, run_set, document["runs"], [path],
                  "one full warmup; five measurements in one process")
    independent = run_documents["independent_20261031:Neuro-Fetcher V16"]

    # Explicit selection avoids mixing historical and current measurements.
    primary = [next(r for r in rows if r["method"] == method and r["run_set"] == run_set)
               for method, run_set in [
                   ("InfiniGen", "memory_v9_20261029"),
                   ("FlexGen CPU-KV", "repeats_v15_20261031"),
                   ("Neuro-Fetcher V16", "paired_v17_20261102"),
                   ("Full GPU", "original_20261026")]]
    current = primary[2]
    comparisons = []
    for reference in (primary[0], primary[1], primary[3], rows[0]):
        comparisons.append({
            "reference_method": reference["method"],
            "reference_run_set": reference["run_set"],
            "v16_decode_throughput_ratio": current["decode_tokens_per_s_mean"] / reference["decode_tokens_per_s_mean"],
            "v16_allocated_delta_gib": current["peak_allocated_gib_mean"] - reference["peak_allocated_gib_mean"],
            "v16_allocated_delta_percent": (current["peak_allocated_bytes_mean"] / reference["peak_allocated_bytes_mean"] - 1) * 100,
            "reference_total_time_divided_by_v16": reference["total_s_mean"] / current["total_s_mean"],
            "comparison_type": "controlled staging implementation comparison" if reference is rows[0] else "cross-runtime/date observation; not an isolated causal speedup",
        })

    components_path = "speedup/results/reviewer_components_20261026/components.json"
    components = read(components_path)
    costs = [{"context": r["context"],
              "predictor_gpu_ms": r["isolated"]["predictor_gpu_ms"]["mean"],
              "cpu_packing_ms": 1000 * r["isolated"]["packing_host_s"]["mean"],
              "h2d_stream_ms": r["isolated"]["h2d_stream_ms"]["mean"]}
             for r in components["results"]]
    original_inf_path = "speedup/results/native_flexgen_v8_20261027-v8a/infinigen/infinigen_p1920_g128_b4.json"
    old_inf = read(original_inf_path)
    facts = {
        "date": "2026-11-03", "workload": cfg, "records": rows,
        "comparisons": comparisons, "historical_isolated_costs": costs,
        "historical_infinigen_decode_tokens_per_s": old_inf["decode_tokens_per_s"],
        "memory_breakdown": {key: independent[key] for key in (
            "model_tensor_bytes", "predictor_tensor_bytes", "predictor_embedding_bytes",
            "predictor_embedding_additional_bytes", "embedding_shared_with_model")},
        "layerwise_buffers_bytes": {key: pair_runs["layer_v16"][0][key] for key in (
            "cpu_kv_home_bytes", "pinned_staging_bytes", "gpu_staging_bytes")},
        "independent_v16_nvml_sampled_peak_bytes": independent["nvml_memory"]["sampled_device_peak_bytes"],
        "definitions": {
            "GiB": "2**30 bytes", "decode_tokens": 508,
            "decode_throughput": "4 * 127 / decode_s; reported mean is mean of per-run throughputs",
            "total_s": "prefill_s + decode_s; excludes checkpoint loading, warmup and separately logged request setup",
            "peak_allocated": "PyTorch reported max of prefill/decode tensor allocation peaks; includes weights, predictor and workspaces; not NVML whole-device use",
        },
    }
    for relative, expected in sources.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    output.mkdir(parents=True, exist_ok=False)
    (output / "facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False) + "\n")
    provenance = {"source_sha256": sources,
                  "generator": str(Path(__file__).relative_to(ROOT)),
                  "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "no_gpu_work_performed": True}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    fields = [k for k in rows[0] if k != "sources"] + ["sources"]
    with (output / "all_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "sources": " | ".join(row["sources"])})
    with (output / "comparisons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    lines = ["# Memory and throughput source tables (2026-11-03)", "",
             "OPT-13B FP16; batch 4; prompt 1920; generation 128 (127 decode steps).",
             "GPU memory is PyTorch peak allocated/reserved, in GiB. SD is sample SD; n=1 has no SD.", "",
             "| Method | Run set | n | Decode tok/s mean +/- SD | Allocated GiB | Reserved GiB | Prefill s | Decode s | Total s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        sd = row["decode_tokens_per_s_sd"]
        throughput = "{:.3f} +/- {:.3f}".format(row["decode_tokens_per_s_mean"], sd) if sd is not None else "{:.3f} (single run)".format(row["decode_tokens_per_s_mean"])
        lines.append("| {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            row["method"], row["run_set"], row["n"], throughput,
            row["peak_allocated_gib_mean"], row["peak_reserved_gib_mean"],
            row["prefill_s_mean"], row["decode_s_mean"], row["total_s_mean"]))
    lines += ["", "Cross-runtime rows differ in prompt token IDs, runtime, warmup, dates and repetition protocol.",
              "Use latest paired V16 as the primary result and retain independent V16 as corroboration.",
              "Do not combine one run set's memory with another run set's throughput.", ""]
    (output / "TABLES.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "records": len(rows),
                      "hashed_sources": len(sources), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
