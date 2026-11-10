#!/usr/bin/env python3
"""Versioned launcher for the isolated native FlexGen throughput runner.

This wrapper keeps prior runner versions unchanged while correcting the
FlexGen model-root convention: frontends append ``opt-13b-np`` themselves.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "speedup/run_native_flexgen_benchmark.py"
WEIGHT_ROOT = Path("/home/liushenyu/opt-13b-flexgen-np")


def main():
    spec = importlib.util.spec_from_file_location("native_flexgen_benchmark_v7", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    weight_dir = WEIGHT_ROOT / "opt-13b-np"
    required = (
        weight_dir / "decoder.embed_positions.weight",
        weight_dir / "decoder.layers.0.self_attn.q_proj.weight",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing native FlexGen OPT-13B weights: " + ", ".join(missing))

    module.MODEL_WEIGHTS = WEIGHT_ROOT
    module.INFINIGEN_SOURCE = WEIGHT_ROOT
    original_loader = module.load_module

    def load_native_module(name, path):
        native = original_loader(name, path)
        if name.startswith("flexgen.flex_opt_review_"):
            if name.endswith("h2o"):
                original_load_cache = native.SelfAttention.load_cache

                def load_cache_with_decode_rows(self, cache_home, cache_read_buf, i):
                    if (i == 0 or self.policy.compress_cache or self.policy.cpu_cache_compute
                            or self.policy.hh_all):
                        return original_load_cache(self, cache_home, cache_read_buf, i)
                    k_home, v_home, acc = cache_home.val
                    length = min(self.hh_k * 2 - 1, self.task.prompt_len) + 1 + i
                    self.prefetch_kv.data[0, :length].copy_(k_home.data[:length])
                    self.prefetch_kv.data[1, :length].copy_(v_home.data[:length])
                    shape = (length, k_home.shape[1], k_home.shape[2])
                    k_copy = native.TorchTensor(shape, k_home.dtype,
                                                self.prefetch_kv.data[0, :length], k_home.device)
                    v_copy = native.TorchTensor(shape, v_home.dtype,
                                                self.prefetch_kv.data[1, :length], v_home.device)
                    indices = (slice(0, length), slice(0, k_home.shape[1]))
                    dst = self.attention_compute
                    cache_read_buf.store((
                        k_copy.smart_copy(dst, indices),
                        v_copy.smart_copy(dst, indices),
                        acc.smart_copy(dst, indices),
                    ))

                native.SelfAttention.load_cache = load_cache_with_decode_rows
            original_run = native.run_flexgen

            def run_with_context_capacity(args):
                args.max_num_kv = max(args.max_num_kv, args.prompt_len + args.gen_len + 1)
                return original_run(args)

            native.run_flexgen = run_with_context_capacity
        return native

    module.load_module = load_native_module
    timer_module = importlib.import_module("flexgen.timer")
    utils_module = importlib.import_module("flexgen.utils")
    if not hasattr(utils_module, "timers"):
        utils_module.timers = timer_module.timers
    module.main()
    if len(sys.argv) > 1 and "--method" in sys.argv and sys.argv[sys.argv.index("--method") + 1] == "h2o":
        output_path = Path(sys.argv[sys.argv.index("--output") + 1])
        if output_path.is_file():
            import json
            result = json.loads(output_path.read_text())
            result["runtime"] = "H2O FlexGen runtime with decode-cache length adapter"
            output_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
