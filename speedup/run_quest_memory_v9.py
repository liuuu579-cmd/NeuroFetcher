#!/usr/bin/env python3
"""Add phase allocator counters to v8 Quest in memory; preserve its algorithm."""
from pathlib import Path
from reviewer_memory_v9 import snapshot, combined_memory, digest

SOURCE = Path(__file__).resolve().with_name('run_quest_native_cached_v8.py')


def instrument(source):
    replacements = [
        ('    start = time.perf_counter()\n',
         '    memory_before = snapshot(torch)\n'
         '    torch.cuda.reset_peak_memory_stats()\n'
         '    start = time.perf_counter()\n'),
        ('        decode_start = time.perf_counter()\n',
         '        memory_prefill = snapshot(torch)\n'
         '        torch.cuda.reset_peak_memory_stats()\n'
         '        decode_start = time.perf_counter()\n'),
        ('    total_s = time.perf_counter() - start\n',
         '    total_s = time.perf_counter() - start\n'
         '    memory_decode = snapshot(torch)\n'),
        ('        "complete": True,\n',
         '        "complete": True,\n'
         '        "memory": combined_memory(memory_before, memory_prefill, memory_decode),\n'
         '        "memory_source_sha256": {str(SOURCE): digest(SOURCE), str(MEMORY_WRAPPER): digest(MEMORY_WRAPPER)},\n'
         '        "memory_note": "One full warmup and one measured run; accepted v8 algorithm unchanged.",\n'),
        ('    args.output.write_text(json.dumps(result, indent=2) + "\\n")\n',
         '    with args.output.open("x") as stream:\n'
         '        stream.write(json.dumps(result, indent=2) + "\\n")\n'),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError('v8 source changed; expected one instrumentation site: ' + repr(old))
        source = source.replace(old, new)
    return source


if __name__ == '__main__':
    context = dict(__name__='__main__', __file__=str(SOURCE), SOURCE=SOURCE,
                   MEMORY_WRAPPER=Path(__file__).resolve(), snapshot=snapshot,
                   combined_memory=combined_memory, digest=digest)
    exec(compile(instrument(SOURCE.read_text()), str(SOURCE), 'exec'), context)
