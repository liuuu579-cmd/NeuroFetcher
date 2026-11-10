"""Allocator measurements for supplementary memory runs (bytes, one GPU)."""
import hashlib
import resource
from pathlib import Path


def snapshot(torch):
    torch.cuda.synchronize()
    return dict(allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def combined_memory(before, prefill, decode):
    return dict(before_generation=before, prefill=prefill, decode=decode,
                peak_allocated_bytes=max(prefill['peak_allocated_bytes'], decode['peak_allocated_bytes']),
                peak_reserved_bytes=max(prefill['peak_reserved_bytes'], decode['peak_reserved_bytes']),
                cpu_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                scope='PyTorch allocator on the sole visible GPU; includes resident weights and buffers. '
                      'Peaks reset after warmup and at the prefill/decode boundary. Reserved memory '
                      'retains warmup allocator blocks; CUDA context/non-PyTorch memory is not included. '
                      'Memory-run timing is supplementary; historical throughput is preserved.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
