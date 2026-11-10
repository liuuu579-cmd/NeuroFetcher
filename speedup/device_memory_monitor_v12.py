"""Read-only NVML device memory samples; no CUDA calls or GPU allocations."""
import ctypes
import threading
import time


class MemoryInfo(ctypes.Structure):
    _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong),
                ('used', ctypes.c_ulonglong)]


class DeviceMemoryMonitor:
    def __init__(self, index):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        self.check(self.lib.nvmlInit_v2())
        self.handle = ctypes.c_void_p()
        self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(self.handle)))
        self.stop = threading.Event()
        self.samples = []
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f'NVML status {status}')

    def run(self):
        while not self.stop.is_set():
            info = MemoryInfo()
            status = self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info))
            if status == 0:
                self.samples.append([time.monotonic() - self.started, info.used])
            self.stop.wait(.1)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=1)
        self.check(self.lib.nvmlShutdown())
        return dict(sample_interval_s=.1, sampled_device_peak_bytes=max(x[1] for x in self.samples),
                    samples=self.samples,
                    scope='NVML whole-device used memory, sampled from model loading through warmup and measured runs; includes CUDA context/non-PyTorch allocations. Not an exact instantaneous peak. GPU exclusively assigned; no other process modified.')
