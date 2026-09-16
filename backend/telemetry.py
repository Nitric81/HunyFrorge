import ctypes
import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def rss_mb():
    if sys.platform == 'win32':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong), ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / 1048576
        return None
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


class StageTelemetry:
    def __init__(self, path: Path, gpu=False):
        self.path = path
        self.gpu = gpu
        self.records = {}
        self.active = {}
        self.operation = None
        self.progress = None
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.thread = None
        self.errors = []

    def _cuda(self):
        torch = sys.modules.get('torch')
        return torch.cuda if self.gpu and torch and torch.cuda.is_initialized() else None

    def _sample(self):
        rss = rss_mb()
        cuda = self._cuda()
        allocated = cuda.memory_allocated() / 1048576 if cuda else None
        reserved = cuda.memory_reserved() / 1048576 if cuda else None
        for name, start in self.active.items():
            record = self.records[name]
            record['elapsed_seconds'] = time.perf_counter() - start
            for key, value in [('peak_rss_mb', rss), ('sampled_peak_allocated_vram_mb', allocated), ('sampled_peak_reserved_vram_mb', reserved)]:
                if value is not None:
                    record[key] = max(record.get(key) or 0, value)

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'pid': os.getpid(), 'sample_interval_seconds': 0.25, 'memory_scope': 'current process RSS and CUDA allocator; sampled per operation, allocator high-water only for top-level stages', 'operation': self.operation, 'progress': self.progress, 'stages': self.records, 'telemetry_errors': self.errors}), encoding='utf-8')
        for attempt in range(10):
            try:
                temporary.replace(self.path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.01)

    def _publish(self):
        try:
            self._flush()
        except OSError as error:
            kind = type(error).__name__
            if kind not in self.errors:
                self.errors.append(kind)
                logging.getLogger(__name__).warning('Telemetry persistence unavailable (%s); measurements may be incomplete', kind)

    def _loop(self):
        while not self.stopped.wait(0.25):
            with self.lock:
                self._sample()
                self._publish()

    def __enter__(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        if self.thread:
            self.thread.join()
        with self.lock:
            self._sample()
            self._publish()

    def update(self, operation, progress=None):
        with self.lock:
            self.operation = operation
            self.progress = progress
            self._sample()
            self._publish()

    @contextmanager
    def measure(self, name):
        cuda = self._cuda()
        if cuda:
            cuda.synchronize()
        with self.lock:
            outermost = not self.active
            if outermost and cuda:
                cuda.reset_peak_memory_stats()
            self.records[name] = {'state': 'running', 'started_at': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': 0, 'peak_rss_mb': None, 'sampled_peak_allocated_vram_mb': None, 'sampled_peak_reserved_vram_mb': None, 'allocator_peak_vram_mb': None}
            self.active[name] = time.perf_counter()
            previous_operation = self.operation
            self.operation = name
            self._sample()
            self._publish()
        state = 'complete'
        try:
            yield
        except BaseException:
            state = 'failed'
            raise
        finally:
            cuda = self._cuda()
            if cuda:
                cuda.synchronize()
            with self.lock:
                self._sample()
                if outermost and cuda:
                    self.records[name]['allocator_peak_vram_mb'] = cuda.max_memory_allocated() / 1048576
                self.records[name]['state'] = state
                del self.active[name]
                self.operation = previous_operation
                self.progress = None
                self._publish()
