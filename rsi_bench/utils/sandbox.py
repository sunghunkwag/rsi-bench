"""Sandbox execution environment for RSI-Bench."""
import multiprocessing
import traceback
import time
from typing import Any, Dict, Optional


class SandboxResult:
      """Result of a sandboxed execution."""

    def __init__(self, value=None, error=None, duration=0.0, timed_out=False):
              self.value = value
              self.error = error
              self.duration = duration
              self.timed_out = timed_out
              self.success = error is None and not timed_out


def _run_in_process(fn, args, kwargs, result_queue):
      try:
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                duration = time.perf_counter() - start
                result_queue.put(SandboxResult(value=result, duration=duration))
except Exception as e:
        duration = time.perf_counter() - start
        result_queue.put(SandboxResult(error=str(e), duration=duration))


class Sandbox:
      """Execute untrusted code with timeout and resource limits."""

    def __init__(self, timeout: float = 30.0, max_memory_mb: int = 512):
              self.timeout = timeout
              self.max_memory_mb = max_memory_mb
              self._executions = 0
              self._failures = 0

    def run(self, fn, *args, **kwargs) -> SandboxResult:
              self._executions += 1
              result_queue = multiprocessing.Queue()
              proc = multiprocessing.Process(
                  target=_run_in_process, args=(fn, args, kwargs, result_queue)
              )
              proc.start()
              proc.join(timeout=self.timeout)
              if proc.is_alive():
                            proc.terminate()
                            proc.join(timeout=5)
                            self._failures += 1
                            return SandboxResult(timed_out=True, duration=self.timeout)
                        if not result_queue.empty():
                                      result = result_queue.get_nowait()
                                      if not result.success:
                                                        self._failures += 1
                                                    return result
        self._failures += 1
        return SandboxResult(error="No result returned")

    def run_code(self, code: str, namespace: Optional[Dict] = None) -> SandboxResult:
              ns = namespace or {}
        def _exec():
                      exec(code, ns)
            return ns.get("result", None)
        return self.run(_exec)

    @property
    def stats(self) -> Dict[str, Any]:
              return {
                            "total": self._executions,
                            "failures": self._failures,
                            "success_rate": 1 - self._failures / max(self._executions, 1),
              }
