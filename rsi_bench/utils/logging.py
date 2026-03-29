"""Structured logging for RSI-Bench."""
import json
import time
import logging
from typing import Any, Dict, Optional
from pathlib import Path


class BenchmarkLogger:
    """Structured logger for benchmark runs."""

    def __init__(self, name: str = "rsi_bench", log_dir: Optional[str] = None, level: int = logging.INFO):
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else None
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
            self._logger.addHandler(handler)
        self._events = []
        self._start_time = time.time()

    def log_event(self, event_type: str, data: Dict[str, Any], level: int = logging.INFO):
        event = {"type": event_type, "timestamp": time.time() - self._start_time, "data": data}
        self._events.append(event)
        self._logger.log(level, f"{event_type}: {json.dumps(data, default=str)}")

    def log_iteration(self, iteration: int, scores: Dict[str, float], metadata: Optional[Dict] = None):
        self.log_event("iteration", {"iteration": iteration, "scores": scores, **(metadata or {})})

    def log_axis_result(self, axis_name: str, score: float, details: Optional[Dict] = None):
        self.log_event("axis_result", {"axis": axis_name, "score": score, **(details or {})})

    def log_error(self, message: str, exception: Optional[Exception] = None):
        self.log_event("error", {"message": message, "exception": str(exception)}, logging.ERROR)

    def save(self, filepath: Optional[str] = None):
        path = Path(filepath) if filepath else (self.log_dir or Path(".")) / f"{self.name}_{int(time.time())}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for event in self._events:
                f.write(json.dumps(event, default=str) + "\n")
        return str(path)

    @property
    def summary(self) -> Dict[str, Any]:
        return {"name": self.name, "events": len(self._events), "duration": time.time() - self._start_time}
