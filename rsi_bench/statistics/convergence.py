"""
O(1) Memory-Efficient Convergence Detector for RSI-Bench
=========================================================
Classifies system state: improving, exploring, converged, degrading.
"""

import numpy as np
from typing import Dict, Optional


class ConvergenceDetector:
    """O(1) memory convergence phase detector."""

    def __init__(self, window_size: int = 50, volatility_threshold: float = 0.05,
                 improvement_threshold: float = 0.01):
        self.window_size = window_size
        self.volatility_threshold = volatility_threshold
        self.improvement_threshold = improvement_threshold
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.prev_mean = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.recent_values = []
        self.state_history = []

    def update(self, value: float) -> Dict:
        """Update with a new observation and return current state classification."""
        self.count += 1
        value = float(value)
        self.recent_values.append(value)
        if len(self.recent_values) > self.window_size:
            self.recent_values.pop(0)
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        variance = self.m2 / max(self.count - 1, 1) if self.count > 1 else 0.0
        volatility = np.sqrt(variance) / max(abs(self.mean), 1e-10)
        if len(self.recent_values) >= 3:
            recent_mean = np.mean(self.recent_values[-min(10, len(self.recent_values)):])
            older_mean = np.mean(self.recent_values[:max(len(self.recent_values) // 2, 1)])
            trend = (recent_mean - older_mean) / max(abs(older_mean), 1e-10)
        else:
            trend = 0.0
        state = self._classify(volatility, trend)
        self.state_history.append(state)
        self.prev_mean = self.mean
        return {
            "state": state,
            "mean": float(self.mean),
            "variance": float(variance),
            "volatility": float(volatility),
            "trend": float(trend),
            "count": self.count,
            "min": float(self.min_val),
            "max": float(self.max_val),
        }

    def _classify(self, volatility: float, trend: float) -> str:
        if trend > self.improvement_threshold:
            return "improving"
        elif trend < -self.improvement_threshold:
            return "degrading"
        elif volatility > self.volatility_threshold:
            return "exploring"
        else:
            return "converged"

    def get_statistics(self) -> Dict:
        """Return summary statistics."""
        variance = self.m2 / max(self.count - 1, 1) if self.count > 1 else 0.0
        return {
            "count": self.count,
            "mean": float(self.mean),
            "variance": float(variance),
            "std": float(np.sqrt(variance)),
            "min": float(self.min_val),
            "max": float(self.max_val),
            "range": float(self.max_val - self.min_val),
        }

    def reset(self):
        """Reset the detector state."""
        self.__init__(self.window_size, self.volatility_threshold, self.improvement_threshold)
