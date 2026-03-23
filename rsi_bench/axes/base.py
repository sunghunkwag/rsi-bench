"""Base class for all RSI-Bench evaluation axes."""

from abc import ABC, abstractmethod
import numpy as np


class AxisBase(ABC):
    """Abstract base class for evaluation axes."""

    name: str = "base"

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(42)

    @abstractmethod
    def evaluate(self, system, max_cycles=50, **kwargs):
        """Evaluate the system on this axis. Returns AxisResult."""
        raise NotImplementedError
