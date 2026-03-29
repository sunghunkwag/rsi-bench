"""RSI-Bench Task Suites."""
from abc import ABC, abstractmethod


class TaskBase(ABC):
    def __init__(self, name="base_task", difficulty="medium"):
        self.name = name
        self.difficulty = difficulty

    @abstractmethod
    def generate(self, seed=None):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, system_output, expected):
        raise NotImplementedError


from rsi_bench.tasks.symbolic_regression import SymbolicRegressionTask
from rsi_bench.tasks.program_synthesis import ProgramSynthesisTask
from rsi_bench.tasks.architecture_search import ArchitectureSearchTask
from rsi_bench.tasks.distribution_shift import DistributionShiftTask

__all__ = [
    "TaskBase", "SymbolicRegressionTask", "ProgramSynthesisTask",
    "ArchitectureSearchTask", "DistributionShiftTask",
]
