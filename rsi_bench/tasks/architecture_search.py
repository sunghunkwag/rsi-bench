"""Architecture Search Tasks for RSI-Bench."""
import numpy as np
from rsi_bench.tasks import TaskBase


class ArchitectureSearchTask(TaskBase):
      """Search for optimal neural network architectures."""

    name = "architecture_search"

    def __init__(self, search_space=None, budget=100):
              self.search_space = search_space or {
                            "layers": [1, 2, 3, 4, 5],
                            "units": [16, 32, 64, 128, 256],
                            "activation": ["relu", "tanh", "sigmoid"],
                            "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
              }
              self.budget = budget

    def get_initial_state(self):
              return {"evaluations": [], "best_arch": None, "best_perf": 0.0, "remaining": self.budget}

    def evaluate_candidate(self, state, architecture):
              if state["remaining"] <= 0:
                            return {"error": "budget_exhausted", "performance": 0.0}
                        state["remaining"] -= 1
        perf = self._simulate_performance(architecture)
        state["evaluations"].append({"arch": architecture, "perf": perf})
        if perf > state["best_perf"]:
                      state["best_perf"] = perf
                      state["best_arch"] = architecture
                  return {"performance": perf, "remaining_budget": state["remaining"]}

    def _simulate_performance(self, arch):
              np.random.seed(hash(str(arch)) % 2**31)
        layers = arch.get("layers", 2)
        units = arch.get("units", 64)
        base = 0.5 + 0.1 * min(layers, 3) + 0.15 * np.log2(max(units, 1)) / 8
        noise = np.random.normal(0, 0.05)
        dropout = arch.get("dropout", 0.0)
        penalty = 0.1 * max(0, dropout - 0.3)
        return float(np.clip(base + noise - penalty, 0.0, 1.0))

    def score(self, state):
              if not state["evaluations"]:
                            return 0.0
                        efficiency = 1.0 - state["remaining"] / self.budget
        return state["best_perf"] * (0.7 + 0.3 * efficiency)
