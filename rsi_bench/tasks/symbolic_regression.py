"""Symbolic Regression Tasks for RSI-Bench."""
import numpy as np
from rsi_bench.tasks import TaskBase


class SymbolicRegressionTask(TaskBase):
      """Discover symbolic expressions fitting target functions."""

    name = "symbolic_regression"

    def __init__(self, target_fn=None, domain=(-5, 5), n_points=200, noise_std=0.0):
              self.target_fn = target_fn or (lambda x: np.sin(x) + x ** 2)
              self.domain = domain
              self.n_points = n_points
              self.noise_std = noise_std
              self._x = np.linspace(*domain, n_points)
              self._y = self.target_fn(self._x) + np.random.normal(0, noise_std, n_points)

    def get_initial_state(self):
              return {"x": self._x.tolist(), "y": self._y.tolist(), "best_expr": None, "best_mse": float("inf")}

    def evaluate_candidate(self, state, candidate_expr_fn):
              y_pred = candidate_expr_fn(self._x)
              mse = float(np.mean((self._y - y_pred) ** 2))
              if mse < state["best_mse"]:
                            state["best_mse"] = mse
                            state["best_expr"] = str(candidate_expr_fn)
                        return {"mse": mse, "r2": 1 - mse / max(np.var(self._y), 1e-12)}

    def score(self, state):
              if state["best_mse"] == float("inf"):
                            return 0.0
                        return max(0.0, 1.0 - state["best_mse"] / max(np.var(self._y), 1e-12))


PRESET_TARGETS = {
      "polynomial": lambda x: 0.5 * x ** 3 - 2 * x ** 2 + x - 1,
      "trigonometric": lambda x: np.sin(2 * x) * np.cos(x / 2),
      "composite": lambda x: np.exp(-x ** 2 / 4) * np.sin(3 * x),
}
