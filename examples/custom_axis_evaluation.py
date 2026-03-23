#!/usr/bin/env python3
"""Example: Creating and evaluating a custom axis.

Shows how to extend RSI-Bench with custom evaluation axes.
"""
import numpy as np
from rsi_bench.axes.base import AxisBase
from rsi_bench.scoring import RSIScorer


class CustomResourceEfficiencyAxis(AxisBase):
      """Custom axis measuring computational resource efficiency."""

    name = "resource_efficiency"
    weight = 1.0

    def evaluate(self, system_data):
              compute_used = system_data.get("compute_history", [])
              perf_history = system_data.get("performance_history", [])
              if not compute_used or not perf_history:
                            return {"score": 0.0, "details": {}}

              gains = [max(0, perf_history[i] - perf_history[i - 1]) for i in range(1, len(perf_history))]
              costs = compute_used[1:] if len(compute_used) > 1 else [1.0]
              efficiency = [g / max(c, 1e-6) for g, c in zip(gains, costs)]
              avg_eff = float(np.mean(efficiency)) if efficiency else 0.0
              trend = float(np.polyfit(range(len(efficiency)), efficiency, 1)[0]) if len(efficiency) > 2 else 0.0

        score = float(np.clip(avg_eff * 10 + max(0, trend) * 5, 0, 1))
        return {"score": score, "details": {"avg_efficiency": avg_eff, "trend": trend, "n_steps": len(gains)}}


def main():
      print("=== Custom Axis Evaluation Example ===\n")
      np.random.seed(42)

    system_data = {
              "compute_history": [10 + np.random.normal(0, 1) for _ in range(50)],
              "performance_history": [0.3 + 0.01 * i + np.random.normal(0, 0.02) for i in range(50)],
    }

    axis = CustomResourceEfficiencyAxis()
    result = axis.evaluate(system_data)
    print(f"Resource Efficiency Score: {result['score']:.4f}")
    print(f"Details: {result['details']}")

    scorer = RSIScorer()
    combined = {"resource_efficiency": result["score"], "modification_depth": 0.75, "trajectory_quality": 0.82}
    final = scorer.aggregate(combined)
    print(f"\nCombined Score with custom axis: {final:.4f}")


if __name__ == "__main__":
      main()
