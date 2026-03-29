#!/usr/bin/env python3
"""Example: Creating and evaluating a custom axis.

Shows how to extend RSI-Bench with custom evaluation axes
and integrate them into the unified scoring system.
"""
import numpy as np
from rsi_bench.axes.base import AxisBase
from rsi_bench.core import AxisResult, SystemInterface
from rsi_bench.scoring import UnifiedScorer


class CustomResourceEfficiencyAxis(AxisBase):
    """Custom axis measuring computational resource efficiency.

    Evaluates how much performance gain the system achieves
    per unit of computational cost during self-modification.
    """

    name = "Resource Efficiency"

    def evaluate(self, system, max_cycles=50, **kwargs):
        """Run system and track performance-per-compute efficiency."""
        compute_used = []
        perf_history = []

        for cycle in range(max_cycles):
            # Each modification has a simulated compute cost
            system.modify_fn()
            perf = system.evaluate_fn()
            if isinstance(perf, dict):
                perf_val = perf.get("fitness", perf.get("score", 0.0))
            else:
                perf_val = float(perf)
            perf_history.append(perf_val)
            # Simulate compute cost (in practice, measure wall time or FLOPs)
            compute_used.append(10 + self.rng.normal(0, 1))

        if len(perf_history) < 2:
            return AxisResult(axis_name=self.name, score=0.0, metrics={})

        # Compute efficiency: performance gain per compute unit
        gains = [max(0, perf_history[i] - perf_history[i - 1])
                 for i in range(1, len(perf_history))]
        costs = compute_used[1:]
        efficiency = [g / max(c, 1e-6) for g, c in zip(gains, costs)]

        avg_eff = float(np.mean(efficiency)) if efficiency else 0.0
        trend = float(np.polyfit(range(len(efficiency)), efficiency, 1)[0]) \
            if len(efficiency) > 2 else 0.0

        score = float(np.clip(avg_eff * 10 + max(0, trend) * 5, 0, 1))

        return AxisResult(
            axis_name=self.name,
            score=score,
            metrics={
                "avg_efficiency": avg_eff,
                "efficiency_trend": trend,
                "n_steps": len(gains),
            },
        )


class DemoSystem:
    """Minimal system for the custom axis demo."""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.perf = 0.3
        self.cycle = 0

    def modify(self):
        self.cycle += 1
        return {"level": 0}

    def evaluate(self):
        self.perf += 0.01 + self.rng.normal(0, 0.005)
        return {"fitness": self.perf}

    def get_state(self):
        return {"cycle": self.cycle}


def main():
    print("=== Custom Axis Evaluation Example ===\n")

    # Create system and interface
    system = DemoSystem(seed=42)
    iface = SystemInterface(
        name="DemoSystem",
        modify_fn=system.modify,
        evaluate_fn=system.evaluate,
        get_state_fn=system.get_state,
    )

    # Evaluate custom axis
    axis = CustomResourceEfficiencyAxis()
    result = axis.evaluate(iface, max_cycles=50)
    print(f"Resource Efficiency Score: {result.score:.4f}")
    print(f"Metrics: {result.metrics}")

    # Combine with standard axis scores using UnifiedScorer
    scorer = UnifiedScorer()
    combined_scores = {
        "smd": 0.75,
        "itq": 0.82,
        "odr": 0.60,
        "mas": 0.55,
        "ssm": 0.90,
        "agg": 0.45,
    }
    composite = scorer.compute_from_dict(combined_scores)
    print(f"\nComposite score (standard axes): {composite:.4f}")
    print(f"Custom axis could be integrated via custom weights.")


if __name__ == "__main__":
    main()
