#!/usr/bin/env python3
"""Run a full RSI-Bench benchmark evaluation.

Usage:
    python examples/run_full_benchmark.py --iterations 50 --output results/
"""
import argparse
import json
import numpy as np
from pathlib import Path

from rsi_bench.core import RSIBenchmark, SystemInterface
from rsi_bench.scoring import UnifiedScorer
from rsi_bench.utils.logging import BenchmarkLogger


class DummyRSISystem:
    """Demonstration RSI system with simple improving behaviour."""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.iteration = 0
        self.performance = 1.0
        self.operators = ["mutate", "crossover", "select"]
        self.goals = []

    def modify(self):
        self.iteration += 1
        lvl = min(self.iteration // 10, 4)
        result = {"level": lvl, "operators": [], "goals": [], "rolled_back": False}
        if self.iteration > 10 and self.iteration % 5 == 0:
            op = {"type": "novel", "name": f"novel_op_{self.iteration}",
                  "cross_task": False}
            self.operators.append(op["name"])
            result["operators"] = [op]
        if self.iteration % 8 == 0:
            goal = {"complexity": self.iteration * 2, "feasible": True}
            self.goals.append(goal)
            result["goals"] = [goal]
        return result

    def evaluate(self):
        self.performance += self.rng.normal(0.01, 0.02)
        return {"fitness": self.performance, "score": self.performance}

    def get_state(self):
        return {
            "iteration": self.iteration,
            "goals": self.goals[-1:],
            "new_operators": [],
            "sandbox_violation": False,
        }

    def reset(self):
        self.__init__()


def main():
    parser = argparse.ArgumentParser(description="Run RSI-Bench full benchmark")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of self-modification cycles per axis")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = BenchmarkLogger(name="full_benchmark", log_dir=args.output)

    print("=== RSI-Bench Full Benchmark ===")
    print(f"Max cycles: {args.iterations}")
    print(f"Seed: {args.seed}")
    print()

    # Create and register system
    system = DummyRSISystem(seed=args.seed)
    benchmark = RSIBenchmark(seed=args.seed)
    benchmark.register_system(
        name="DummyRSISystem",
        modify_fn=system.modify,
        evaluate_fn=system.evaluate,
        get_state_fn=system.get_state,
        reset_fn=system.reset,
    )

    # Run full benchmark
    results = benchmark.run(max_cycles=args.iterations, verbose=True)

    # Log axis results
    for axis_name, axis_result in results.axis_results.items():
        logger.log_axis_result(axis_name, axis_result.score, axis_result.metrics)

    # Save results
    results.to_json(str(output_dir / "scores.json"))
    log_path = logger.save()
    print(f"\nResults saved to {args.output}/")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
