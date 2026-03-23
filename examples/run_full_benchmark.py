#!/usr/bin/env python3
"""Run a full RSI-Bench benchmark evaluation.

Usage:
    python examples/run_full_benchmark.py --iterations 50 --output results/
    """
import argparse
import json
import numpy as np
from pathlib import Path

from rsi_bench.core import RSIBenchmark
from rsi_bench.scoring import RSIScorer
from rsi_bench.utils.logging import BenchmarkLogger


def create_dummy_system():
      """Create a dummy RSI system for demonstration."""
      class DummyRSISystem:
                def __init__(self):
                              self.iteration = 0
                          def step(self):
                                        self.iteration += 1
                                        return {"modified": True, "depth": min(self.iteration, 5)}
                                    def get_trajectory(self):
                                                  return [0.1 * i + np.random.normal(0, 0.02) for i in range(self.iteration)]
                                              def get_operators(self):
                                                            ops = ["mutate", "crossover", "select"]
                                                            if self.iteration > 10:
                                                                              ops.append("novel_op_1")
                                                                          return ops
                                                    return DummyRSISystem()


def main():
      parser = argparse.ArgumentParser(description="Run RSI-Bench full benchmark")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    logger = BenchmarkLogger(name="full_benchmark", log_dir=args.output)
    benchmark = RSIBenchmark()
    scorer = RSIScorer()
    system = create_dummy_system()

    print("=== RSI-Bench Full Benchmark ===")
    print(f"Iterations: {args.iterations}")

    for i in range(args.iterations):
              result = system.step()
        trajectory = system.get_trajectory()
        logger.log_iteration(i, {"step_result": str(result)})

    scores = benchmark.evaluate_all(system)
    for axis_name, score in scores.items():
              logger.log_axis_result(axis_name, score)
        print(f"  {axis_name}: {score:.4f}")

    final = scorer.aggregate(scores)
    print(f"\nFinal RSI Score: {final:.4f}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = logger.save()
    with open(output_dir / "scores.json", "w") as f:
              json.dump({"scores": scores, "final": final}, f, indent=2)
    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
      main()
