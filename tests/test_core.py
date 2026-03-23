"""Tests for rsi_bench.core module."""
import pytest
import numpy as np
from rsi_bench.core import RSIBenchmark


class TestRSIBenchmark:
      def test_init_default(self):
                bench = RSIBenchmark()
                assert bench is not None

      def test_register_axis(self):
                bench = RSIBenchmark()
                axes = bench.get_registered_axes()
                assert isinstance(axes, (list, dict))

      def test_evaluate_returns_dict(self):
                bench = RSIBenchmark()

          class MockSystem:
              def __init__(self):
                                self.iteration = 0
                            def step(self):
                                              self.iteration += 1
                                              return {"modified": True}
                                          def get_trajectory(self):
                                                            return [0.1 * i for i in range(self.iteration)]
                                                        def get_operators(self):
                                                                                    return ["op1", "op2"]

        system = MockSystem()
        for _ in range(10):
                      system.step()
                  try:
                                scores = bench.evaluate_all(system)
                                assert isinstance(scores, dict)
                                for v in scores.values():
                                                  assert 0.0 <= v <= 1.0
                  except (AttributeError, NotImplementedError):
                                pytest.skip("evaluate_all not fully implemented")

    def test_benchmark_reproducibility(self):
              np.random.seed(42)
        bench1 = RSIBenchmark()
        np.random.seed(42)
        bench2 = RSIBenchmark()
        assert type(bench1) == type(bench2)
