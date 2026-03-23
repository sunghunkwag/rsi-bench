"""Tests for rsi_bench.statistics module."""
import pytest
import numpy as np
from rsi_bench.statistics.bootstrap import BootstrapCI
from rsi_bench.statistics.convergence import ConvergenceDetector
from rsi_bench.statistics.pareto import ParetoAnalyzer


class TestBootstrapCI:
      def test_basic_ci(self):
                np.random.seed(42)
                data = np.random.normal(10, 2, 100)
                ci = BootstrapCI()
                result = ci.compute(data)
                assert "lower" in result and "upper" in result
                assert result["lower"] < result["upper"]
                assert result["lower"] < 10 < result["upper"]

      def test_effect_size(self):
                a = np.random.normal(10, 1, 50)
                b = np.random.normal(12, 1, 50)
                ci = BootstrapCI()
                d = ci.cohens_d(a, b)
                assert abs(d) > 1.0


class TestConvergenceDetector:
      def test_improving(self):
                detector = ConvergenceDetector()
                for v in [0.1 * i for i in range(20)]:
                              detector.update(v)
                          phase = detector.get_phase()
                assert phase in ("improving", "exploring", "converged", "degrading")

      def test_converged(self):
                detector = ConvergenceDetector()
                for _ in range(50):
                              detector.update(0.5 + np.random.normal(0, 0.001))
                          phase = detector.get_phase()
                assert phase == "converged"

      def test_stats(self):
                detector = ConvergenceDetector()
                for v in range(10):
                              detector.update(float(v))
                          stats = detector.stats
                assert "mean" in stats
                assert "variance" in stats


class TestParetoAnalyzer:
      def test_dominance(self):
                analyzer = ParetoAnalyzer()
                assert analyzer.dominates([0.9, 0.8], [0.7, 0.6])
                assert not analyzer.dominates([0.9, 0.6], [0.7, 0.8])

      def test_frontier(self):
                analyzer = ParetoAnalyzer()
                points = [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.3, 0.3]]
                front = analyzer.get_frontier(points)
                assert len(front) >= 2
                assert [0.3, 0.3] not in front

      def test_hypervolume_2d(self):
                analyzer = ParetoAnalyzer()
                front = [[0.8, 0.6], [0.5, 0.9]]
                hv = analyzer.hypervolume_2d(front, ref=[0.0, 0.0])
                assert hv > 0.0
