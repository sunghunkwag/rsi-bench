"""Tests for rsi_bench.statistics module.

BootstrapCI API:
  - compute(data, statistic=None) -> dict with estimate, ci_lower, ci_upper, std_error, ...
  - compute_effect_size(group_a, group_b) -> dict with cohens_d, rank_biserial, pooled_std

ConvergenceDetector API:
  - update(value) -> dict with state, mean, variance, volatility, trend, count, min, max
  - get_statistics() -> dict with count, mean, variance, std, min, max, range
  - reset()

ParetoAnalyzer API:
  - __init__(objective_directions, reference_point)
  - add_solution(solution) -> bool
  - compute_hypervolume() -> float
  - get_report() -> dict with frontier_size, total_solutions, metrics, frontier
  - reset()
"""
import pytest
import numpy as np
from rsi_bench.statistics.bootstrap import BootstrapCI
from rsi_bench.statistics.convergence import ConvergenceDetector
from rsi_bench.statistics.pareto import ParetoAnalyzer


# ── BootstrapCI ─────────────────────────────────────────────────────────────

class TestBootstrapCI:

    def test_basic_confidence_interval(self):
        data = np.random.default_rng(42).normal(10, 2, 100)
        ci = BootstrapCI(n_samples=1000, seed=42)
        result = ci.compute(data)
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "estimate" in result
        assert result["ci_lower"] < result["estimate"] < result["ci_upper"]

    def test_ci_contains_true_mean(self):
        """95% CI from normal data should contain the true mean."""
        data = np.random.default_rng(42).normal(10, 2, 200)
        ci = BootstrapCI(n_samples=2000, confidence=0.95, seed=42)
        result = ci.compute(data)
        assert result["ci_lower"] < 10 < result["ci_upper"]

    def test_std_error_positive(self):
        data = np.random.default_rng(42).normal(5, 1, 50)
        ci = BootstrapCI(n_samples=500, seed=42)
        result = ci.compute(data)
        assert result["std_error"] > 0

    def test_single_value(self):
        """Single data point should return point estimate with zero-width CI."""
        ci = BootstrapCI(seed=42)
        result = ci.compute([7.0])
        assert result["estimate"] == 7.0
        assert result["ci_lower"] == result["ci_upper"]

    def test_custom_statistic(self):
        data = np.random.default_rng(42).exponential(5, 100)
        ci = BootstrapCI(n_samples=500, seed=42)
        result = ci.compute(data, statistic=np.median)
        assert result["estimate"] == pytest.approx(np.median(data))

    def test_bias_reported(self):
        data = np.random.default_rng(42).normal(0, 1, 100)
        ci = BootstrapCI(n_samples=500, seed=42)
        result = ci.compute(data)
        assert "bias" in result
        assert "boot_mean" in result

    def test_effect_size_large_difference(self):
        """Two well-separated groups should yield large Cohen's d."""
        a = np.random.default_rng(42).normal(10, 1, 50)
        b = np.random.default_rng(42).normal(15, 1, 50)
        ci = BootstrapCI(seed=42)
        result = ci.compute_effect_size(a, b)
        assert "cohens_d" in result
        assert "rank_biserial" in result
        assert "pooled_std" in result
        assert abs(result["cohens_d"]) > 1.0  # large effect

    def test_effect_size_no_difference(self):
        """Same distribution should yield small Cohen's d."""
        rng = np.random.default_rng(42)
        a = rng.normal(10, 2, 100)
        b = rng.normal(10, 2, 100)
        ci = BootstrapCI(seed=42)
        result = ci.compute_effect_size(a, b)
        assert abs(result["cohens_d"]) < 1.0


# ── ConvergenceDetector ─────────────────────────────────────────────────────

class TestConvergenceDetector:

    def test_improving_detection(self):
        detector = ConvergenceDetector(window_size=10)
        for i in range(20):
            result = detector.update(float(i))
        assert result["state"] == "improving"

    def test_converged_detection(self):
        detector = ConvergenceDetector(window_size=10, volatility_threshold=0.05)
        rng = np.random.default_rng(42)
        for _ in range(50):
            result = detector.update(0.5 + rng.normal(0, 0.001))
        assert result["state"] == "converged"

    def test_degrading_detection(self):
        detector = ConvergenceDetector(window_size=10)
        for i in range(20):
            result = detector.update(20.0 - float(i))
        assert result["state"] == "degrading"

    def test_update_returns_full_dict(self):
        detector = ConvergenceDetector()
        result = detector.update(1.0)
        assert "state" in result
        assert "mean" in result
        assert "variance" in result
        assert "volatility" in result
        assert "trend" in result
        assert "count" in result
        assert "min" in result
        assert "max" in result

    def test_get_statistics(self):
        detector = ConvergenceDetector()
        for v in range(10):
            detector.update(float(v))
        stats = detector.get_statistics()
        assert "mean" in stats
        assert "variance" in stats
        assert "std" in stats
        assert "count" in stats
        assert stats["count"] == 10
        assert stats["min"] == 0.0
        assert stats["max"] == 9.0

    def test_reset(self):
        detector = ConvergenceDetector()
        for v in range(10):
            detector.update(float(v))
        detector.reset()
        stats = detector.get_statistics()
        assert stats["count"] == 0

    def test_state_in_valid_set(self):
        valid_states = {"improving", "exploring", "converged", "degrading"}
        detector = ConvergenceDetector(window_size=5)
        rng = np.random.default_rng(42)
        for _ in range(30):
            result = detector.update(rng.normal(0, 1))
            assert result["state"] in valid_states


# ── ParetoAnalyzer ──────────────────────────────────────────────────────────

class TestParetoAnalyzer:

    def test_single_solution_on_frontier(self):
        pa = ParetoAnalyzer(
            objective_directions={"perf": "maximize", "cost": "minimize"},
            reference_point={"perf": 0.0, "cost": 100.0},
        )
        result = pa.add_solution({"perf": 0.8, "cost": 30})
        assert result is True
        report = pa.get_report()
        assert report["frontier_size"] == 1

    def test_dominated_solution_excluded(self):
        pa = ParetoAnalyzer(
            objective_directions={"perf": "maximize", "cost": "minimize"},
            reference_point={"perf": 0.0, "cost": 100.0},
        )
        pa.add_solution({"perf": 0.9, "cost": 20})
        pa.add_solution({"perf": 0.7, "cost": 30})  # dominated
        pa.add_solution({"perf": 0.5, "cost": 60})  # dominated
        report = pa.get_report()
        assert report["frontier_size"] == 1  # only the (0.9, 20) solution

    def test_non_dominated_solutions(self):
        pa = ParetoAnalyzer(
            objective_directions={"perf": "maximize", "cost": "minimize"},
            reference_point={"perf": 0.0, "cost": 100.0},
        )
        pa.add_solution({"perf": 0.9, "cost": 50})
        pa.add_solution({"perf": 0.7, "cost": 30})
        report = pa.get_report()
        # Both are non-dominated (trade-off between perf and cost)
        assert report["frontier_size"] == 2

    def test_hypervolume_positive(self):
        pa = ParetoAnalyzer(
            objective_directions={"perf": "maximize", "cost": "minimize"},
            reference_point={"perf": 0.0, "cost": 100.0},
        )
        pa.add_solution({"perf": 0.8, "cost": 40})
        pa.add_solution({"perf": 0.5, "cost": 20})
        hv = pa.compute_hypervolume()
        assert hv > 0.0

    def test_empty_hypervolume(self):
        pa = ParetoAnalyzer(
            objective_directions={"a": "maximize", "b": "maximize"},
            reference_point={"a": 0.0, "b": 0.0},
        )
        assert pa.compute_hypervolume() == 0.0

    def test_report_structure(self):
        pa = ParetoAnalyzer(
            objective_directions={"x": "maximize"},
            reference_point={"x": 0.0},
        )
        pa.add_solution({"x": 0.5})
        report = pa.get_report()
        assert "frontier_size" in report
        assert "total_solutions" in report
        assert "metrics" in report
        assert "frontier" in report
        assert "hypervolume" in report["metrics"]
        assert "dominance_ratio" in report["metrics"]

    def test_reset(self):
        pa = ParetoAnalyzer(
            objective_directions={"a": "maximize"},
            reference_point={"a": 0.0},
        )
        pa.add_solution({"a": 0.5})
        pa.add_solution({"a": 0.8})
        pa.reset()
        assert pa.get_report()["frontier_size"] == 0
        assert pa.get_report()["total_solutions"] == 0
