"""Tests for rsi_bench.core module.

RSIBenchmark API:
  - register_system(name, modify_fn, evaluate_fn, get_state_fn, reset_fn)
  - run(max_cycles=50, seed=None, verbose=True) -> BenchmarkResults
  - run_single_axis(axis, max_cycles=50) -> AxisResult
"""
import pytest
import numpy as np
from rsi_bench.core import RSIBenchmark, AxisResult, BenchmarkResults, SystemInterface


class MockSystem:
    """Minimal mock for core benchmark tests."""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.cycle = 0
        self.perf = 1.0

    def modify(self):
        self.cycle += 1
        lvl = min(self.cycle // 5, 4)
        return {"level": lvl, "operators": [], "goals": [], "rolled_back": False}

    def evaluate(self):
        self.perf += self.rng.normal(0.02, 0.01)
        return {"fitness": self.perf, "score": self.perf}

    def get_state(self):
        return {"cycle": self.cycle, "goals": [], "new_operators": [],
                "sandbox_violation": False}

    def reset(self):
        self.cycle = 0
        self.perf = 1.0


class TestRSIBenchmark:

    def test_init_default(self):
        bench = RSIBenchmark()
        assert bench is not None
        assert bench.seed == 42

    def test_init_custom_axes(self):
        bench = RSIBenchmark(axes=["smd", "itq"])
        assert bench._active_axes == ["smd", "itq"]

    def test_register_system(self):
        bench = RSIBenchmark()
        sys = MockSystem()
        bench.register_system(
            name="TestSystem",
            modify_fn=sys.modify,
            evaluate_fn=sys.evaluate,
            get_state_fn=sys.get_state,
            reset_fn=sys.reset,
        )
        assert bench.system is not None
        assert bench.system.name == "TestSystem"

    def test_run_without_registration_raises(self):
        bench = RSIBenchmark()
        with pytest.raises(RuntimeError, match="No system registered"):
            bench.run()

    def test_run_returns_benchmark_results(self):
        bench = RSIBenchmark(seed=42)
        sys = MockSystem()
        bench.register_system(
            name="TestSystem",
            modify_fn=sys.modify,
            evaluate_fn=sys.evaluate,
            get_state_fn=sys.get_state,
            reset_fn=sys.reset,
        )
        results = bench.run(max_cycles=10, verbose=False)
        assert isinstance(results, BenchmarkResults)
        assert results.system_name == "TestSystem"
        assert results.composite_score >= 0.0
        assert len(results.axis_results) == 6

    def test_all_axis_scores_bounded(self):
        bench = RSIBenchmark(seed=42)
        sys = MockSystem()
        bench.register_system(
            name="TestSystem",
            modify_fn=sys.modify,
            evaluate_fn=sys.evaluate,
            get_state_fn=sys.get_state,
            reset_fn=sys.reset,
        )
        results = bench.run(max_cycles=10, verbose=False)
        for name, axis_result in results.axis_results.items():
            assert 0.0 <= axis_result.score <= 1.0, \
                f"Axis {name} score {axis_result.score} out of bounds"

    def test_run_single_axis(self):
        bench = RSIBenchmark(seed=42)
        sys = MockSystem()
        bench.register_system(
            name="TestSystem",
            modify_fn=sys.modify,
            evaluate_fn=sys.evaluate,
            get_state_fn=sys.get_state,
            reset_fn=sys.reset,
        )
        result = bench.run_single_axis("itq", max_cycles=10)
        assert isinstance(result, AxisResult)
        assert 0.0 <= result.score <= 1.0

    def test_run_single_axis_invalid_raises(self):
        bench = RSIBenchmark(seed=42)
        sys = MockSystem()
        bench.register_system(
            name="TestSystem",
            modify_fn=sys.modify,
            evaluate_fn=sys.evaluate,
            get_state_fn=sys.get_state,
        )
        with pytest.raises(ValueError, match="Unknown axis"):
            bench.run_single_axis("nonexistent")

    def test_benchmark_reproducibility(self):
        """Same seed should produce same composite score."""
        scores = []
        for _ in range(2):
            bench = RSIBenchmark(seed=99)
            sys = MockSystem(seed=99)
            bench.register_system(
                name="Repro",
                modify_fn=sys.modify,
                evaluate_fn=sys.evaluate,
                get_state_fn=sys.get_state,
                reset_fn=sys.reset,
            )
            results = bench.run(max_cycles=10, seed=99, verbose=False)
            scores.append(results.composite_score)
        assert scores[0] == pytest.approx(scores[1], abs=1e-6)

    def test_summary_string(self):
        bench = RSIBenchmark(seed=42)
        sys = MockSystem()
        bench.register_system(
            name="TestSystem",
            modify_fn=sys.modify,
            evaluate_fn=sys.evaluate,
            get_state_fn=sys.get_state,
        )
        results = bench.run(max_cycles=5, verbose=False)
        summary = results.summary()
        assert "TestSystem" in summary
        assert "Composite RSI Score" in summary


class TestAxisResult:

    def test_score_clipped(self):
        r = AxisResult(axis_name="test", score=1.5)
        assert r.score == 1.0
        r2 = AxisResult(axis_name="test", score=-0.3)
        assert r2.score == 0.0

    def test_default_metrics(self):
        r = AxisResult(axis_name="test", score=0.5)
        assert r.metrics == {}
        assert r.details == {}


class TestSystemInterface:

    def test_construction(self):
        sys = MockSystem()
        iface = SystemInterface(
            name="Test",
            modify_fn=sys.modify,
            evaluate_fn=sys.evaluate,
            get_state_fn=sys.get_state,
        )
        assert iface.name == "Test"
        assert iface.reset_fn is None
