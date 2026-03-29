"""RSI-Bench Integration Tests.

End-to-end tests verifying that all components work together:
  MockRSISystem -> RSIBenchmark -> all 6 axes -> UnifiedScorer -> BenchmarkResults
"""
import pytest
import numpy as np
from rsi_bench.core import RSIBenchmark, AxisResult, BenchmarkResults, SystemInterface
from rsi_bench.scoring import UnifiedScorer
from rsi_bench.statistics.bootstrap import BootstrapCI
from rsi_bench.statistics.convergence import ConvergenceDetector
from rsi_bench.statistics.pareto import ParetoAnalyzer


class MockRSISystem:
    """Full-featured mock RSI system for integration tests."""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.cycle = 0
        self.perf = 1.0
        self.ops = []
        self.goals = []

    def modify(self):
        self.cycle += 1
        self.perf += self.rng.normal(0.02, 0.05)
        lvl = min(self.cycle // 10, 4)
        res = {"level": lvl, "operators": [], "goals": [], "rolled_back": False}
        if self.cycle % 5 == 0:
            op = {"type": self.rng.choice(["unary", "binary"]),
                  "name": f"op_{self.cycle}", "cross_task": self.cycle % 10 == 0}
            self.ops.append(op)
            res["operators"] = [op]
        if self.cycle % 7 == 0:
            g = {"complexity": self.cycle * 3, "feasible": True}
            self.goals.append(g)
            res["goals"] = [g]
        return res

    def evaluate(self):
        return {"fitness": self.perf, "score": self.perf}

    def get_state(self):
        return {"cycle": self.cycle, "goals": self.goals[-1:],
                "new_operators": [], "sandbox_violation": False}

    def reset(self):
        self.__init__()


# ── Scoring integration ─────────────────────────────────────────────────────

class TestUnifiedScorerIntegration:

    def test_uniform_scores(self):
        s = UnifiedScorer()
        scores = {"smd": 0.8, "itq": 0.8, "odr": 0.8,
                  "mas": 0.8, "ssm": 0.8, "agg": 0.8}
        assert abs(s.compute_from_dict(scores) - 0.8) < 1e-6

    def test_zero_axis_penalty(self):
        s = UnifiedScorer()
        scores = {"smd": 1.0, "itq": 1.0, "odr": 0.0,
                  "mas": 1.0, "ssm": 1.0, "agg": 1.0}
        assert s.compute_from_dict(scores) < 0.01

    def test_empty(self):
        assert UnifiedScorer().compute_from_dict({}) == 0.0


# ── Statistics integration ──────────────────────────────────────────────────

class TestBootstrapCIIntegration:

    def test_ci_bracketing(self):
        data = np.random.default_rng(42).normal(10, 2, 100)
        r = BootstrapCI(n_samples=500, seed=42).compute(data)
        assert r["ci_lower"] < r["estimate"] < r["ci_upper"]


class TestConvergenceDetectorIntegration:

    def test_improving_sequence(self):
        det = ConvergenceDetector(window_size=10)
        for i in range(20):
            r = det.update(float(i))
        assert r["state"] == "improving"

    def test_flat_sequence_converges(self):
        det = ConvergenceDetector(window_size=10, volatility_threshold=0.1)
        for _ in range(50):
            r = det.update(5.0)
        assert r["state"] == "converged"


class TestParetoAnalyzerIntegration:

    def test_frontier_size(self):
        pa = ParetoAnalyzer(
            objective_directions={"perf": "maximize", "cost": "minimize"},
            reference_point={"perf": 0.0, "cost": 100.0},
        )
        pa.add_solution({"perf": 0.9, "cost": 50})
        pa.add_solution({"perf": 0.7, "cost": 30})
        pa.add_solution({"perf": 0.5, "cost": 60})  # dominated by (0.7, 30)
        report = pa.get_report()
        assert report["metrics"]["frontier_size"] == 2


# ── Full benchmark integration ──────────────────────────────────────────────

class TestFullBenchmark:

    def test_end_to_end(self):
        """Full pipeline: register system -> run all axes -> get composite."""
        sys = MockRSISystem(seed=42)
        bench = RSIBenchmark(seed=42)
        bench.register_system(
            name="MockRSI",
            modify_fn=sys.modify,
            evaluate_fn=sys.evaluate,
            get_state_fn=sys.get_state,
            reset_fn=sys.reset,
        )
        results = bench.run(max_cycles=20, verbose=False)

        assert isinstance(results, BenchmarkResults)
        assert results.composite_score >= 0.0
        assert len(results.axis_results) == 6
        assert results.total_duration > 0

        # Every axis should have run
        for name in RSIBenchmark.AXIS_NAMES:
            assert name in results.axis_results
            assert 0.0 <= results.axis_results[name].score <= 1.0

    def test_summary_output(self):
        sys = MockRSISystem(seed=42)
        bench = RSIBenchmark(seed=42)
        bench.register_system(
            name="MockRSI",
            modify_fn=sys.modify,
            evaluate_fn=sys.evaluate,
            get_state_fn=sys.get_state,
            reset_fn=sys.reset,
        )
        results = bench.run(max_cycles=10, verbose=False)
        summary = results.summary()
        assert "MockRSI" in summary
        assert "Composite RSI Score" in summary

    def test_json_export(self, tmp_path):
        sys = MockRSISystem(seed=42)
        bench = RSIBenchmark(seed=42)
        bench.register_system(
            name="MockRSI",
            modify_fn=sys.modify,
            evaluate_fn=sys.evaluate,
            get_state_fn=sys.get_state,
            reset_fn=sys.reset,
        )
        results = bench.run(max_cycles=10, verbose=False)
        export_path = str(tmp_path / "results.json")
        results.to_json(export_path)

        import json
        with open(export_path) as f:
            data = json.load(f)
        assert "composite_score" in data
        assert "axes" in data
        assert len(data["axes"]) == 6

    def test_single_axis_mode(self):
        """Run only one axis and verify it works independently."""
        sys = MockRSISystem(seed=42)
        bench = RSIBenchmark(seed=42)
        bench.register_system(
            name="MockRSI",
            modify_fn=sys.modify,
            evaluate_fn=sys.evaluate,
            get_state_fn=sys.get_state,
            reset_fn=sys.reset,
        )

        for axis_key in ["smd", "itq", "odr", "mas", "ssm", "agg"]:
            sys.reset()
            result = bench.run_single_axis(axis_key, max_cycles=10)
            assert isinstance(result, AxisResult)
            assert 0.0 <= result.score <= 1.0, \
                f"Axis {axis_key} returned score {result.score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
