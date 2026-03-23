"""RSI-Bench Integration Tests with MockRSISystem."""
import pytest
import numpy as np
from rsi_bench.core import RSIBenchmark, AxisResult, BenchmarkResults
from rsi_bench.scoring import UnifiedScorer
from rsi_bench.statistics.bootstrap import BootstrapCI
from rsi_bench.statistics.convergence import ConvergenceDetector
from rsi_bench.statistics.pareto import ParetoAnalyzer

class MockRSISystem:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.cycle = 0; self.perf = 1.0; self.ops = []; self.goals = []
    def modify(self):
        self.cycle += 1; self.perf += self.rng.normal(0.02, 0.05)
        lvl = min(self.cycle // 10, 4)
        res = {"level": lvl, "operators": [], "goals": []}
        if self.cycle % 5 == 0:
            op = {"type": self.rng.choice(["unary","binary"]), "name": f"op_{self.cycle}"}
            self.ops.append(op); res["operators"] = [op]
        if self.cycle % 7 == 0:
            g = {"complexity": self.cycle, "feasible": True}
            self.goals.append(g); res["goals"] = [g]
        return res
    def evaluate(self):
        return {"fitness": self.perf, "score": self.perf}
    def get_state(self):
        return {"cycle": self.cycle, "goals": self.goals[-1:], "new_operators": []}
    def reset(self): self.__init__()

class TestUnifiedScorer:
    def test_uniform(self):
        s = UnifiedScorer()
        assert abs(s.compute_from_dict({"smd":.8,"itq":.8,"odr":.8,"mas":.8,"ssm":.8,"agg":.8}) - 0.8) < 1e-6
    def test_zero_penalty(self):
        s = UnifiedScorer()
        assert s.compute_from_dict({"smd":1,"itq":1,"odr":0,"mas":1,"ssm":1,"agg":1}) < 0.01
    def test_empty(self):
        assert UnifiedScorer().compute_from_dict({}) == 0.0

class TestBootstrapCI:
    def test_basic(self):
        d = np.random.default_rng(42).normal(10, 2, 100)
        r = BootstrapCI(n_samples=500).compute(d)
        assert r["ci_lower"] < r["estimate"] < r["ci_upper"]

class TestConvergenceDetector:
    def test_improving(self):
        det = ConvergenceDetector(window_size=10)
        for i in range(20): r = det.update(float(i))
        assert r["state"] == "improving"

class TestParetoAnalyzer:
    def test_frontier(self):
        pa = ParetoAnalyzer({"perf":"maximize","cost":"minimize"}, {"perf":0,"cost":100})
        pa.add_solution({"perf": 0.9, "cost": 50})
        pa.add_solution({"perf": 0.7, "cost": 30})
        pa.add_solution({"perf": 0.5, "cost": 60})
        assert pa.get_report()["metrics"]["frontier_size"] == 2

class TestFullBenchmark:
    def test_mock(self):
        sys = MockRSISystem()
        b = RSIBenchmark(seed=42)
        b.register_system(name="Mock", modify_fn=sys.modify, evaluate_fn=sys.evaluate,
                          get_state_fn=sys.get_state, reset_fn=sys.reset)
        r = b.run(max_cycles=20, verbose=False)
        assert isinstance(r, BenchmarkResults)
        assert r.composite_score >= 0.0
        assert len(r.axis_results) == 6

if __name__ == "__main__": pytest.main([__file__, "-v"])
