"""Tests for all evaluation axes.

Each axis evaluates a SystemInterface object with:
  - modify_fn() -> modification result
  - evaluate_fn() -> performance value or dict
  - get_state_fn() -> system state dict
  - reset_fn() -> reset (optional)

All axes return AxisResult with .score in [0, 1].
"""
import pytest
import numpy as np
from rsi_bench.core import AxisResult, SystemInterface
from rsi_bench.axes.axis1_modification_depth import SelfModificationDepth
from rsi_bench.axes.axis2_trajectory_quality import TrajectoryQuality
from rsi_bench.axes.axis3_operator_discovery import OperatorDiscovery
from rsi_bench.axes.axis4_meta_adaptation import MetaAdaptation
from rsi_bench.axes.axis5_safety_stability import SafetyStability
from rsi_bench.axes.axis6_goal_generation import GoalGeneration


# ── Helper: MockRSISystem ───────────────────────────────────────────────────

class MockRSISystem:
    """Configurable mock system for axis-level tests."""

    def __init__(self, trajectory="improving", seed=42):
        self.rng = np.random.default_rng(seed)
        self.cycle = 0
        self.perf = 1.0
        self.trajectory = trajectory
        self.ops = []
        self.goals = []

    def modify(self):
        self.cycle += 1
        lvl = min(self.cycle // 3, 4)
        result = {"level": lvl, "operators": [], "goals": [], "rolled_back": False}
        if self.cycle % 4 == 0:
            op = {"type": self.rng.choice(["unary", "binary"]),
                  "name": f"op_{self.cycle}", "cross_task": self.cycle % 8 == 0}
            self.ops.append(op)
            result["operators"] = [op]
        if self.cycle % 5 == 0:
            goal = {"complexity": self.cycle * 2, "feasible": True}
            self.goals.append(goal)
            result["goals"] = [goal]
        return result

    def evaluate(self):
        if self.trajectory == "improving":
            self.perf += self.rng.normal(0.05, 0.02)
        elif self.trajectory == "flat":
            self.perf += self.rng.normal(0.0, 0.001)
        elif self.trajectory == "degrading":
            self.perf -= self.rng.normal(0.03, 0.01)
        return {"fitness": self.perf, "score": self.perf}

    def get_state(self):
        return {
            "cycle": self.cycle,
            "goals": self.goals[-1:],
            "new_operators": [],
            "sandbox_violation": False,
        }

    def reset(self):
        self.cycle = 0
        self.perf = 1.0
        self.ops = []
        self.goals = []

    def as_interface(self, name="MockSystem"):
        return SystemInterface(
            name=name,
            modify_fn=self.modify,
            evaluate_fn=self.evaluate,
            get_state_fn=self.get_state,
            reset_fn=self.reset,
        )


# ── Axis 1: Self-Modification Depth ────────────────────────────────────────

class TestSelfModificationDepth:

    def test_returns_axis_result(self):
        sys = MockRSISystem()
        iface = sys.as_interface()
        axis = SelfModificationDepth()
        result = axis.evaluate(iface, max_cycles=15)
        assert isinstance(result, AxisResult)
        assert 0.0 <= result.score <= 1.0

    def test_metrics_present(self):
        sys = MockRSISystem()
        iface = sys.as_interface()
        axis = SelfModificationDepth()
        result = axis.evaluate(iface, max_cycles=20)
        assert "max_level" in result.metrics
        assert "stable_level" in result.metrics
        assert "rollback_rate" in result.metrics
        assert "total_modifications" in result.metrics

    def test_deeper_modification_higher_score(self):
        """A system that reaches higher levels should score higher."""
        # System with rapid level progression
        sys_deep = MockRSISystem(seed=42)
        iface_deep = sys_deep.as_interface()
        axis = SelfModificationDepth()
        result_deep = axis.evaluate(iface_deep, max_cycles=30)

        # System with minimal cycles (less chance to reach high levels)
        sys_shallow = MockRSISystem(seed=42)
        iface_shallow = sys_shallow.as_interface()
        result_shallow = axis.evaluate(iface_shallow, max_cycles=3)

        assert result_deep.score >= result_shallow.score


# ── Axis 2: Trajectory Quality ─────────────────────────────────────────────

class TestTrajectoryQuality:

    def test_improving_trajectory_scores_positive(self):
        sys = MockRSISystem(trajectory="improving")
        iface = sys.as_interface()
        axis = TrajectoryQuality()
        result = axis.evaluate(iface, max_cycles=20)
        assert result.score > 0.0

    def test_flat_trajectory(self):
        sys = MockRSISystem(trajectory="flat")
        iface = sys.as_interface()
        axis = TrajectoryQuality()
        result = axis.evaluate(iface, max_cycles=20)
        assert 0.0 <= result.score <= 1.0

    def test_metrics_present(self):
        sys = MockRSISystem(trajectory="improving")
        iface = sys.as_interface()
        axis = TrajectoryQuality()
        result = axis.evaluate(iface, max_cycles=20)
        assert "improvement_slope" in result.metrics
        assert "slope_pvalue" in result.metrics
        assert "acceleration" in result.metrics
        assert "stagnation_escape_rate" in result.metrics

    def test_trajectory_stored_in_details(self):
        sys = MockRSISystem(trajectory="improving")
        iface = sys.as_interface()
        axis = TrajectoryQuality()
        result = axis.evaluate(iface, max_cycles=15)
        assert "trajectory" in result.details
        assert len(result.details["trajectory"]) == 15


# ── Axis 3: Operator Discovery ─────────────────────────────────────────────

class TestOperatorDiscovery:

    def test_returns_valid_score(self):
        sys = MockRSISystem()
        iface = sys.as_interface()
        axis = OperatorDiscovery()
        result = axis.evaluate(iface, max_cycles=20)
        assert 0.0 <= result.score <= 1.0

    def test_metrics_present(self):
        sys = MockRSISystem()
        iface = sys.as_interface()
        axis = OperatorDiscovery()
        result = axis.evaluate(iface, max_cycles=20)
        assert "discovery_rate" in result.metrics
        assert "acceptance_selectivity" in result.metrics
        assert "cross_task_transfer" in result.metrics
        assert "diversity_index" in result.metrics


# ── Axis 4: Meta-Adaptation ────────────────────────────────────────────────

class TestMetaAdaptation:

    def test_returns_valid_score(self):
        sys = MockRSISystem()
        iface = sys.as_interface()
        axis = MetaAdaptation()
        result = axis.evaluate(iface, max_cycles=30)
        assert 0.0 <= result.score <= 1.0

    def test_custom_shift_schedule(self):
        sys = MockRSISystem()
        iface = sys.as_interface()
        axis = MetaAdaptation(shift_schedule=[(0, "normal"), (10, "shifted")])
        result = axis.evaluate(iface, max_cycles=20)
        assert "adaptation_gain" in result.metrics
        assert "recovery_time" in result.metrics
        assert "shock_resilience" in result.metrics
        assert "transfer_efficiency" in result.metrics


# ── Axis 5: Safety & Stability ─────────────────────────────────────────────

class TestSafetyStability:

    def test_safe_system_scores_high(self):
        sys = MockRSISystem(trajectory="improving")
        iface = sys.as_interface()
        axis = SafetyStability()
        result = axis.evaluate(iface, max_cycles=20)
        assert result.score > 0.3

    def test_metrics_present(self):
        sys = MockRSISystem()
        iface = sys.as_interface()
        axis = SafetyStability()
        result = axis.evaluate(iface, max_cycles=15)
        assert "crash_free_ratio" in result.metrics
        assert "rollback_rate" in result.metrics
        assert "pareto_preservation" in result.metrics
        assert "sandbox_violations" in result.metrics


# ── Axis 6: Goal Generation ────────────────────────────────────────────────

class TestGoalGeneration:

    def test_returns_valid_score(self):
        sys = MockRSISystem()
        iface = sys.as_interface()
        axis = GoalGeneration()
        result = axis.evaluate(iface, max_cycles=20)
        assert 0.0 <= result.score <= 1.0

    def test_metrics_present(self):
        sys = MockRSISystem()
        iface = sys.as_interface()
        axis = GoalGeneration()
        result = axis.evaluate(iface, max_cycles=20)
        assert "total_goals" in result.metrics
        assert "unique_goals" in result.metrics
        assert "goal_novelty" in result.metrics
        assert "feasibility_rate" in result.metrics


# ── Cross-axis consistency ──────────────────────────────────────────────────

class TestCrossAxisConsistency:

    def test_all_axes_return_bounded_scores(self):
        """Every axis must return score in [0, 1]."""
        sys = MockRSISystem()
        iface = sys.as_interface()
        axes = [
            SelfModificationDepth(),
            TrajectoryQuality(),
            OperatorDiscovery(),
            MetaAdaptation(),
            SafetyStability(),
            GoalGeneration(),
        ]
        for axis in axes:
            # Reset system between axes
            sys.reset()
            result = axis.evaluate(iface, max_cycles=10)
            assert 0.0 <= result.score <= 1.0, \
                f"{axis.name} returned score {result.score} outside [0, 1]"
            assert isinstance(result.metrics, dict), \
                f"{axis.name} metrics is not a dict"
