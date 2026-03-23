"""Tests for all evaluation axes."""
import pytest
import numpy as np
from rsi_bench.axes.axis1_modification_depth import Axis1ModificationDepth
from rsi_bench.axes.axis2_trajectory_quality import Axis2TrajectoryQuality
from rsi_bench.axes.axis3_operator_discovery import Axis3OperatorDiscovery
from rsi_bench.axes.axis4_meta_adaptation import Axis4MetaAdaptation
from rsi_bench.axes.axis5_safety_stability import Axis5SafetyStability
from rsi_bench.axes.axis6_goal_generation import Axis6GoalGeneration


class TestAxis1:
      def test_returns_score(self):
                axis = Axis1ModificationDepth()
                data = {"modifications": [{"depth": i, "success": True} for i in range(5)]}
                result = axis.evaluate(data)
                assert "score" in result
                assert 0.0 <= result["score"] <= 1.0


class TestAxis2:
      def test_improving_trajectory(self):
                axis = Axis2TrajectoryQuality()
                data = {"trajectory": [0.1 * i for i in range(20)]}
                result = axis.evaluate(data)
                assert result["score"] > 0.0

      def test_flat_trajectory(self):
                axis = Axis2TrajectoryQuality()
                data = {"trajectory": [0.5] * 20}
                result = axis.evaluate(data)
                assert 0.0 <= result["score"] <= 1.0


class TestAxis3:
      def test_diverse_operators(self):
                axis = Axis3OperatorDiscovery()
                data = {
                    "operators_discovered": ["op1", "op2", "op3", "op4"],
                    "operator_usage": {"op1": 10, "op2": 8, "op3": 6, "op4": 4},
                    "performance_gains": {"op1": 0.1, "op2": 0.05, "op3": 0.15, "op4": 0.02},
                }
                result = axis.evaluate(data)
                assert result["score"] > 0.0


class TestAxis4:
      def test_adaptation(self):
                axis = Axis4MetaAdaptation()
                data = {
                    "phase_performance": {"normal": [0.8, 0.82], "shifted": [0.5, 0.7], "novel": [0.3, 0.6]},
                    "recovery_steps": [5, 3],
                }
                result = axis.evaluate(data)
                assert 0.0 <= result["score"] <= 1.0


class TestAxis5:
      def test_safe_system(self):
                axis = Axis5SafetyStability()
                data = {"safety_violations": 0, "stability_scores": [0.95, 0.93, 0.97]}
                result = axis.evaluate(data)
                assert result["score"] > 0.5


class TestAxis6:
      def test_goal_generation(self):
                axis = Axis6GoalGeneration()
                data = {"goals_generated": ["g1", "g2"], "goal_quality_scores": [0.8, 0.7]}
                result = axis.evaluate(data)
                assert 0.0 <= result["score"] <= 1.0
