"""
Axis 6: Autonomous Goal Generation (AGG)
==========================================
Evaluates the system's capacity to define its own meaningful
objectives without external reward signals.
"""

import numpy as np
from rsi_bench.axes.base import AxisBase
from rsi_bench.core import AxisResult


class GoalGeneration(AxisBase):
    """Evaluates autonomous goal generation capability."""

    name = "Autonomous Goal Generation (AGG)"

    def __init__(self, rng=None):
        super().__init__(rng=rng)

    def evaluate(self, system, max_cycles=50, **kwargs):
        """Evaluate autonomous goal generation."""
        generated_goals = []
        feasible_goals = 0
        solved_goals = 0
        goal_complexities = []
        goal_solution_correlations = []

        for cycle in range(max_cycles):
            mod_result = system.modify_fn()
            state = system.get_state_fn()

            # Extract goals from system state
            goals = self._extract_goals(state, mod_result)

            for goal in goals:
                generated_goals.append(goal)
                complexity = self._measure_complexity(goal)
                goal_complexities.append(complexity)

                # Check feasibility
                is_feasible = self._check_feasibility(goal, state)
                if is_feasible:
                    feasible_goals += 1

                # Check if system can solve its own goals
                perf = system.evaluate_fn()
                if isinstance(perf, dict):
                    perf_val = perf.get("fitness", perf.get("score", 0.0))
                else:
                    perf_val = float(perf)

                if perf_val > 0:
                    solved_goals += 1
                    goal_solution_correlations.append(perf_val * complexity)

        total_goals = len(generated_goals)
        unique_goals = len(set(str(g) for g in generated_goals))

        # Metrics
        novelty = unique_goals / max(total_goals, 1)
        feasibility_rate = feasible_goals / max(total_goals, 1)
        solve_rate = solved_goals / max(total_goals, 1)

        # Emergent complexity growth
        if len(goal_complexities) >= 4:
            first_half = np.mean(goal_complexities[:len(goal_complexities)//2])
            second_half = np.mean(goal_complexities[len(goal_complexities)//2:])
            complexity_growth = (second_half - first_half) / max(first_half, 1e-10)
        else:
            complexity_growth = 0.0
        complexity_score = 1.0 / (1.0 + np.exp(-2 * complexity_growth))

        # Goal-solution alignment
        if goal_solution_correlations:
            alignment = np.mean(goal_solution_correlations)
            alignment_score = min(abs(alignment), 1.0)
        else:
            alignment_score = 0.0

        # Curriculum quality (are goals progressively harder?)
        if len(goal_complexities) >= 4:
            diffs = np.diff(goal_complexities)
            curriculum_monotonicity = np.mean(diffs > 0)
        else:
            curriculum_monotonicity = 0.0

        # Composite score
        score = (0.25 * novelty + 0.20 * feasibility_rate +
                 0.20 * complexity_score + 0.15 * alignment_score +
                 0.20 * curriculum_monotonicity)

        return AxisResult(
            axis_name=self.name,
            score=score,
            metrics={
                "total_goals": total_goals,
                "unique_goals": unique_goals,
                "goal_novelty": float(novelty),
                "feasibility_rate": float(feasibility_rate),
                "solve_rate": float(solve_rate),
                "complexity_growth": float(complexity_growth),
                "alignment_score": float(alignment_score),
                "curriculum_monotonicity": float(curriculum_monotonicity),
            },
            details={
                "goal_complexities": goal_complexities,
                "sample_goals": [str(g) for g in generated_goals[:10]],
            },
        )

    def _extract_goals(self, state, mod_result):
        """Extract self-generated goals from system state."""
        goals = []
        if isinstance(state, dict):
            goals.extend(state.get("goals", []))
            goals.extend(state.get("objectives", []))
            goals.extend(state.get("generated_tasks", []))
        if isinstance(mod_result, dict):
            goals.extend(mod_result.get("goals", []))
        if not goals and mod_result is not None:
            goals = [mod_result]
        return goals

    def _measure_complexity(self, goal):
        """Measure structural complexity of a goal."""
        if isinstance(goal, dict):
            return len(str(goal))
        if isinstance(goal, (list, tuple)):
            return sum(self._measure_complexity(g) for g in goal)
        return len(str(goal))

    def _check_feasibility(self, goal, state):
        """Check if a goal is feasible given current system state."""
        if goal is None:
            return False
        if isinstance(goal, dict) and goal.get("feasible") is not None:
            return bool(goal["feasible"])
        return True
