"""Axis 6: Autonomous Goal Generation (AGG), evaluation protocol v2.

Only a benchmark-owned goal verifier can establish feasibility, completion,
identity, and difficulty. Generic fitness and goal metadata are not evidence.
"""

import copy
import json
import math

import numpy as np

from rsi_bench.axes.base import AxisBase
from rsi_bench.core import AxisResult, EVALUATION_PROTOCOL


class GoalGeneration(AxisBase):
    """Evaluate explicitly proposed goals against independent task checks."""

    name = "Autonomous Goal Generation (AGG)"

    def __init__(self, rng=None, goal_verifier=None):
        super().__init__(rng=rng)
        if goal_verifier is not None and not callable(goal_verifier):
            raise ValueError("goal_verifier must be callable")
        self.goal_verifier = goal_verifier

    def evaluate(self, system, max_cycles=50, **kwargs):
        records, identities, unverified = {}, {}, set()
        completed_difficulties, samples = [], []
        failures = 0

        for cycle in range(max_cycles):
            modification = system.modify_fn()
            state = system.get_state_fn()
            seen = set()
            for index, goal in enumerate(self._extract_goals(state, modification)):
                try:
                    goal = self._json_goal(goal)
                    key = json.dumps(goal, sort_keys=True, allow_nan=False,
                                     separators=(",", ":"))
                    if goal is None:
                        raise ValueError("A goal must not be null")
                except (ValueError, TypeError, RecursionError):
                    unverified.add("invalid:{}:{}".format(cycle, index))
                    failures += 1
                    continue
                if key in seen:
                    continue
                seen.add(key)
                unverified.add(key)
                previous_id = identities.get(key)
                if previous_id is not None and records[previous_id]["solved"]:
                    unverified.discard(key)
                    continue
                if len(samples) < 10:
                    samples.append(key)
                if self.goal_verifier is None:
                    continue
                try:
                    assessment = self.goal_verifier(copy.deepcopy(goal), copy.deepcopy(state))
                    self._validate_assessment(assessment)
                    identity = assessment["goal_id"]
                    if previous_id is not None and identity != previous_id:
                        raise ValueError("Verifier changed the identity of an existing goal")
                    if identity in records and not math.isclose(
                            records[identity]["complexity"], assessment["complexity"],
                            rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError("Verifier changed the difficulty of an existing goal")
                except Exception:
                    failures += 1
                    if previous_id is not None:
                        unverified.discard(key)
                    continue
                identities[key] = identity
                unverified.discard(key)
                record = records.setdefault(identity, {
                    "feasible": False, "solved": False,
                    "complexity": float(assessment["complexity"]),
                })
                record["feasible"] |= assessment["feasible"]
                if assessment["solved"] and not record["solved"]:
                    record["solved"] = True
                    completed_difficulties.append(record["complexity"])

        total = len(records) + len(unverified)
        feasible = sum(r["feasible"] for r in records.values())
        solved = sum(r["solved"] for r in records.values())
        # Repeating one verified identity throughout the cycle budget is not
        # sustained goal discovery. Aliases cannot inflate the numerator.
        novelty = len(records) / max(max_cycles, total, 1)
        feasibility_rate = feasible / max(total, 1)
        solve_rate = solved / max(total, 1)
        alignment = solved / max(feasible, 1)
        growth, curriculum = 0.0, 0.0
        if len(completed_difficulties) >= 4:
            half = len(completed_difficulties) // 2
            first = float(np.mean(completed_difficulties[:half]))
            second = float(np.mean(completed_difficulties[half:]))
            growth = (second - first) / max(first, 1e-10)
            curriculum = float(np.mean(np.diff(completed_difficulties) > 0))
        complexity_score = 1.0 / (1.0 + np.exp(-2 * growth))
        # Gate every component on verified goal completion.
        score = solve_rate * (
            0.25 * novelty + 0.20 * feasibility_rate + 0.20 * complexity_score +
            0.15 * alignment + 0.20 * curriculum)
        return AxisResult(
            axis_name=self.name, score=score,
            metrics={
                "total_goals": total, "unique_goals": len(records),
                "verified_goals": len(records), "unverified_goals": len(unverified),
                "feasible_goals": feasible, "solved_goals": solved,
                "verification_failures": failures,
                "goal_novelty": float(novelty),
                "feasibility_rate": float(feasibility_rate),
                "solve_rate": float(solve_rate),
                "complexity_growth": float(growth),
                "alignment_score": float(alignment),
                "curriculum_monotonicity": float(curriculum),
            },
            details={
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "goal_verification_available": self.goal_verifier is not None,
                "goal_complexities": completed_difficulties,
                "sample_goals": samples,
            },
        )

    @staticmethod
    def _extract_goals(state, modification):
        goals = []
        for source, fields in ((state, ("goals", "objectives", "generated_tasks")),
                               (modification, ("goals",))):
            if isinstance(source, dict):
                for field in fields:
                    value = source.get(field, [])
                    if isinstance(value, (list, tuple)):
                        goals.extend(value)
        return goals

    @staticmethod
    def _json_goal(value):
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if type(value) is list:
            return [GoalGeneration._json_goal(v) for v in value]
        if type(value) is dict and all(type(k) is str for k in value):
            return {k: GoalGeneration._json_goal(v) for k, v in value.items()}
        raise TypeError("Goals must contain JSON values")

    @staticmethod
    def _validate_assessment(value):
        if type(value) is not dict or set(value) != {"goal_id", "feasible", "solved", "complexity"}:
            raise ValueError("Goal verifier must return an explicit assessment")
        if type(value["goal_id"]) is not str or not value["goal_id"].strip():
            raise ValueError("Goal verifier must provide a stable, nonempty identity")
        if type(value["feasible"]) is not bool or type(value["solved"]) is not bool:
            raise ValueError("Goal verification flags must be booleans")
        difficulty = value["complexity"]
        if type(difficulty) not in (int, float) or not 0 <= difficulty <= 1:
            raise ValueError("Verified complexity must be finite and between zero and one")
        if value["solved"] and not value["feasible"]:
            raise ValueError("An infeasible goal cannot be solved")
