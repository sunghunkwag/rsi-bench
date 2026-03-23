"""
RSI-Bench Core Orchestrator
============================
Central benchmark runner that coordinates all six evaluation axes.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from rsi_bench.scoring import UnifiedScorer


@dataclass
class SystemInterface:
    """Interface contract for systems under evaluation."""
    name: str
    modify_fn: Callable
    evaluate_fn: Callable
    get_state_fn: Callable
    reset_fn: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AxisResult:
    """Result from a single evaluation axis."""
    axis_name: str
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def __post_init__(self):
        self.score = float(np.clip(self.score, 0.0, 1.0))


@dataclass
class BenchmarkResults:
    """Aggregated results from all evaluation axes."""
    system_name: str
    axis_results: Dict[str, AxisResult] = field(default_factory=dict)
    composite_score: float = 0.0
    total_duration: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"RSI-Bench Results: {self.system_name}",
            "=" * 60,
        ]
        for name, result in self.axis_results.items():
            lines.append(f"  {name:40s} {result.score:.4f}")
        lines.append("-" * 60)
        lines.append(f"  {'Composite RSI Score':40s} {self.composite_score:.4f}")
        lines.append(f"  {'Total Duration':40s} {self.total_duration:.1f}s")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_json(self, path: str):
        data = {
            "system_name": self.system_name,
            "composite_score": self.composite_score,
            "total_duration": self.total_duration,
            "config": self.config,
            "axes": {
                name: {
                    "score": r.score,
                    "metrics": r.metrics,
                    "duration": r.duration_seconds,
                }
                for name, r in self.axis_results.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class RSIBenchmark:
    """
    Central benchmark orchestrator for RSI evaluation.
    Coordinates all six evaluation axes and produces unified scores.
    """

    AXIS_NAMES = [
        "Self-Modification Depth (SMD)",
        "Improvement Trajectory Quality (ITQ)",
        "Operator Discovery Rate (ODR)",
        "Meta-Adaptation Speed (MAS)",
        "Safety & Stability (SSM)",
        "Autonomous Goal Generation (AGG)",
    ]

    def __init__(self, axes=None, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.system = None
        self.scorer = UnifiedScorer()
        self._active_axes = axes or ["smd", "itq", "odr", "mas", "ssm", "agg"]

    def register_system(self, name, modify_fn, evaluate_fn, get_state_fn,
                        reset_fn=None, **metadata):
        """Register an RSI system for evaluation."""
        self.system = SystemInterface(
            name=name, modify_fn=modify_fn, evaluate_fn=evaluate_fn,
            get_state_fn=get_state_fn, reset_fn=reset_fn, metadata=metadata,
        )

    def run(self, max_cycles=50, seed=None, verbose=True):
        """Run the full benchmark suite."""
        if self.system is None:
            raise RuntimeError("No system registered. Call register_system() first.")
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        results = BenchmarkResults(
            system_name=self.system.name,
            config={"max_cycles": max_cycles, "seed": seed or self.seed},
        )
        total_start = time.time()

        from rsi_bench.axes import (
            SelfModificationDepth, TrajectoryQuality, OperatorDiscovery,
            MetaAdaptation, SafetyStability, GoalGeneration,
        )

        axis_map = {
            "smd": ("Self-Modification Depth (SMD)", SelfModificationDepth),
            "itq": ("Improvement Trajectory Quality (ITQ)", TrajectoryQuality),
            "odr": ("Operator Discovery Rate (ODR)", OperatorDiscovery),
            "mas": ("Meta-Adaptation Speed (MAS)", MetaAdaptation),
            "ssm": ("Safety & Stability (SSM)", SafetyStability),
            "agg": ("Autonomous Goal Generation (AGG)", GoalGeneration),
        }

        for axis_key in self._active_axes:
            if axis_key not in axis_map:
                continue
            axis_name, axis_cls = axis_map[axis_key]
            if verbose:
                print(f"[RSI-Bench] Evaluating {axis_name}...")
            evaluator = axis_cls(rng=self.rng)
            t0 = time.time()
            axis_result = evaluator.evaluate(self.system, max_cycles=max_cycles)
            axis_result.duration_seconds = time.time() - t0
            results.axis_results[axis_name] = axis_result
            if verbose:
                print(f"  -> Score: {axis_result.score:.4f} ({axis_result.duration_seconds:.1f}s)")

        results.total_duration = time.time() - total_start
        results.composite_score = self.scorer.compute(results)
        if verbose:
            print(results.summary())
        return results

    def run_single_axis(self, axis, max_cycles=50, **kwargs):
        """Run a single evaluation axis."""
        if self.system is None:
            raise RuntimeError("No system registered.")
        from rsi_bench.axes import (
            SelfModificationDepth, TrajectoryQuality, OperatorDiscovery,
            MetaAdaptation, SafetyStability, GoalGeneration,
        )
        axis_map = {
            "smd": SelfModificationDepth, "itq": TrajectoryQuality,
            "odr": OperatorDiscovery, "mas": MetaAdaptation,
            "ssm": SafetyStability, "agg": GoalGeneration,
        }
        if axis not in axis_map:
            raise ValueError(f"Unknown axis '{axis}'. Choose from {list(axis_map)}")
        evaluator = axis_map[axis](rng=self.rng)
        return evaluator.evaluate(self.system, max_cycles=max_cycles, **kwargs)
