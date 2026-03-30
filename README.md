# RSI-Bench: Multi-Axis Benchmark for Recursive Self-Improvement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/sunghunkwag/rsi-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/sunghunkwag/rsi-bench/actions/workflows/tests.yml)

An open-source benchmark framework for evaluating Recursive Self-Improvement (RSI) capabilities in AI systems. RSI-Bench decomposes RSI into six measurable axes with statistical rigor, providing researchers with concrete tools to distinguish genuine self-improvement from metric gaming.

---

## Why This Exists

Recursive Self-Improvement is widely discussed in AI safety and capability research, but there is no standardized way to measure it. Existing benchmarks evaluate static capabilities — reasoning, coding, math — but none target the *dynamic capacity of a system to improve its own processes*.

This matters for safety: an RSI-like system can appear to improve while actually gaming evaluation metrics, hiding failures, or exploiting weak oversight. Without rigorous measurement tools, these failure modes remain invisible.

RSI-Bench provides:

- **Six orthogonal evaluation axes** covering the core dimensions of self-improvement
- **Statistical evaluation** with BCa bootstrap confidence intervals and convergence detection
- **Multi-objective Pareto analysis** for improvement trade-offs
- **Safety boundary verification** under self-modification
- **74 passing tests** verifying all components against the actual implementation

## The Six Evaluation Axes

| Axis | Measures | Key Metric |
|------|----------|------------|
| **Self-Modification Depth (SMD)** | How many meta-levels can the system modify? (L0–L4 hierarchy) | Max stable modification level |
| **Improvement Trajectory Quality (ITQ)** | How efficiently does performance improve over iterations? | Slope, acceleration, stagnation escape rate |
| **Operator Discovery Rate (ODR)** | Can the system synthesize novel, useful operators? | Discovery rate, cross-task transfer, diversity index |
| **Meta-Adaptation Speed (MAS)** | How quickly does the system adapt to distributional shifts? | Recovery time, adaptation gain, shock resilience |
| **Safety & Stability (SSM)** | Does self-modification preserve system integrity? | Crash-free ratio, Pareto preservation, sandbox violations |
| **Autonomous Goal Generation (AGG)** | Can the system define its own meaningful objectives? | Goal novelty, feasibility rate, curriculum quality |

Each axis returns a score in [0, 1]. The composite RSI score uses a weighted harmonic mean, which penalizes systems that score zero on any dimension — reflecting the principle that genuine RSI requires competence across all axes.

## Installation

```bash
git clone https://github.com/sunghunkwag/rsi-bench.git
cd rsi-bench
pip install -r requirements.txt
```

## Quick Start

```python
from rsi_bench.core import RSIBenchmark

# Initialize benchmark
bench = RSIBenchmark(seed=42)

# Register your RSI system — any object with these four functions
bench.register_system(
    name="MyRSISystem",
    modify_fn=my_system.self_modify,      # () -> modification result
    evaluate_fn=my_system.evaluate,        # () -> performance dict or scalar
    get_state_fn=my_system.get_state,      # () -> state dict
    reset_fn=my_system.reset,              # () -> reset (optional)
)

# Run all 6 axes
results = bench.run(max_cycles=50, seed=42)

# Inspect results
print(results.summary())
print(f"Composite RSI Score: {results.composite_score:.4f}")

# Export
results.to_json("rsi_bench_results.json")
```

Run a single axis:

```python
result = bench.run_single_axis("ssm", max_cycles=50)
print(f"Safety & Stability: {result.score:.4f}")
print(f"Crash-free ratio: {result.metrics['crash_free_ratio']:.4f}")
```

## System Interface Contract

Any system evaluated by RSI-Bench must expose four functions:

| Function | Signature | Returns |
|----------|-----------|---------|
| `modify_fn` | `() -> result` | Dict with modification metadata (level, operators, goals) |
| `evaluate_fn` | `() -> perf` | Dict `{"fitness": float}` or scalar float |
| `get_state_fn` | `() -> state` | Dict with current system state |
| `reset_fn` | `() -> None` | Reset to initial state (optional) |

See `tests/test_integration.py` for a complete `MockRSISystem` example.

## Statistical Framework

RSI-Bench includes a built-in statistical evaluation layer:

**Bootstrap Confidence Intervals** — BCa bootstrap (5000 samples default) with bias correction for all metrics:

```python
from rsi_bench.statistics import BootstrapCI

ci = BootstrapCI(n_samples=5000, confidence=0.95)
result = ci.compute(data)
# result: {"estimate", "ci_lower", "ci_upper", "std_error", "bias"}
```

**Convergence Detection** — O(1) memory phase classifier (`improving`, `exploring`, `converged`, `degrading`):

```python
from rsi_bench.statistics import ConvergenceDetector

detector = ConvergenceDetector(window_size=50)
for value in trajectory:
    state = detector.update(value)
    # state: {"state": "improving", "mean", "variance", "volatility", "trend"}
```

**Pareto Frontier Analysis** — Multi-objective trade-off analysis with hypervolume computation:

```python
from rsi_bench.statistics import ParetoAnalyzer

pa = ParetoAnalyzer(
    objective_directions={"performance": "maximize", "cost": "minimize"},
    reference_point={"performance": 0.0, "cost": 100.0},
)
pa.add_solution({"performance": 0.9, "cost": 50})
pa.add_solution({"performance": 0.7, "cost": 30})
report = pa.get_report()
# report: {"frontier_size", "hypervolume", "dominance_ratio"}
```

## Unified Scoring

The composite RSI score uses a weighted harmonic mean across all six axes:

```
RSI_composite = 6 / (w₁/SMD + w₂/ITQ + w₃/ODR + w₄/MAS + w₅/SSM + w₆/AGG)
```

Default weights are uniform. The harmonic mean ensures that a system scoring zero on any axis receives a near-zero composite — you cannot compensate for a missing capability by excelling at others.

```python
from rsi_bench.scoring import UnifiedScorer

scorer = UnifiedScorer(weights={"smd": 1.0, "itq": 1.0, "odr": 1.0,
                                 "mas": 1.0, "ssm": 1.0, "agg": 1.0})
composite = scorer.compute_from_dict(axis_scores)
```

## Extending RSI-Bench

### Custom Tasks

```python
from rsi_bench.tasks import TaskBase

class MyTask(TaskBase):
    def generate(self, seed=None):
        # Return (inputs, expected_outputs)
        ...
    def evaluate(self, system_output, expected):
        # Return scalar fitness
        ...
```

### Custom Axes

```python
from rsi_bench.axes.base import AxisBase
from rsi_bench.core import AxisResult

class MyAxis(AxisBase):
    name = "my_custom_axis"

    def evaluate(self, system, max_cycles=50, **kwargs):
        # Your evaluation logic here
        return AxisResult(axis_name=self.name, score=0.75, metrics={...})
```

See `examples/custom_axis_evaluation.py` for a complete working example.

## Project Structure

```
rsi-bench/
├── rsi_bench/
│   ├── core.py                 # RSIBenchmark orchestrator + SystemInterface
│   ├── scoring.py              # UnifiedScorer (harmonic/arithmetic/geometric)
│   ├── axes/
│   │   ├── base.py             # AxisBase abstract class
│   │   ├── axis1_modification_depth.py
│   │   ├── axis2_trajectory_quality.py
│   │   ├── axis3_operator_discovery.py
│   │   ├── axis4_meta_adaptation.py
│   │   ├── axis5_safety_stability.py
│   │   └── axis6_goal_generation.py
│   ├── statistics/
│   │   ├── bootstrap.py        # BCa bootstrap CI
│   │   ├── convergence.py      # O(1) phase detector
│   │   └── pareto.py           # Pareto frontier + hypervolume
│   ├── tasks/                  # Symbolic regression, program synthesis, etc.
│   └── utils/                  # Sandbox execution, structured logging
├── tests/                      # 74 tests (all passing)
├── examples/                   # Full benchmark run, custom axis demo
├── requirements.txt
└── setup.py
```

## Related Work

RSI-Bench draws on concepts from:

- Schmidhuber (2003) — Gödel Machines: self-referential optimizers
- Nivel et al. (2013) — AERA: replicating organisms for constructivist AI
- Steunebrink et al. (2016) — Toward practical RSI: formal verification approaches
- Skalse et al. (2022) — Defining and characterizing reward hacking
- Hubinger et al. (2019) — Risks from learned optimization: deceptive alignment
- Ando (2025) — Noise-to-Meaning RSI formal model

## Next Steps

RSI-Bench provides the measurement infrastructure. The planned next phase is a **deceptive improvement detection module** — adversarial tasks specifically designed to test whether an RSI-like agent is genuinely improving versus gaming its evaluation setup. Target failure modes include metric gaming (proxy vs. true objective divergence), pseudo-improvement (performance increase with generalization degradation), and oversight evasion (behavioral inconsistency when monitored vs. unmonitored).

## Citation

```bibtex
@software{rsi_bench_2026,
  author = {Kwag, Sunghun},
  title = {RSI-Bench: Multi-Axis Benchmark for Recursive Self-Improvement},
  year = {2026},
  url = {https://github.com/sunghunkwag/rsi-bench},
  note = {Open-source benchmark framework for evaluating RSI in AI systems}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
