"""Axis 3: Operator Discovery Rate (ODR)."""

import numpy as np
from collections import Counter
from rsi_bench.axes.base import AxisBase
from rsi_bench.core import AxisResult


class OperatorDiscovery(AxisBase):
      name = "Operator Discovery Rate (ODR)"

    def __init__(self, rng=None):
              super().__init__(rng=rng)

    def evaluate(self, system, max_cycles=50, **kwargs):
              proposed, accepted, op_types, cross_task = [], [], [], 0
              pb = system.evaluate_fn()
              if isinstance(pb, dict): pb = pb.get("fitness", pb.get("score", 0.0))
                        gains = []
        for _ in range(max_cycles):
                      mod = system.modify_fn()
                      st = system.get_state_fn()
                      ops = []
                      if isinstance(mod, dict): ops = mod.get("operators", [])
                                    if isinstance(st, dict): ops.extend(st.get("new_operators", []))
                                                  for op in ops:
                                                                    proposed.append(op)
                                                                    pa = system.evaluate_fn()
                                                                    if isinstance(pa, dict): pa = pa.get("fitness", pa.get("score", 0.0))
                                                                                      if float(pa) >= float(pb) - 1e-10:
                                                                                                            accepted.append(op)
                                                                                                            gains.append(float(pa) - float(pb))
                                                                                                            tp = op.get("type", "unknown") if isinstance(op, dict) else "generic"
                                                                                                            op_types.append(tp)
                                                                                                            if isinstance(op, dict) and op.get("cross_task"): cross_task += 1
                                                                                                                              pb = float(pa)
        tp, ta = max(len(proposed), 1), len(accepted)
        dr = ta / max_cycles
        sel = ta / tp
        xfer = cross_task / max(ta, 1)
        if op_types:
                      c = Counter(op_types)
            p = np.array(list(c.values()), dtype=float); p /= p.sum()
            div = float(-np.sum(p * np.log(p + 1e-10)))
else: div = 0.0
        imp = min(sum(gains) / max(abs(float(pb)), 1e-10), 1.0) if gains else 0.0
        score = 0.25*min(dr,1) + 0.20*sel + 0.20*xfer + 0.15*min(div,1) + 0.20*max(imp,0)
        return AxisResult(axis_name=self.name, score=score, metrics={
                      "discovery_rate": float(dr), "acceptance_selectivity": float(sel),
                      "cross_task_transfer": float(xfer), "diversity_index": float(div),
                      "improvement_from_discovered": float(imp),
                      "total_proposed": len(proposed), "total_accepted": ta})
