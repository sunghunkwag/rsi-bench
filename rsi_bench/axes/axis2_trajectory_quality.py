"""Axis 2: Improvement Trajectory Quality (ITQ)."""

import numpy as np
from scipy import stats as sp_stats
from rsi_bench.axes.base import AxisBase
from rsi_bench.core import AxisResult


class TrajectoryQuality(AxisBase):
      name = "Improvement Trajectory Quality (ITQ)"

    def __init__(self, min_data_points=15, bootstrap_samples=5000, rng=None):
              super().__init__(rng=rng)
              self.min_data_points = min_data_points
              self.bootstrap_samples = bootstrap_samples

    def evaluate(self, system, max_cycles=50, **kwargs):
              trajectory = []
              for cycle in range(max_cycles):
                            system.modify_fn()
                            perf = system.evaluate_fn()
                            if isinstance(perf, dict):
                                              perf = perf.get("fitness", perf.get("score", 0.0))
                                          trajectory.append(float(perf))
                        t = np.arange(len(trajectory), dtype=float)
        y = np.array(trajectory, dtype=float)
        if len(y) < 2:
                      return AxisResult(axis_name=self.name, score=0.0, metrics={})
                  res = sp_stats.linregress(t, y)
        slope, pval = res.slope, res.pvalue
        accel = float(np.mean(np.diff(y, n=2))) if len(y) >= 3 else 0.0
        stag, esc, in_s = 0, 0, False
        for i in range(1, len(y)):
                      if abs(y[i] - y[i-1]) < 1e-8:
                                        if not in_s: stag += 1; in_s = True
                      elif in_s: esc += 1; in_s = False
                                se = esc / max(stag, 1)
        fir = y[-1] / max(abs(y[0]), 1e-10)
        auc = float(np.trapz(y, t)) / max(len(y), 1)
        auc_n = auc / max(abs(np.max(y)), 1e-10)
        ss = 1.0 / (1.0 + np.exp(-10 * slope))
        ac = 1.0 / (1.0 + np.exp(-5 * accel))
        fs = min(max(fir - 1.0, 0.0) / 2.0, 1.0)
        score = 0.30*ss + 0.20*ac + 0.20*min(se, 1.0) + 0.15*fs + 0.15*min(auc_n, 1.0)
        return AxisResult(axis_name=self.name, score=score, metrics={
                      "improvement_slope": float(slope), "slope_pvalue": float(pval),
                      "acceleration": float(accel), "stagnation_escape_rate": float(se),
                      "final_initial_ratio": float(fir), "area_under_curve": float(auc_n)},
                                      details={"trajectory": trajectory})
