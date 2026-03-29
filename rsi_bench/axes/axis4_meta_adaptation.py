"""Axis 4: Meta-Adaptation Speed (MAS)."""

import numpy as np
from rsi_bench.axes.base import AxisBase
from rsi_bench.core import AxisResult


class MetaAdaptation(AxisBase):
    name = "Meta-Adaptation Speed (MAS)"

    def __init__(self, shift_schedule=None, rng=None):
        super().__init__(rng=rng)
        self.shift_schedule = shift_schedule or [(0, "normal"), (20, "shifted"), (40, "novel")]

    def evaluate(self, system, max_cycles=50, **kwargs):
        phases = self._build_phases(max_cycles)
        perfs, phase_perfs = [], {}
        for cycle in range(max_cycles):
            ph = phases[cycle]
            system.modify_fn()
            p = system.evaluate_fn()
            if isinstance(p, dict):
                p = p.get("fitness", p.get("score", 0.0))
            p = float(p)
            perfs.append(p)
            phase_perfs.setdefault(ph, []).append(p)
        pre = phase_perfs.get("normal", perfs[:max_cycles // 3])
        post = phase_perfs.get("shifted", perfs[max_cycles // 3:2 * max_cycles // 3])
        novel = phase_perfs.get("novel", perfs[2 * max_cycles // 3:])
        pm = np.mean(pre) if pre else 0.0
        if post and len(post) > 1:
            rt = max_cycles
            for i, v in enumerate(post):
                if v >= pm * 0.9:
                    rt = i + 1
                    break
            ag = (np.mean(post[-5:]) - post[0]) / max(abs(pm), 1e-10)
            sr = np.mean(post[:3]) / max(pm, 1e-10)
        else:
            rt, ag, sr = max_cycles, 0.0, 0.0
        te = np.mean(novel) / max(pm, 1e-10) if novel and pre else 0.0
        rts = 1.0 - min(rt / max(len(post) if post else 1, 1), 1.0)
        ags = 1.0 / (1.0 + np.exp(-3 * ag))
        srs = min(max(sr, 0.0), 1.0)
        tes = min(max(te, 0.0), 1.0)
        score = 0.30 * ags + 0.25 * rts + 0.25 * srs + 0.20 * tes
        return AxisResult(
            axis_name=self.name, score=score,
            metrics={
                "adaptation_gain": float(ag),
                "recovery_time": int(rt),
                "shock_resilience": float(sr),
                "transfer_efficiency": float(te),
            },
            details={"trajectory": perfs, "phases": phases},
        )

    def _build_phases(self, mc):
        ph = ["normal"] * mc
        for s, n in sorted(self.shift_schedule, key=lambda x: x[0]):
            for i in range(s, mc):
                ph[i] = n
        return ph
