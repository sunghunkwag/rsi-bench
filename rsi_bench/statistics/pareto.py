"""Pareto Frontier Analysis for RSI-Bench."""

import numpy as np
from sortedcontainers import SortedList
from typing import Dict, List


class ParetoAnalyzer:
    """Multi-objective Pareto frontier analyzer."""

    def __init__(self, objective_directions: Dict[str, str], reference_point: Dict[str, float]):
        self.directions = objective_directions
        self.reference_point = reference_point
        self.solutions = []
        self.frontier = SortedList(key=lambda s: tuple(s[k] for k in sorted(self.directions)))

    def add_solution(self, solution: Dict[str, float]) -> bool:
        """Add a solution; return True if on Pareto frontier."""
        self.solutions.append(solution.copy())
        if self._is_dominated(solution):
            return False
        self.frontier = SortedList(key=lambda s: tuple(s[k] for k in sorted(self.directions)))
        for s in self.solutions:
            if not self._is_dominated(s):
                self.frontier.add(s)
        return solution in self.frontier

    def _is_dominated(self, candidate: Dict) -> bool:
        for s in self.solutions:
            if s is candidate:
                continue
            dom = True
            strict = False
            for obj, d in self.directions.items():
                cv, sv = candidate.get(obj, 0), s.get(obj, 0)
                if d == "maximize":
                    if sv < cv:
                        dom = False
                        break
                    if sv > cv:
                        strict = True
                else:
                    if sv > cv:
                        dom = False
                        break
                    if sv < cv:
                        strict = True
            if dom and strict:
                return True
        return False

    def compute_hypervolume(self) -> float:
        if not self.frontier:
            return 0.0
        objs = sorted(self.directions.keys())
        if len(objs) == 2:
            return self._hv2d(objs)
        return self._hv_approx(objs)

    def _hv2d(self, objs) -> float:
        a_k, b_k = objs
        pts = []
        for s in self.frontier:
            a = s[a_k] if self.directions[a_k] == "maximize" else -s[a_k]
            b = s[b_k] if self.directions[b_k] == "maximize" else -s[b_k]
            pts.append((a, b))
        ra = self.reference_point[a_k] if self.directions[a_k] == "maximize" else -self.reference_point[a_k]
        rb = self.reference_point[b_k] if self.directions[b_k] == "maximize" else -self.reference_point[b_k]
        pts.sort(key=lambda p: -p[0])
        hv, pb = 0.0, rb
        for a, b in pts:
            if a > ra and b > pb:
                hv += (a - ra) * (b - pb)
                pb = b
        return float(hv)

    def _hv_approx(self, objs) -> float:
        rng = np.random.default_rng(42)
        bounds = {o: (min(s[o] for s in self.frontier), max(s[o] for s in self.frontier)) for o in objs}
        n, cnt = 10000, 0
        for _ in range(n):
            pt = {o: rng.uniform(*bounds[o]) for o in objs}
            for s in self.frontier:
                if all((s[o] >= pt[o] if self.directions[o] == "maximize" else s[o] <= pt[o]) for o in objs):
                    cnt += 1
                    break
        vol = 1.0
        for o in objs:
            vol *= bounds[o][1] - bounds[o][0]
        return float(vol * cnt / n)

    def get_report(self) -> Dict:
        hv = self.compute_hypervolume()
        return {
            "frontier_size": len(self.frontier),
            "total_solutions": len(self.solutions),
            "metrics": {
                "frontier_size": len(self.frontier),
                "hypervolume": hv,
                "dominance_ratio": len(self.frontier) / max(len(self.solutions), 1),
            },
            "frontier": [dict(s) for s in self.frontier],
        }

    def reset(self):
        self.solutions.clear()
        self.frontier.clear()
