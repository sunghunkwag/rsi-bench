"""
Bootstrap Confidence Intervals for RSI-Bench
=============================================
BCa (Bias-Corrected and Accelerated) bootstrap with O(n) memory.
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, List


class BootstrapCI:
      """BCa Bootstrap Confidence Interval estimator."""

    def __init__(self, n_samples: int = 5000, confidence: float = 0.95, seed: int = 42):
              self.n_samples = n_samples
              self.confidence = confidence
              self.rng = np.random.default_rng(seed)

    def compute(self, data, statistic=None) -> Dict:
              """Compute BCa bootstrap CI for the given data."""
              data = np.asarray(data, dtype=float)
              if len(data) < 2:
                            val = float(data[0]) if len(data) == 1 else 0.0
                            return {"estimate": val, "ci_lower": val, "ci_upper": val,
                                    "std_error": 0.0, "n_samples": len(data)}
                        if statistic is None:
                                      statistic = np.mean
                                  observed = float(statistic(data))
        n = len(data)
        boot_stats = np.empty(self.n_samples)
        for i in range(self.n_samples):
                      idx = self.rng.integers(0, n, size=n)
                      boot_stats[i] = statistic(data[idx])
                  prop_less = np.mean(boot_stats < observed)
        z0 = stats.norm.ppf(max(min(prop_less, 1 - 1e-10), 1e-10))
        jackknife_stats = np.empty(n)
        for i in range(n):
                      jack_sample = np.concatenate([data[:i], data[i + 1:]])
                      jackknife_stats[i] = statistic(jack_sample)
                  jack_mean = np.mean(jackknife_stats)
        num = np.sum((jack_mean - jackknife_stats) ** 3)
        den = np.sum((jack_mean - jackknife_stats) ** 2)
        a_hat = num / (6.0 * max(den ** 1.5, 1e-20))
        alpha = 1 - self.confidence
        z_lo = stats.norm.ppf(alpha / 2)
        z_hi = stats.norm.ppf(1 - alpha / 2)
        adj_lo = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a_hat * (z0 + z_lo)))
        adj_hi = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a_hat * (z0 + z_hi)))
        ci_lower = float(np.percentile(boot_stats, 100 * adj_lo))
        ci_upper = float(np.percentile(boot_stats, 100 * adj_hi))
        return {
                      "estimate": observed, "ci_lower": ci_lower, "ci_upper": ci_upper,
                      "std_error": float(np.std(boot_stats, ddof=1)), "n_samples": n,
                      "boot_mean": float(np.mean(boot_stats)),
                      "bias": float(np.mean(boot_stats) - observed),
        }

    def compute_effect_size(self, group_a, group_b) -> Dict:
              """Compute Cohen's d effect size between two groups."""
        a, b = np.asarray(group_a, dtype=float), np.asarray(group_b, dtype=float)
        pooled_std = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) +
                                                            (len(b) - 1) * np.var(b, ddof=1)) /
                                                          max(len(a) + len(b) - 2, 1))
        cohens_d = (np.mean(a) - np.mean(b)) / max(pooled_std, 1e-10)
        u_stat = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
        n1, n2 = len(a), len(b)
        rbc = 1 - (2 * u_stat) / (n1 * n2) if n1 * n2 > 0 else 0.0
        return {"cohens_d": float(cohens_d), "rank_biserial": float(rbc), "pooled_std": float(pooled_std)}
