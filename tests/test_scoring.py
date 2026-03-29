"""Tests for rsi_bench.scoring module.

UnifiedScorer API:
  - compute(results: BenchmarkResults) -> float
  - compute_from_dict(scores: dict) -> float
  - compute_arithmetic(scores: dict) -> float
  - compute_geometric(scores: dict) -> float

Uses weighted harmonic mean by default, penalizing zero scores.
"""
import pytest
import numpy as np
from rsi_bench.scoring import UnifiedScorer


class TestUnifiedScorer:

    def test_init_default_weights(self):
        scorer = UnifiedScorer()
        assert scorer is not None
        assert "smd" in scorer.weights
        assert "itq" in scorer.weights
        assert len(scorer.weights) == 6

    def test_init_custom_weights(self):
        scorer = UnifiedScorer(weights={"smd": 2.0, "itq": 1.0, "ssm": 3.0})
        assert scorer.weights["smd"] == 2.0
        assert scorer.weights["ssm"] == 3.0

    def test_uniform_scores(self):
        """All equal scores should return that value."""
        scorer = UnifiedScorer()
        scores = {"smd": 0.8, "itq": 0.8, "odr": 0.8,
                  "mas": 0.8, "ssm": 0.8, "agg": 0.8}
        result = scorer.compute_from_dict(scores)
        assert result == pytest.approx(0.8, abs=1e-6)

    def test_perfect_scores(self):
        scorer = UnifiedScorer()
        scores = {"smd": 1.0, "itq": 1.0, "odr": 1.0,
                  "mas": 1.0, "ssm": 1.0, "agg": 1.0}
        result = scorer.compute_from_dict(scores)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_zero_score_penalized(self):
        """Harmonic mean should heavily penalize a zero axis."""
        scorer = UnifiedScorer()
        scores = {"smd": 1.0, "itq": 1.0, "odr": 0.0,
                  "mas": 1.0, "ssm": 1.0, "agg": 1.0}
        result = scorer.compute_from_dict(scores)
        assert result < 0.01  # near-zero due to harmonic mean

    def test_empty_scores(self):
        scorer = UnifiedScorer()
        result = scorer.compute_from_dict({})
        assert result == 0.0

    def test_partial_scores(self):
        """Should handle dict with only some axes."""
        scorer = UnifiedScorer()
        scores = {"smd": 0.7, "itq": 0.9}
        result = scorer.compute_from_dict(scores)
        assert 0.0 <= result <= 1.0

    def test_scores_always_bounded(self):
        scorer = UnifiedScorer()
        rng = np.random.default_rng(42)
        for _ in range(50):
            scores = {k: rng.random() for k in ["smd", "itq", "odr", "mas", "ssm", "agg"]}
            result = scorer.compute_from_dict(scores)
            assert 0.0 <= result <= 1.0

    def test_higher_scores_produce_higher_composite(self):
        scorer = UnifiedScorer()
        low = {"smd": 0.1, "itq": 0.1, "odr": 0.1,
               "mas": 0.1, "ssm": 0.1, "agg": 0.1}
        high = {"smd": 0.9, "itq": 0.9, "odr": 0.9,
                "mas": 0.9, "ssm": 0.9, "agg": 0.9}
        assert scorer.compute_from_dict(high) > scorer.compute_from_dict(low)

    def test_arithmetic_mean(self):
        scorer = UnifiedScorer()
        scores = {"smd": 0.4, "itq": 0.6, "odr": 0.8}
        result = scorer.compute_arithmetic(scores)
        assert result == pytest.approx(0.6, abs=1e-6)

    def test_geometric_mean(self):
        scorer = UnifiedScorer()
        scores = {"smd": 0.5, "itq": 0.5, "odr": 0.5,
                  "mas": 0.5, "ssm": 0.5, "agg": 0.5}
        result = scorer.compute_geometric(scores)
        assert result == pytest.approx(0.5, abs=1e-3)

    def test_harmonic_less_than_arithmetic(self):
        """Harmonic mean <= arithmetic mean for non-uniform scores."""
        scorer = UnifiedScorer()
        scores = {"smd": 0.2, "itq": 0.8, "odr": 0.5,
                  "mas": 0.3, "ssm": 0.9, "agg": 0.6}
        harmonic = scorer.compute_from_dict(scores)
        arithmetic = scorer.compute_arithmetic(scores)
        assert harmonic <= arithmetic + 1e-10
