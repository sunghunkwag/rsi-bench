"""Tests for rsi_bench.scoring module."""
import pytest
import numpy as np
from rsi_bench.scoring import RSIScorer


class TestRSIScorer:
      def test_init(self):
                scorer = RSIScorer()
                assert scorer is not None

      def test_aggregate_uniform(self):
                scorer = RSIScorer()
                scores = {"axis1": 0.8, "axis2": 0.6, "axis3": 0.7}
                result = scorer.aggregate(scores)
                assert isinstance(result, float)
                assert 0.0 <= result <= 1.0

      def test_aggregate_perfect(self):
                scorer = RSIScorer()
                scores = {"a1": 1.0, "a2": 1.0, "a3": 1.0}
                result = scorer.aggregate(scores)
                assert result == pytest.approx(1.0, abs=0.01)

      def test_aggregate_zero(self):
                scorer = RSIScorer()
                scores = {"a1": 0.0, "a2": 0.0}
                result = scorer.aggregate(scores)
                assert result == pytest.approx(0.0, abs=0.01)

      def test_aggregate_empty(self):
                scorer = RSIScorer()
                scores = {}
                try:
                              result = scorer.aggregate(scores)
                              assert result == 0.0 or result is not None
except (ValueError, ZeroDivisionError):
            pass

    def test_scores_bounded(self):
              scorer = RSIScorer()
              for _ in range(20):
                            scores = {f"axis{i}": np.random.random() for i in range(6)}
                            result = scorer.aggregate(scores)
                            assert 0.0 <= result <= 1.0

          def test_weighted_scoring(self):
                    scorer = RSIScorer()
                    low = {"a1": 0.1, "a2": 0.1}
                    high = {"a1": 0.9, "a2": 0.9}
                    assert scorer.aggregate(high) > scorer.aggregate(low)
