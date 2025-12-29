"""Tests for confidence module."""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.confidence import ConfidenceEstimator, ObservabilityHandler, entropy, softmax_margin


class TestHelpers:
    def test_entropy_uniform(self):
        import math
        result = entropy([0.2]*5)
        assert abs(result - math.log(5)) < 0.0001
    
    def test_margin_tied(self):
        assert abs(softmax_margin([0.2]*5)) < 0.0001
    
    def test_margin_high(self):
        assert abs(softmax_margin([0.9, 0.05, 0.025, 0.015, 0.01]) - 0.85) < 0.001


class TestConfidenceEstimator:
    @pytest.fixture
    def est(self):
        return ConfidenceEstimator()
    
    def test_high_conf(self, est):
        result = est.estimate([0.9, 0.05, 0.02, 0.02, 0.01])
        assert result["confidence"] > 0.7
    
    def test_low_conf(self, est):
        result = est.estimate([0.2]*5)
        assert result["confidence"] < 0.5
    
    def test_levels(self, est):
        assert est.get_uncertainty_level(0.9) == "high_confidence"
        assert est.get_uncertainty_level(0.3) == "very_low_confidence"


class TestObservability:
    def test_not_observable(self):
        h = ObservabilityHandler()
        result = h.handle_not_observable("hormone_level")
        assert result["not_observable"] and result["confidence"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
