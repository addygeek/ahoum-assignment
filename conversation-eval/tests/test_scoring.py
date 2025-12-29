"""Tests for scoring module."""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.scoring_heads import FacetScorer, FacetEmbeddings, softmax


class TestSoftmax:
    def test_sums_to_one(self):
        probs = softmax([1, 2, 3, 4, 5])
        assert abs(sum(probs) - 1.0) < 0.0001
    
    def test_positive(self):
        probs = softmax([-1, 0, 1])
        assert all(p > 0 for p in probs)


class TestFacetEmbeddings:
    def test_get(self):
        emb = FacetEmbeddings(10, 64)
        assert len(emb.get_embedding(1)) == 64
    
    def test_invalid_id(self):
        emb = FacetEmbeddings(10, 64)
        with pytest.raises(ValueError):
            emb.get_embedding(999)


class TestFacetScorer:
    @pytest.fixture
    def scorer(self):
        return FacetScorer(100, 256, 64)
    
    def test_score_single(self, scorer):
        result = scorer.score_facet([0.1]*256, 1, "emotional")
        assert 0 <= result["score"] <= 4
        assert len(result["probabilities"]) == 5
    
    def test_score_multiple(self, scorer):
        infos = [{"facet_id": i, "cluster": "emotional"} for i in [1,2,3]]
        results = scorer.score_multiple_facets([0.1]*256, infos)
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
