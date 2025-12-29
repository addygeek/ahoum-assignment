"""Tests for encoder module."""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.encoder import HashBasedEncoder, TurnEncoder, create_encoder


class TestHashBasedEncoder:
    def test_encode_returns_list(self):
        encoder = HashBasedEncoder(256)
        result = encoder.encode("Hello world")
        assert isinstance(result, list) and len(result) == 256
    
    def test_deterministic(self):
        encoder = HashBasedEncoder(256)
        e1 = encoder.encode("same text")
        e2 = encoder.encode("same text")
        assert e1 == e2
    
    def test_different_texts(self):
        encoder = HashBasedEncoder(256)
        assert encoder.encode("A") != encoder.encode("B")
    
    def test_normalized(self):
        import math
        encoder = HashBasedEncoder(256)
        emb = encoder.encode("test")
        norm = math.sqrt(sum(x*x for x in emb))
        assert abs(norm - 1.0) < 0.001
    
    def test_batch(self):
        encoder = HashBasedEncoder(256)
        result = encoder.encode_batch(["a", "b", "c"])
        assert len(result) == 3


class TestTurnEncoder:
    def test_create_encoder(self):
        encoder = create_encoder(backend="openrouter")
        assert encoder is not None
    
    def test_encode_basic(self):
        encoder = create_encoder(backend="openrouter")
        emb = encoder.encode_turn("Hello")
        assert isinstance(emb, list)
    
    def test_with_context(self):
        encoder = create_encoder(backend="openrouter")
        e1 = encoder.encode_turn("Hi")
        e2 = encoder.encode_turn("Hi", "context")
        assert e1 != e2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
