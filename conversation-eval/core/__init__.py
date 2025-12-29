"""
Custom exceptions for ACEF.
"""


class ACEFError(Exception):
    """Base exception for all ACEF errors."""
    pass


class EncodingError(ACEFError):
    """Failed to encode text into embeddings."""
    pass


class FacetNotFoundError(ACEFError):
    """Requested facet does not exist in registry."""
    pass


class ScoringError(ACEFError):
    """Failed to score a turn-facet pair."""
    pass


class ConfigurationError(ACEFError):
    """Invalid configuration."""
    pass


class ValidationError(ACEFError):
    """Input validation failed."""
    pass


class InferenceTimeoutError(ACEFError):
    """Inference exceeded time limit."""
    pass


class ModelNotLoadedError(ACEFError):
    """Model weights not loaded."""
    pass
