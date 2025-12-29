"""ACEF Custom Exceptions"""


class ACEFError(Exception):
    """Base exception for ACEF."""
    pass


class EncodingError(ACEFError):
    """Failed to encode text."""
    pass


class FacetNotFoundError(ACEFError):
    """Facet not found in registry."""
    pass


class ScoringError(ACEFError):
    """Scoring failed."""
    pass


class ConfigurationError(ACEFError):
    """Invalid configuration."""
    pass


class ValidationError(ACEFError):
    """Input validation failed."""
    pass


class InferenceTimeoutError(ACEFError):
    """Inference timeout."""
    pass


class ModelNotLoadedError(ACEFError):
    """Model not loaded."""
    pass
