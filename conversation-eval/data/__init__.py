"""
Data module for ACEF.

Provides:
- FacetRegistry: Scalable facet management
- ConversationPreprocessor: Turn preprocessing
- Conversation/Turn: Data models
"""

from data.preprocessor import (
    Conversation, Turn, Speaker,
    ConversationPreprocessor, build_turn_context
)
from data.facet_registry import (
    FacetRegistry, Facet, FacetCategory, 
    Observability, SignalType, Scope,
    create_registry_from_csv
)

__all__ = [
    # Preprocessor
    "Conversation",
    "Turn", 
    "Speaker",
    "ConversationPreprocessor",
    "build_turn_context",
    # Registry
    "FacetRegistry",
    "Facet",
    "FacetCategory",
    "Observability",
    "SignalType",
    "Scope",
    "create_registry_from_csv",
]
