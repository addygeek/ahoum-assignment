"""
Models module for ACEF.

Provides:
- TurnEncoder: LLM-based turn encoding
- FacetScorer: Scalable facet scoring with embeddings
- ConfidenceEstimator: Uncertainty quantification
- ConversationEvaluator: Main evaluation pipeline
"""

from models.encoder import TurnEncoder, EncoderConfig, create_encoder
from models.scoring_heads import FacetScorer, FacetEmbeddings, ScoringConfig
from models.confidence import ConfidenceEstimator, ObservabilityHandler
from models.evaluator import (
    ConversationEvaluator, EvaluationConfig, 
    TurnScore, ConversationEvaluation, create_evaluator
)

__all__ = [
    # Encoder
    "TurnEncoder",
    "EncoderConfig", 
    "create_encoder",
    # Scoring
    "FacetScorer",
    "FacetEmbeddings",
    "ScoringConfig",
    # Confidence
    "ConfidenceEstimator",
    "ObservabilityHandler",
    # Evaluator
    "ConversationEvaluator",
    "EvaluationConfig",
    "TurnScore",
    "ConversationEvaluation",
    "create_evaluator",
]
