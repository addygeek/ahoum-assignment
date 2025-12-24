"""
Pydantic schemas for ACEF API request/response models.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class Speaker(str, Enum):
    """Speaker types."""
    user = "user"
    assistant = "assistant"
    system = "system"


class TurnInput(BaseModel):
    """Input for a single conversation turn."""
    turn_id: int = Field(..., description="Sequential turn ID")
    speaker: Speaker = Field(..., description="Speaker type")
    text: str = Field(..., description="Turn text content")


class ConversationInput(BaseModel):
    """Input for a complete conversation."""
    conversation_id: str = Field(..., description="Unique conversation identifier")
    turns: List[TurnInput] = Field(..., description="List of conversation turns")
    domain_hint: Optional[str] = Field("", description="Optional domain hint")
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_001",
                "turns": [
                    {"turn_id": 1, "speaker": "user", "text": "I need help with my code"},
                    {"turn_id": 2, "speaker": "assistant", "text": "I'd be happy to help. What language?"}
                ]
            }
        }


class EvaluationRequest(BaseModel):
    """Request for conversation evaluation."""
    conversation: ConversationInput = Field(..., description="Conversation to evaluate")
    facet_ids: Optional[List[int]] = Field(None, description="Specific facets to evaluate")
    include_probabilities: bool = Field(False, description="Include probability distributions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation": {
                    "conversation_id": "conv_001",
                    "turns": [
                        {"turn_id": 1, "speaker": "user", "text": "Hello!"}
                    ]
                },
                "facet_ids": [1, 2, 3]
            }
        }


class FacetScore(BaseModel):
    """Score for a single facet."""
    facet_id: int
    facet_name: str
    score: int = Field(..., ge=0, le=4, description="Score level 0-4")
    label: str = Field(..., description="Score label")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    not_observable: bool = Field(False, description="Facet not observable from text")
    probabilities: Optional[List[float]] = None


class TurnEvaluation(BaseModel):
    """Evaluation results for a single turn."""
    turn_id: int
    facet_scores: Dict[str, FacetScore]


class EvaluationSummary(BaseModel):
    """Summary statistics for evaluation."""
    total_scores: int
    avg_score: float
    avg_confidence: float
    not_observable_count: int


class EvaluationResponse(BaseModel):
    """Response for conversation evaluation."""
    conversation_id: str
    total_turns: int
    total_facets_evaluated: int
    summary: EvaluationSummary
    turns: List[TurnEvaluation]
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_001",
                "total_turns": 2,
                "total_facets_evaluated": 10,
                "summary": {
                    "total_scores": 20,
                    "avg_score": 2.5,
                    "avg_confidence": 0.75,
                    "not_observable_count": 0
                },
                "turns": []
            }
        }


class BatchEvaluationRequest(BaseModel):
    """Request for batch evaluation."""
    conversations: List[ConversationInput]
    facet_ids: Optional[List[int]] = None


class BatchEvaluationResponse(BaseModel):
    """Response for batch evaluation."""
    total_processed: int
    processing_time_ms: int
    results: List[EvaluationResponse]


class FacetInfo(BaseModel):
    """Information about a facet."""
    facet_id: int
    name: str
    original_name: str
    category: str
    signal_type: str
    scope: str
    observability: str
    description: str


class FacetListResponse(BaseModel):
    """Response with list of facets."""
    total_facets: int
    facets: List[FacetInfo]


class RegistrySummary(BaseModel):
    """Summary of facet registry."""
    total_facets: int
    by_category: Dict[str, int]
    by_observability: Dict[str, int]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    facets_loaded: int
    encoder_backend: str


class StreamTurnRequest(BaseModel):
    """Request for streaming turn evaluation."""
    session_id: str
    turn_id: int
    speaker: Speaker
    text: str
    facet_ids: Optional[List[int]] = None


class StreamTurnResponse(BaseModel):
    """Response for streaming turn evaluation."""
    session_id: str
    turn_id: int
    scores: Dict[str, FacetScore]


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
