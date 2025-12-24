"""
FastAPI Application for ACEF.

Provides REST API endpoints for:
- Conversation evaluation
- Batch processing
- Facet registry queries
- Streaming evaluation
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    ConversationInput, EvaluationRequest, EvaluationResponse,
    BatchEvaluationRequest, BatchEvaluationResponse,
    FacetInfo, FacetListResponse, RegistrySummary,
    HealthResponse, StreamTurnRequest, StreamTurnResponse,
    ErrorResponse, TurnEvaluation, FacetScore, EvaluationSummary
)
from data.preprocessor import Conversation, Turn
from data.facet_registry import FacetRegistry, create_registry_from_csv
from models.evaluator import ConversationEvaluator, EvaluationConfig
from inference.pipeline import StreamingInference

# Global state
_evaluator = None
_registry = None
_streaming = None

# Configuration
CSV_PATH = os.getenv(
    "FACETS_CSV_PATH",
    str(Path(__file__).parent.parent.parent / "Facets Assignment.csv")
)


def get_evaluator():
    """Get or create the global evaluator."""
    global _evaluator, _registry
    
    if _evaluator is None:
        _registry = create_registry_from_csv(CSV_PATH)
        config = EvaluationConfig()
        _evaluator = ConversationEvaluator(_registry, config)
    
    return _evaluator


def get_registry():
    """Get the facet registry."""
    global _registry
    if _registry is None:
        get_evaluator()  # This initializes the registry
    return _registry


def get_streaming():
    """Get streaming inference handler."""
    global _streaming
    if _streaming is None:
        _streaming = StreamingInference(get_evaluator())
    return _streaming


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("Loading ACEF evaluator...")
    get_evaluator()
    print(f"Loaded {len(get_registry().facets)} facets")
    yield
    # Shutdown
    print("Shutting down ACEF...")


# Create FastAPI app
app = FastAPI(
    title="ACEF - Conversation Evaluation API",
    description="Ahoum Conversation Evaluation Framework API for scoring conversations across multiple facets",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and status."""
    registry = get_registry()
    evaluator = get_evaluator()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        facets_loaded=len(registry.facets),
        encoder_backend=evaluator.config.encoder_backend
    )


@app.get("/facets", response_model=FacetListResponse, tags=["Facets"])
async def list_facets(
    category: Optional[str] = Query(None, description="Filter by category"),
    observability: Optional[str] = Query(None, description="Filter by observability"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """List all available facets with optional filtering."""
    registry = get_registry()
    
    facets = list(registry.facets.values())
    
    # Apply filters
    if category:
        facets = [f for f in facets if f.category.value == category]
    if observability:
        facets = [f for f in facets if f.observability.value == observability]
    
    # Paginate
    total = len(facets)
    facets = facets[offset:offset + limit]
    
    return FacetListResponse(
        total_facets=total,
        facets=[
            FacetInfo(
                facet_id=f.facet_id,
                name=f.name,
                original_name=f.original_name,
                category=f.category.value,
                signal_type=f.signal_type.value,
                scope=f.scope.value,
                observability=f.observability.value,
                description=f.description
            )
            for f in facets
        ]
    )


@app.get("/facets/summary", response_model=RegistrySummary, tags=["Facets"])
async def get_registry_summary():
    """Get summary statistics of the facet registry."""
    registry = get_registry()
    summary = registry.summary()
    
    return RegistrySummary(
        total_facets=summary["total_facets"],
        by_category=summary["by_category"],
        by_observability=summary["by_observability"]
    )


@app.get("/facets/{facet_id}", response_model=FacetInfo, tags=["Facets"])
async def get_facet(facet_id: int):
    """Get details for a specific facet."""
    registry = get_registry()
    facet = registry.get_by_id(facet_id)
    
    if not facet:
        raise HTTPException(status_code=404, detail=f"Facet {facet_id} not found")
    
    return FacetInfo(
        facet_id=facet.facet_id,
        name=facet.name,
        original_name=facet.original_name,
        category=facet.category.value,
        signal_type=facet.signal_type.value,
        scope=facet.scope.value,
        observability=facet.observability.value,
        description=facet.description
    )


def convert_input_to_conversation(input_data: ConversationInput) -> Conversation:
    """Convert API input to internal Conversation object."""
    turns = [
        Turn(
            turn_id=t.turn_id,
            speaker=t.speaker.value,
            text=t.text
        )
        for t in input_data.turns
    ]
    
    return Conversation(
        conversation_id=input_data.conversation_id,
        turns=turns,
        domain_hint=input_data.domain_hint or ""
    )


def format_evaluation_response(
    result,  # ConversationEvaluation
    include_probs: bool = False
) -> EvaluationResponse:
    """Format evaluation result for API response."""
    turns = []
    
    for turn_id in sorted(result.scores.keys()):
        facet_scores = {}
        for facet_id, score in result.scores[turn_id].items():
            facet_scores[str(facet_id)] = FacetScore(
                facet_id=score.facet_id,
                facet_name=score.facet_name,
                score=score.score,
                label=score.label,
                confidence=score.confidence,
                not_observable=score.not_observable,
                probabilities=score.probabilities if include_probs else None
            )
        
        turns.append(TurnEvaluation(
            turn_id=turn_id,
            facet_scores=facet_scores
        ))
    
    return EvaluationResponse(
        conversation_id=result.conversation_id,
        total_turns=result.total_turns,
        total_facets_evaluated=result.total_facets_evaluated,
        summary=EvaluationSummary(
            total_scores=result.summary.get("total_scores", 0),
            avg_score=result.summary.get("avg_score", 0),
            avg_confidence=result.summary.get("avg_confidence", 0),
            not_observable_count=result.summary.get("not_observable_count", 0)
        ),
        turns=turns
    )


@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate_conversation(request: EvaluationRequest):
    """
    Evaluate a conversation across all or selected facets.
    
    Returns scores and confidence for each turn-facet pair.
    """
    try:
        evaluator = get_evaluator()
        
        # Convert input
        conversation = convert_input_to_conversation(request.conversation)
        
        # Evaluate
        result = evaluator.evaluate_conversation(
            conversation,
            request.facet_ids
        )
        
        return format_evaluation_response(result, request.include_probabilities)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/batch", response_model=BatchEvaluationResponse, tags=["Evaluation"])
async def evaluate_batch(request: BatchEvaluationRequest):
    """
    Evaluate multiple conversations in batch.
    
    More efficient for processing multiple conversations.
    """
    import time
    start = time.time()
    
    try:
        evaluator = get_evaluator()
        
        results = []
        for conv_input in request.conversations:
            conversation = convert_input_to_conversation(conv_input)
            result = evaluator.evaluate_conversation(
                conversation,
                request.facet_ids
            )
            results.append(format_evaluation_response(result))
        
        processing_time = int((time.time() - start) * 1000)
        
        return BatchEvaluationResponse(
            total_processed=len(results),
            processing_time_ms=processing_time,
            results=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream/start", tags=["Streaming"])
async def start_streaming_session(session_id: str):
    """Start a new streaming evaluation session."""
    streaming = get_streaming()
    streaming.start_session(session_id)
    return {"session_id": session_id, "status": "started"}


@app.post("/stream/turn", response_model=StreamTurnResponse, tags=["Streaming"])
async def stream_turn(request: StreamTurnRequest):
    """Add a turn to a streaming session and get immediate scores."""
    try:
        streaming = get_streaming()
        
        scores = streaming.add_turn(
            request.session_id,
            request.turn_id,
            request.speaker.value,
            request.text,
            request.facet_ids
        )
        
        # Convert to response format
        formatted_scores = {}
        for fid, score_data in scores.items():
            formatted_scores[str(fid)] = FacetScore(**score_data)
        
        return StreamTurnResponse(
            session_id=request.session_id,
            turn_id=request.turn_id,
            scores=formatted_scores
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream/end", tags=["Streaming"])
async def end_streaming_session(session_id: str):
    """End a streaming session and get complete results."""
    streaming = get_streaming()
    result = streaming.end_session(session_id)
    return result


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
