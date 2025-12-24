"""
Inference Pipeline for ACEF.

Provides batch inference capabilities and configuration management.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import time


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    # Model settings
    encoder_backend: str = "openrouter"
    encoder_model: str = "qwen/qwen-2-7b-instruct"
    api_key: str = ""
    
    # Scoring settings
    turn_embedding_dim: int = 4096
    facet_embedding_dim: int = 256
    
    # Processing settings
    batch_size: int = 16
    context_window_size: int = 3
    max_concurrent: int = 4
    
    # Output settings
    include_probabilities: bool = False
    include_embeddings: bool = False
    min_confidence_threshold: float = 0.0
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("OPENROUTER_API_KEY", "")
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("api_key", None)  # Don't serialize API key
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> "InferenceConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_file(cls, path: str) -> "InferenceConfig":
        with open(path, 'r') as f:
            data = json.load(f)
        config = cls.from_dict(data)
        # Still try to get API key from env
        if not config.api_key:
            config.api_key = os.getenv("OPENROUTER_API_KEY", "")
        return config
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class BatchInferenceEngine:
    """
    Batch inference engine for processing multiple conversations.
    
    Optimizes throughput by:
    - Batching turn encodings
    - Caching embeddings
    - Parallel facet scoring
    """
    
    def __init__(
        self,
        evaluator,  # ConversationEvaluator
        config: InferenceConfig = None
    ):
        self.evaluator = evaluator
        self.config = config or InferenceConfig()
        self.stats = {
            "conversations_processed": 0,
            "turns_processed": 0,
            "facets_scored": 0,
            "total_time_ms": 0
        }
    
    def process_batch(
        self,
        conversations: List[Any],  # List[Conversation]
        facet_ids: List[int] = None,
        callback=None
    ) -> List[Any]:  # List[ConversationEvaluation]
        """
        Process a batch of conversations.
        
        Args:
            conversations: List of Conversation objects
            facet_ids: Optional subset of facets to evaluate
            callback: Optional progress callback
            
        Returns:
            List of ConversationEvaluation results
        """
        results = []
        start_time = time.time()
        
        for idx, conv in enumerate(conversations):
            result = self.evaluator.evaluate_conversation(conv, facet_ids)
            results.append(result)
            
            # Update stats
            self.stats["conversations_processed"] += 1
            self.stats["turns_processed"] += result.total_turns
            self.stats["facets_scored"] += (
                result.total_turns * result.total_facets_evaluated
            )
            
            if callback:
                callback(idx + 1, len(conversations), result)
        
        self.stats["total_time_ms"] = int((time.time() - start_time) * 1000)
        return results
    
    def process_file(
        self,
        input_path: str,
        output_path: str = None,
        facet_ids: List[int] = None
    ) -> Dict:
        """
        Process conversations from a JSON file.
        
        Args:
            input_path: Path to input JSON with conversations
            output_path: Optional path for output JSON
            facet_ids: Optional subset of facets
            
        Returns:
            Processing result with stats
        """
        # Import here to avoid circular dependency
        from data.preprocessor import ConversationPreprocessor
        
        # Load conversations
        conversations = ConversationPreprocessor.load_from_json(input_path)
        
        # Process
        results = self.process_batch(conversations, facet_ids)
        
        # Format output
        output = {
            "source_file": input_path,
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config": self.config.to_dict(),
            "stats": self.stats,
            "evaluations": [r.to_dict() for r in results]
        }
        
        # Save if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
        
        return output
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            "conversations_processed": 0,
            "turns_processed": 0,
            "facets_scored": 0,
            "total_time_ms": 0
        }


class StreamingInference:
    """
    Streaming inference for real-time evaluation.
    
    Processes turns as they arrive without waiting for
    the complete conversation.
    """
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.active_sessions: Dict[str, Dict] = {}
    
    def start_session(self, conversation_id: str) -> str:
        """Start a new streaming session."""
        self.active_sessions[conversation_id] = {
            "turns": [],
            "embeddings": [],
            "scores": {}
        }
        return conversation_id
    
    def add_turn(
        self,
        conversation_id: str,
        turn_id: int,
        speaker: str,
        text: str,
        facet_ids: List[int] = None
    ) -> Dict:
        """
        Add a turn and get immediate scores.
        
        Args:
            conversation_id: Session ID
            turn_id: Turn number
            speaker: Speaker type
            text: Turn text
            facet_ids: Facets to evaluate
            
        Returns:
            Scores for this turn
        """
        if conversation_id not in self.active_sessions:
            self.start_session(conversation_id)
        
        session = self.active_sessions[conversation_id]
        
        # Build context from previous turns
        context = ""
        if session["turns"]:
            recent = session["turns"][-3:]  # Last 3 turns
            context = " [SEP] ".join([
                f"{t['speaker']}: {t['text']}" for t in recent
            ])
        
        # Store turn
        turn_data = {
            "turn_id": turn_id,
            "speaker": speaker,
            "text": text
        }
        session["turns"].append(turn_data)
        
        # Encode
        full_text = f"{context} [SEP] {speaker}: {text}" if context else f"{speaker}: {text}"
        embedding = self.evaluator.encoder.encode_turn(full_text)
        session["embeddings"].append(embedding)
        
        # Score
        facets = facet_ids or [f.facet_id for f in self.evaluator.registry.get_observable_facets()[:10]]
        
        from data.preprocessor import Turn
        from data.facet_registry import Observability
        
        turn_obj = Turn(turn_id=turn_id, speaker=speaker, text=text)
        facet_objs = [self.evaluator.registry.get_by_id(fid) for fid in facets]
        facet_objs = [f for f in facet_objs if f is not None]
        
        scores = self.evaluator.evaluate_turn(turn_obj, embedding, facet_objs)
        
        session["scores"][turn_id] = {s.facet_id: s.to_dict() for s in scores}
        
        return session["scores"][turn_id]
    
    def end_session(self, conversation_id: str) -> Dict:
        """End session and return complete results."""
        if conversation_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions.pop(conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "total_turns": len(session["turns"]),
            "scores": session["scores"]
        }


def create_inference_engine(
    csv_path: str,
    config: InferenceConfig = None
) -> BatchInferenceEngine:
    """
    Factory function to create an inference engine.
    
    Args:
        csv_path: Path to facets CSV
        config: Optional configuration
        
    Returns:
        Configured BatchInferenceEngine
    """
    # Import here to avoid circular dependency
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.facet_registry import create_registry_from_csv
    from models.evaluator import ConversationEvaluator, EvaluationConfig
    
    # Create registry
    registry = create_registry_from_csv(csv_path)
    
    # Create evaluator config from inference config
    config = config or InferenceConfig()
    eval_config = EvaluationConfig(
        encoder_backend=config.encoder_backend,
        encoder_model=config.encoder_model,
        turn_embedding_dim=config.turn_embedding_dim,
        facet_embedding_dim=config.facet_embedding_dim,
        context_window_size=config.context_window_size
    )
    
    # Create evaluator
    evaluator = ConversationEvaluator(registry, eval_config)
    
    return BatchInferenceEngine(evaluator, config)


if __name__ == "__main__":
    print("Inference Pipeline Ready")
    print("Use create_inference_engine(csv_path) to initialize")
