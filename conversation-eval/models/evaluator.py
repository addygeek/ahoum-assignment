"""
Main Evaluation Pipeline for ACEF.

Orchestrates the full pipeline:
1. Conversation preprocessing
2. Turn encoding
3. Facet routing
4. Scoring
5. Confidence estimation
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Import local modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessor import (
    Conversation, Turn, ConversationPreprocessor, build_turn_context
)
from data.facet_registry import (
    FacetRegistry, Facet, FacetCategory, Observability, create_registry_from_csv
)
from models.encoder import TurnEncoder, EncoderConfig, create_encoder
from models.scoring_heads import FacetScorer, FacetEmbeddings, ScoringConfig
from models.confidence import (
    ConfidenceEstimator, ObservabilityHandler, MCDropoutEstimator
)


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation pipeline."""
    # Encoder settings
    encoder_backend: str = "openrouter"
    encoder_model: str = "qwen/qwen-2-7b-instruct"
    
    # Scoring settings
    turn_embedding_dim: int = 4096
    facet_embedding_dim: int = 256
    
    # Confidence settings
    use_mc_dropout: bool = False
    mc_samples: int = 10
    
    # Processing settings
    context_window_size: int = 3
    batch_size: int = 16


@dataclass
class TurnScore:
    """Score result for a single turn-facet pair."""
    turn_id: int
    facet_id: int
    facet_name: str
    score: int
    label: str
    confidence: float
    not_observable: bool = False
    probabilities: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "facet_id": self.facet_id,
            "facet_name": self.facet_name,
            "score": self.score,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "not_observable": self.not_observable
        }


@dataclass
class ConversationEvaluation:
    """Complete evaluation result for a conversation."""
    conversation_id: str
    total_turns: int
    total_facets_evaluated: int
    scores: Dict[int, Dict[int, TurnScore]] = field(default_factory=dict)
    summary: Dict = field(default_factory=dict)
    
    def add_score(self, turn_id: int, facet_id: int, score: TurnScore):
        if turn_id not in self.scores:
            self.scores[turn_id] = {}
        self.scores[turn_id][facet_id] = score
    
    def get_turn_scores(self, turn_id: int) -> Dict[int, TurnScore]:
        return self.scores.get(turn_id, {})
    
    def get_facet_scores_across_turns(self, facet_id: int) -> List[TurnScore]:
        return [
            scores[facet_id] 
            for turn_id, scores in self.scores.items() 
            if facet_id in scores
        ]
    
    def compute_summary(self):
        """Compute summary statistics."""
        all_scores = []
        all_confidences = []
        not_observable_count = 0
        
        for turn_scores in self.scores.values():
            for score in turn_scores.values():
                if not score.not_observable:
                    all_scores.append(score.score)
                    all_confidences.append(score.confidence)
                else:
                    not_observable_count += 1
        
        self.summary = {
            "total_scores": len(all_scores),
            "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            "avg_confidence": sum(all_confidences) / len(all_confidences) if all_confidences else 0,
            "not_observable_count": not_observable_count
        }
    
    def to_dict(self) -> dict:
        scores_dict = {}
        for turn_id, facet_scores in self.scores.items():
            scores_dict[str(turn_id)] = {
                str(fid): score.to_dict() 
                for fid, score in facet_scores.items()
            }
        
        return {
            "conversation_id": self.conversation_id,
            "total_turns": self.total_turns,
            "total_facets_evaluated": self.total_facets_evaluated,
            "summary": self.summary,
            "scores": scores_dict
        }


class FacetRouter:
    """
    Routes facets to appropriate scoring clusters.
    
    Maps facet categories to scoring head clusters for efficient
    batch processing.
    """
    
    CATEGORY_TO_CLUSTER = {
        FacetCategory.BEHAVIORAL: "behavioral",
        FacetCategory.COGNITIVE: "cognitive",
        FacetCategory.EMOTIONAL: "emotional",
        FacetCategory.SAFETY: "safety",
        FacetCategory.SPIRITUAL: "spiritual",
        FacetCategory.PHYSIOLOGICAL: "physiological",
        FacetCategory.SOCIAL: "social",
        FacetCategory.PERSONALITY: "personality",
        FacetCategory.LINGUISTIC: "linguistic",
        FacetCategory.OTHER: "other"
    }
    
    def __init__(self, registry: FacetRegistry):
        self.registry = registry
        self._cluster_cache: Dict[str, List[Facet]] = {}
        self._build_cluster_cache()
    
    def _build_cluster_cache(self):
        """Pre-compute facets per cluster."""
        for facet in self.registry.facets.values():
            cluster = self.get_cluster(facet)
            if cluster not in self._cluster_cache:
                self._cluster_cache[cluster] = []
            self._cluster_cache[cluster].append(facet)
    
    def get_cluster(self, facet: Facet) -> str:
        """Get cluster name for a facet."""
        return self.CATEGORY_TO_CLUSTER.get(facet.category, "other")
    
    def get_facets_by_cluster(self, cluster: str) -> List[Facet]:
        """Get all facets in a cluster."""
        return self._cluster_cache.get(cluster, [])
    
    def route_facets(self, facet_ids: List[int]) -> Dict[str, List[Facet]]:
        """Route facets to their clusters."""
        routed = {}
        for fid in facet_ids:
            facet = self.registry.get_by_id(fid)
            if facet:
                cluster = self.get_cluster(facet)
                if cluster not in routed:
                    routed[cluster] = []
                routed[cluster].append(facet)
        return routed


class ConversationEvaluator:
    """
    Main evaluation pipeline for conversations.
    
    This is the core system that:
    1. Preprocesses conversations
    2. Encodes turns
    3. Routes facets to scoring clusters
    4. Computes scores with confidence
    """
    
    def __init__(
        self,
        registry: FacetRegistry,
        config: EvaluationConfig = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            registry: Populated facet registry
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.registry = registry
        
        # Initialize components
        self.preprocessor = ConversationPreprocessor(
            context_window_size=self.config.context_window_size
        )
        
        # Initialize encoder
        try:
            self.encoder = create_encoder(
                backend=self.config.encoder_backend,
                model=self.config.encoder_model
            )
        except ValueError:
            # Fallback if no API key
            print("Warning: Using pseudo-embeddings (no API key found)")
            self.encoder = create_encoder(backend="openrouter")
        
        # Initialize scorer
        self.scorer = FacetScorer(
            num_facets=len(registry.facets),
            turn_embedding_dim=self.config.turn_embedding_dim,
            facet_embedding_dim=self.config.facet_embedding_dim
        )
        
        # Initialize confidence estimation
        self.confidence_estimator = ConfidenceEstimator()
        self.observability_handler = ObservabilityHandler(self.confidence_estimator)
        
        # Initialize router
        self.router = FacetRouter(registry)
    
    def evaluate_turn(
        self,
        turn: Turn,
        turn_embedding: List[float],
        facets: List[Facet]
    ) -> List[TurnScore]:
        """
        Evaluate a single turn across multiple facets.
        
        Args:
            turn: The turn to evaluate
            turn_embedding: Encoded representation
            facets: Facets to evaluate
            
        Returns:
            List of TurnScore results
        """
        results = []
        
        for facet in facets:
            # Handle non-observable facets
            if facet.observability == Observability.NOT_OBSERVABLE:
                not_obs = self.observability_handler.handle_not_observable(facet.name)
                score = TurnScore(
                    turn_id=turn.turn_id,
                    facet_id=facet.facet_id,
                    facet_name=facet.name,
                    score=not_obs["score"],
                    label=not_obs["label"],
                    confidence=not_obs["confidence"],
                    not_observable=True,
                    probabilities=not_obs["probabilities"]
                )
                results.append(score)
                continue
            
            # Score the facet
            cluster = self.router.get_cluster(facet)
            score_result = self.scorer.score_facet(
                turn_embedding,
                facet.facet_id,
                cluster
            )
            
            # Estimate confidence
            conf_result = self.confidence_estimator.estimate(score_result["probabilities"])
            
            # Handle explicit-only observability
            if facet.observability == Observability.EXPLICIT_ONLY:
                # For explicit-only, we'd need keywords - for now just slightly lower confidence
                conf_result["confidence"] *= 0.8
            
            score = TurnScore(
                turn_id=turn.turn_id,
                facet_id=facet.facet_id,
                facet_name=facet.name,
                score=score_result["score"],
                label=score_result["label"],
                confidence=conf_result["confidence"],
                not_observable=False,
                probabilities=score_result["probabilities"]
            )
            results.append(score)
        
        return results
    
    def evaluate_conversation(
        self,
        conversation: Conversation,
        facet_ids: List[int] = None
    ) -> ConversationEvaluation:
        """
        Evaluate a complete conversation.
        
        Args:
            conversation: Conversation to evaluate
            facet_ids: Optional subset of facets to evaluate (default: all observable)
            
        Returns:
            ConversationEvaluation with all scores
        """
        # Preprocess
        processed = self.preprocessor.preprocess(conversation)
        
        # Determine facets to evaluate
        if facet_ids:
            facets = [self.registry.get_by_id(fid) for fid in facet_ids]
            facets = [f for f in facets if f is not None]
        else:
            facets = self.registry.get_observable_facets()
        
        # Initialize result
        evaluation = ConversationEvaluation(
            conversation_id=processed.conversation_id,
            total_turns=len(processed.turns),
            total_facets_evaluated=len(facets)
        )
        
        # Encode all turns
        turn_contexts = [
            build_turn_context(turn, processed) 
            for turn in processed.turns
        ]
        turn_embeddings = [self.encoder.encode_turn(ctx) for ctx in turn_contexts]
        
        # Evaluate each turn
        for turn, embedding in zip(processed.turns, turn_embeddings):
            turn_scores = self.evaluate_turn(turn, embedding, facets)
            
            for score in turn_scores:
                evaluation.add_score(turn.turn_id, score.facet_id, score)
        
        # Compute summary
        evaluation.compute_summary()
        
        return evaluation
    
    def evaluate_batch(
        self,
        conversations: List[Conversation],
        facet_ids: List[int] = None
    ) -> List[ConversationEvaluation]:
        """Evaluate multiple conversations."""
        return [self.evaluate_conversation(conv, facet_ids) for conv in conversations]


def create_evaluator(
    csv_path: str,
    config: EvaluationConfig = None
) -> ConversationEvaluator:
    """
    Factory function to create an evaluator from CSV.
    
    Args:
        csv_path: Path to Facets Assignment.csv
        config: Optional configuration
        
    Returns:
        Configured ConversationEvaluator
    """
    registry = create_registry_from_csv(csv_path)
    return ConversationEvaluator(registry, config)


if __name__ == "__main__":
    # Example usage
    print("Testing Conversation Evaluator...")
    
    # Create a sample conversation
    sample_conv = Conversation(
        conversation_id="test_001",
        turns=[
            Turn(turn_id=1, speaker="user", 
                 text="I'm feeling really stressed about my exam tomorrow."),
            Turn(turn_id=2, speaker="assistant", 
                 text="I understand exam stress can be overwhelming. What subject is it?"),
            Turn(turn_id=3, speaker="user", 
                 text="It's calculus. I've been studying for weeks but still feel unprepared."),
            Turn(turn_id=4, speaker="assistant", 
                 text="It sounds like you've put in significant effort. Let's focus on the key concepts.")
        ]
    )
    
    # Create a minimal registry for testing
    registry = FacetRegistry()
    
    # Add some test facets manually
    from data.facet_registry import Facet, SignalType, Scope
    
    test_facets = [
        Facet(1, "stress_level", "Stress Level", FacetCategory.EMOTIONAL,
              SignalType.STATE, Scope.SINGLE_TURN, Observability.IMPLICIT_ALLOWED),
        Facet(2, "helpfulness", "Helpfulness", FacetCategory.BEHAVIORAL,
              SignalType.BEHAVIOR, Scope.SINGLE_TURN, Observability.IMPLICIT_ALLOWED),
        Facet(3, "empathy", "Empathy", FacetCategory.EMOTIONAL,
              SignalType.LATENT_TRAIT, Scope.MULTI_TURN, Observability.IMPLICIT_ALLOWED),
    ]
    
    for facet in test_facets:
        registry.facets[facet.facet_id] = facet
        registry.name_to_id[facet.name] = facet.facet_id
    
    # Create evaluator
    config = EvaluationConfig()
    evaluator = ConversationEvaluator(registry, config)
    
    # Evaluate
    print("\nEvaluating conversation...")
    result = evaluator.evaluate_conversation(sample_conv, [1, 2, 3])
    
    print(f"\nConversation: {result.conversation_id}")
    print(f"Total turns: {result.total_turns}")
    print(f"Facets evaluated: {result.total_facets_evaluated}")
    print(f"Summary: {result.summary}")
    
    print("\nSample scores (Turn 1):")
    for fid, score in result.get_turn_scores(1).items():
        print(f"  {score.facet_name}: {score.label} (conf={score.confidence:.2f})")
    
    print("\n✓ Conversation evaluator working correctly!")
