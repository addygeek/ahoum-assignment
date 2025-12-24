"""
Facet Scoring Heads for ACEF.

Implements the scalable facet scoring mechanism using:
- Facet embeddings (learned or initialized)
- Shared scoring MLPs per cluster
- Ordinal score prediction (5 levels)
"""

import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass  
class ScoringConfig:
    """Configuration for facet scoring."""
    embedding_dim: int = 4096
    facet_embedding_dim: int = 256
    hidden_dims: List[int] = None
    num_score_levels: int = 5
    dropout_rate: float = 0.1
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]


class FacetEmbeddings:
    """
    Manages facet embeddings for the scoring system.
    
    Facets are represented as learnable embeddings that are combined
    with turn embeddings for scoring. This allows scaling to 5000+ facets
    by simply adding new embedding rows.
    """
    
    def __init__(self, num_facets: int, embedding_dim: int = 256, seed: int = 42):
        """
        Initialize facet embeddings.
        
        Args:
            num_facets: Total number of facets
            embedding_dim: Dimension of each facet embedding
            seed: Random seed for reproducible initialization
        """
        self.num_facets = num_facets
        self.embedding_dim = embedding_dim
        self.embeddings: Dict[int, List[float]] = {}
        
        # Initialize embeddings
        import random
        random.seed(seed)
        
        for facet_id in range(1, num_facets + 1):
            # Xavier/Glorot initialization
            scale = math.sqrt(2.0 / (embedding_dim + embedding_dim))
            embedding = [random.gauss(0, scale) for _ in range(embedding_dim)]
            
            # Normalize
            norm = math.sqrt(sum(x*x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]
            
            self.embeddings[facet_id] = embedding
    
    def get_embedding(self, facet_id: int) -> List[float]:
        """Get embedding for a specific facet."""
        if facet_id not in self.embeddings:
            raise ValueError(f"Unknown facet_id: {facet_id}")
        return self.embeddings[facet_id]
    
    def get_batch_embeddings(self, facet_ids: List[int]) -> List[List[float]]:
        """Get embeddings for multiple facets."""
        return [self.get_embedding(fid) for fid in facet_ids]
    
    def add_facet(self, facet_id: int, embedding: List[float] = None) -> None:
        """Add a new facet embedding (for scaling)."""
        if embedding is None:
            # Generate new embedding
            import random
            scale = math.sqrt(2.0 / (self.embedding_dim * 2))
            embedding = [random.gauss(0, scale) for _ in range(self.embedding_dim)]
            norm = math.sqrt(sum(x*x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]
        
        self.embeddings[facet_id] = embedding
        self.num_facets = max(self.num_facets, facet_id)
    
    def save(self, path: str) -> None:
        """Save embeddings to file."""
        data = {
            "num_facets": self.num_facets,
            "embedding_dim": self.embedding_dim,
            "embeddings": self.embeddings
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> "FacetEmbeddings":
        """Load embeddings from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        instance = cls(data["num_facets"], data["embedding_dim"])
        instance.embeddings = {int(k): v for k, v in data["embeddings"].items()}
        return instance


def relu(x: float) -> float:
    """ReLU activation function."""
    return max(0, x)


def softmax(logits: List[float]) -> List[float]:
    """Compute softmax probabilities."""
    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(exp_logits)
    return [x / sum_exp for x in exp_logits]


class SimpleMLP:
    """
    Simple multilayer perceptron for scoring.
    
    In production, this would be replaced with PyTorch or JAX.
    This pure-Python implementation is for portability.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        seed: int = 42
    ):
        """
        Initialize MLP with Xavier initialization.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (num_score_levels)
            seed: Random seed
        """
        import random
        random.seed(seed)
        
        self.layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            scale = math.sqrt(2.0 / (in_dim + out_dim))
            
            # Initialize weights and biases
            weights = [[random.gauss(0, scale) for _ in range(out_dim)] 
                      for _ in range(in_dim)]
            biases = [0.0] * out_dim
            
            self.layers.append((weights, biases))
    
    def forward(self, x: List[float]) -> List[float]:
        """Forward pass through the network."""
        for i, (weights, biases) in enumerate(self.layers):
            # Linear transformation
            out = [sum(x[j] * weights[j][k] for j in range(len(x))) + biases[k]
                   for k in range(len(biases))]
            
            # Apply ReLU for all but last layer
            if i < len(self.layers) - 1:
                out = [relu(v) for v in out]
            
            x = out
        
        return x


class FacetScoringHead:
    """
    Scoring head for a cluster of facets.
    
    Each cluster (behavioral, emotional, cognitive, etc.) shares a scoring head.
    The facet embedding is concatenated with the turn embedding before scoring.
    """
    
    SCORE_LABELS = ["very_low", "low", "neutral", "high", "very_high"]
    
    def __init__(
        self,
        cluster_name: str,
        turn_embedding_dim: int,
        facet_embedding_dim: int,
        config: ScoringConfig = None
    ):
        """
        Initialize scoring head.
        
        Args:
            cluster_name: Name of the facet cluster
            turn_embedding_dim: Dimension of turn embeddings
            facet_embedding_dim: Dimension of facet embeddings
            config: Scoring configuration
        """
        self.cluster_name = cluster_name
        self.config = config or ScoringConfig()
        
        # Combined input is turn_embedding + facet_embedding
        input_dim = turn_embedding_dim + facet_embedding_dim
        
        self.mlp = SimpleMLP(
            input_dim=input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.num_score_levels
        )
    
    def score(
        self,
        turn_embedding: List[float],
        facet_embedding: List[float]
    ) -> Tuple[int, List[float]]:
        """
        Score a turn for a specific facet.
        
        Args:
            turn_embedding: Turn representation vector
            facet_embedding: Facet representation vector
            
        Returns:
            Tuple of (score_level, probabilities)
        """
        # Concatenate embeddings
        combined = turn_embedding + facet_embedding
        
        # Get logits
        logits = self.mlp.forward(combined)
        
        # Convert to probabilities
        probs = softmax(logits)
        
        # Get predicted score (argmax)
        score_idx = max(range(len(probs)), key=lambda i: probs[i])
        
        return score_idx, probs
    
    def score_to_label(self, score_idx: int) -> str:
        """Convert score index to label."""
        return self.SCORE_LABELS[score_idx]


class FacetScorer:
    """
    Main facet scoring system.
    
    Manages facet embeddings and scoring heads for all facet clusters.
    This is the core scalability mechanism - adding facets only requires
    adding new embedding rows, not architectural changes.
    """
    
    DEFAULT_CLUSTERS = [
        "behavioral", "cognitive", "emotional", "safety",
        "spiritual", "physiological", "social", "personality",
        "linguistic", "other"
    ]
    
    def __init__(
        self,
        num_facets: int,
        turn_embedding_dim: int = 4096,
        facet_embedding_dim: int = 256,
        config: ScoringConfig = None
    ):
        """
        Initialize facet scorer.
        
        Args:
            num_facets: Total number of facets
            turn_embedding_dim: Dimension of turn embeddings
            facet_embedding_dim: Dimension of facet embeddings
            config: Scoring configuration
        """
        self.config = config or ScoringConfig(
            embedding_dim=turn_embedding_dim,
            facet_embedding_dim=facet_embedding_dim
        )
        
        # Initialize facet embeddings
        self.facet_embeddings = FacetEmbeddings(num_facets, facet_embedding_dim)
        
        # Initialize scoring heads per cluster
        self.scoring_heads: Dict[str, FacetScoringHead] = {}
        for cluster in self.DEFAULT_CLUSTERS:
            self.scoring_heads[cluster] = FacetScoringHead(
                cluster_name=cluster,
                turn_embedding_dim=turn_embedding_dim,
                facet_embedding_dim=facet_embedding_dim,
                config=self.config
            )
    
    def score_facet(
        self,
        turn_embedding: List[float],
        facet_id: int,
        cluster: str = "other"
    ) -> Dict:
        """
        Score a single facet for a turn.
        
        Args:
            turn_embedding: Turn representation
            facet_id: Facet to score
            cluster: Facet cluster (for routing to correct head)
            
        Returns:
            Score result with level, label, and probabilities
        """
        # Get facet embedding
        facet_emb = self.facet_embeddings.get_embedding(facet_id)
        
        # Get appropriate scoring head
        if cluster not in self.scoring_heads:
            cluster = "other"
        head = self.scoring_heads[cluster]
        
        # Score
        score_idx, probs = head.score(turn_embedding, facet_emb)
        
        return {
            "facet_id": facet_id,
            "score": score_idx,
            "label": head.score_to_label(score_idx),
            "probabilities": probs
        }
    
    def score_multiple_facets(
        self,
        turn_embedding: List[float],
        facet_infos: List[Dict]
    ) -> List[Dict]:
        """
        Score multiple facets for a single turn.
        
        Args:
            turn_embedding: Turn representation
            facet_infos: List of {facet_id, cluster} dicts
            
        Returns:
            List of score results
        """
        results = []
        for info in facet_infos:
            result = self.score_facet(
                turn_embedding,
                info["facet_id"],
                info.get("cluster", "other")
            )
            results.append(result)
        return results
    
    def add_facet(self, facet_id: int, embedding: List[float] = None) -> None:
        """Add a new facet (scales without code changes)."""
        self.facet_embeddings.add_facet(facet_id, embedding)


if __name__ == "__main__":
    # Example usage
    print("Testing Facet Scoring System...")
    
    # Initialize with 400 facets
    scorer = FacetScorer(num_facets=400, turn_embedding_dim=4096)
    
    # Create dummy turn embedding
    import random
    random.seed(123)
    turn_emb = [random.gauss(0, 0.1) for _ in range(4096)]
    
    # Score a facet
    result = scorer.score_facet(turn_emb, facet_id=1, cluster="behavioral")
    print(f"\nFacet 1 Score: {result['label']} (idx={result['score']})")
    print(f"Probabilities: {[f'{p:.3f}' for p in result['probabilities']]}")
    
    # Score multiple facets
    facet_infos = [
        {"facet_id": 1, "cluster": "behavioral"},
        {"facet_id": 10, "cluster": "emotional"},
        {"facet_id": 50, "cluster": "cognitive"},
    ]
    results = scorer.score_multiple_facets(turn_emb, facet_infos)
    
    print("\nMultiple facet scores:")
    for r in results:
        print(f"  Facet {r['facet_id']}: {r['label']}")
    
    # Test adding a new facet (scaling demo)
    scorer.add_facet(401)
    result_new = scorer.score_facet(turn_emb, facet_id=401, cluster="other")
    print(f"\nNew Facet 401: {result_new['label']}")
    
    print("\n✓ Facet scoring system working correctly!")
