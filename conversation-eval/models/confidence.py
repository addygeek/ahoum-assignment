"""
Confidence Estimation for ACEF.

Implements confidence scoring using:
1. Softmax margin (difference between top-2 predictions)
2. Prediction entropy (uncertainty in distribution)
3. Optional MC Dropout variance (for neural networks)
"""

import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConfidenceConfig:
    """Configuration for confidence estimation."""
    margin_weight: float = 0.4
    entropy_weight: float = 0.4
    calibration_weight: float = 0.2
    min_confidence: float = 0.1
    max_confidence: float = 0.99
    num_classes: int = 5


def safe_log(x: float, eps: float = 1e-10) -> float:
    """Safe logarithm to avoid log(0)."""
    return math.log(max(x, eps))


def entropy(probs: List[float]) -> float:
    """
    Compute entropy of a probability distribution.
    
    Higher entropy = more uncertainty = lower confidence.
    
    Args:
        probs: Probability distribution (sums to 1)
        
    Returns:
        Entropy value (0 = certain, log(n) = maximum uncertainty)
    """
    return -sum(p * safe_log(p) for p in probs if p > 0)


def normalized_entropy(probs: List[float]) -> float:
    """
    Compute normalized entropy (0-1 scale).
    
    Args:
        probs: Probability distribution
        
    Returns:
        Normalized entropy (0 = certain, 1 = max uncertainty)
    """
    n = len(probs)
    if n <= 1:
        return 0.0
    
    max_entropy = math.log(n)
    if max_entropy == 0:
        return 0.0
    
    return entropy(probs) / max_entropy


def softmax_margin(probs: List[float]) -> float:
    """
    Compute margin between top-2 predictions.
    
    Larger margin = more confident.
    
    Args:
        probs: Probability distribution
        
    Returns:
        Margin value (0 = tied, 1 = max confidence)
    """
    if len(probs) < 2:
        return 1.0
    
    sorted_probs = sorted(probs, reverse=True)
    return sorted_probs[0] - sorted_probs[1]


def max_probability(probs: List[float]) -> float:
    """Get the maximum probability value."""
    return max(probs)


class ConfidenceEstimator:
    """
    Estimates confidence scores for facet predictions.
    
    Combines multiple signals:
    - Softmax margin: How much the model prefers one class over others
    - Entropy: Overall uncertainty in the distribution
    - Calibration: Optional learned calibration factor
    """
    
    def __init__(self, config: ConfidenceConfig = None):
        """
        Initialize confidence estimator.
        
        Args:
            config: Confidence configuration
        """
        self.config = config or ConfidenceConfig()
    
    def estimate(self, probs: List[float]) -> Dict:
        """
        Estimate confidence from prediction probabilities.
        
        Args:
            probs: Softmax probabilities from scoring head
            
        Returns:
            Dictionary with confidence score and components
        """
        # Compute component scores
        margin = softmax_margin(probs)
        norm_entropy = normalized_entropy(probs)
        max_prob = max_probability(probs)
        
        # Entropy-based confidence (inverted - higher entropy = lower confidence)
        entropy_confidence = 1.0 - norm_entropy
        
        # Combine signals
        confidence = (
            self.config.margin_weight * margin +
            self.config.entropy_weight * entropy_confidence +
            self.config.calibration_weight * max_prob
        )
        
        # Clamp to valid range
        confidence = max(self.config.min_confidence, 
                        min(self.config.max_confidence, confidence))
        
        return {
            "confidence": confidence,
            "margin": margin,
            "entropy": norm_entropy,
            "max_prob": max_prob,
            "components": {
                "margin_contribution": self.config.margin_weight * margin,
                "entropy_contribution": self.config.entropy_weight * entropy_confidence,
                "calibration_contribution": self.config.calibration_weight * max_prob
            }
        }
    
    def estimate_batch(self, probs_batch: List[List[float]]) -> List[Dict]:
        """Estimate confidence for multiple predictions."""
        return [self.estimate(probs) for probs in probs_batch]
    
    def get_uncertainty_level(self, confidence: float) -> str:
        """
        Categorize confidence level.
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Uncertainty level label
        """
        if confidence >= 0.85:
            return "high_confidence"
        elif confidence >= 0.65:
            return "moderate_confidence"
        elif confidence >= 0.4:
            return "low_confidence"
        else:
            return "very_low_confidence"


class ObservabilityHandler:
    """
    Handles observability-aware scoring.
    
    Some facets cannot be observed from conversation text:
    - Physiological facets (hormone levels, etc.)
    - Genetic facets
    
    These should return neutral scores with low confidence.
    """
    
    NEUTRAL_SCORE = 2  # Index for "neutral" in 5-level scale
    
    def __init__(self, confidence_estimator: ConfidenceEstimator = None):
        """
        Initialize observability handler.
        
        Args:
            confidence_estimator: Confidence estimator instance
        """
        self.confidence_estimator = confidence_estimator or ConfidenceEstimator()
    
    def handle_not_observable(self, facet_name: str) -> Dict:
        """
        Generate result for non-observable facet.
        
        Args:
            facet_name: Name of the facet
            
        Returns:
            Score result with neutral value and flag
        """
        # Uniform distribution for non-observable
        uniform_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        return {
            "score": self.NEUTRAL_SCORE,
            "label": "neutral",
            "probabilities": uniform_probs,
            "confidence": 0.0,
            "not_observable": True,
            "reason": f"Facet '{facet_name}' cannot be observed from text"
        }
    
    def handle_explicit_only(
        self,
        facet_name: str,
        text: str,
        keywords: List[str],
        score_result: Dict
    ) -> Dict:
        """
        Handle facets that require explicit mention.
        
        Args:
            facet_name: Name of the facet
            text: Turn text to check
            keywords: Keywords that indicate explicit mention
            score_result: Original scoring result
            
        Returns:
            Modified result if no explicit mention found
        """
        text_lower = text.lower()
        
        # Check if any keyword is mentioned
        mentioned = any(kw.lower() in text_lower for kw in keywords)
        
        if not mentioned:
            # Return neutral with explanation
            return {
                "score": self.NEUTRAL_SCORE,
                "label": "neutral",
                "probabilities": [0.1, 0.2, 0.4, 0.2, 0.1],  # Centered distribution
                "confidence": 0.3,
                "not_observable": False,
                "explicit_required": True,
                "reason": f"Facet '{facet_name}' requires explicit mention"
            }
        
        return score_result


class MCDropoutEstimator:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    In production with PyTorch/JAX, this would run multiple forward
    passes with dropout enabled to estimate prediction variance.
    
    This is a simulation for the pure-Python implementation.
    """
    
    def __init__(self, num_samples: int = 10, dropout_rate: float = 0.1):
        """
        Initialize MC Dropout estimator.
        
        Args:
            num_samples: Number of forward passes
            dropout_rate: Dropout probability
        """
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
    
    def estimate_with_dropout(
        self,
        base_probs: List[float],
        seed: int = None
    ) -> Dict:
        """
        Simulate MC Dropout variance estimation.
        
        Args:
            base_probs: Base prediction probabilities
            seed: Random seed
            
        Returns:
            Variance-based uncertainty estimate
        """
        import random
        if seed is not None:
            random.seed(seed)
        
        # Simulate multiple predictions with noise
        all_probs = [base_probs]
        for _ in range(self.num_samples - 1):
            # Add noise to simulate dropout effect
            noisy = []
            for p in base_probs:
                noise = random.gauss(0, 0.05)
                new_p = max(0.001, min(0.999, p + noise))
                noisy.append(new_p)
            
            # Renormalize
            total = sum(noisy)
            noisy = [p / total for p in noisy]
            all_probs.append(noisy)
        
        # Compute variance for each class
        num_classes = len(base_probs)
        variances = []
        for i in range(num_classes):
            class_probs = [probs[i] for probs in all_probs]
            mean = sum(class_probs) / len(class_probs)
            var = sum((p - mean) ** 2 for p in class_probs) / len(class_probs)
            variances.append(var)
        
        # Mean variance as uncertainty
        mean_variance = sum(variances) / len(variances)
        
        # Convert to confidence (lower variance = higher confidence)
        variance_confidence = 1.0 - min(1.0, mean_variance * 10)
        
        return {
            "mean_variance": mean_variance,
            "class_variances": variances,
            "variance_confidence": variance_confidence
        }


def combine_confidence_signals(
    softmax_confidence: Dict,
    mc_dropout_result: Optional[Dict] = None,
    weights: Dict = None
) -> float:
    """
    Combine multiple confidence signals into final score.
    
    Args:
        softmax_confidence: Result from ConfidenceEstimator
        mc_dropout_result: Optional MC Dropout result
        weights: Optional custom weights
        
    Returns:
        Final combined confidence score
    """
    if weights is None:
        weights = {
            "softmax": 0.7,
            "mc_dropout": 0.3
        }
    
    if mc_dropout_result is None:
        return softmax_confidence["confidence"]
    
    combined = (
        weights["softmax"] * softmax_confidence["confidence"] +
        weights["mc_dropout"] * mc_dropout_result["variance_confidence"]
    )
    
    return max(0.1, min(0.99, combined))


if __name__ == "__main__":
    # Example usage
    print("Testing Confidence Estimation...")
    
    # Initialize estimator
    estimator = ConfidenceEstimator()
    
    # Test with different probability distributions
    test_cases = [
        [0.8, 0.1, 0.05, 0.03, 0.02],  # High confidence
        [0.3, 0.25, 0.2, 0.15, 0.1],    # Medium confidence
        [0.2, 0.2, 0.2, 0.2, 0.2],      # Low confidence (uniform)
        [0.95, 0.02, 0.01, 0.01, 0.01], # Very high confidence
    ]
    
    print("\nConfidence Estimates:")
    for i, probs in enumerate(test_cases):
        result = estimator.estimate(probs)
        level = estimator.get_uncertainty_level(result["confidence"])
        print(f"  Case {i+1}: conf={result['confidence']:.3f}, "
              f"margin={result['margin']:.3f}, "
              f"entropy={result['entropy']:.3f} -> {level}")
    
    # Test observability handling
    obs_handler = ObservabilityHandler()
    
    # Non-observable facet
    not_obs = obs_handler.handle_not_observable("hormone_level")
    print(f"\nNon-observable facet: score={not_obs['label']}, "
          f"conf={not_obs['confidence']}")
    
    # Test MC Dropout
    mc_estimator = MCDropoutEstimator()
    mc_result = mc_estimator.estimate_with_dropout(test_cases[0], seed=42)
    print(f"\nMC Dropout: variance={mc_result['mean_variance']:.4f}, "
          f"conf={mc_result['variance_confidence']:.3f}")
    
    # Combined confidence
    combined = combine_confidence_signals(
        estimator.estimate(test_cases[0]),
        mc_result
    )
    print(f"Combined confidence: {combined:.3f}")
    
    print("\nConfidence estimation working correctly.")
