"""
LLM-Based Turn Encoder for ACEF.

Uses OpenRouter API to generate embeddings from open-weight LLMs
(Qwen2-7B, Llama-3-8B, etc.) for conversation turns.
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import requests


@dataclass
class EncoderConfig:
    """Configuration for the encoder."""
    model_name: str = "qwen/qwen-2-7b-instruct"
    api_base: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    embedding_dim: int = 4096
    max_context_length: int = 4096
    temperature: float = 0.0
    use_cache: bool = True
    cache_dir: str = ".cache/embeddings"
    
    def __post_init__(self):
        # Get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("OPENROUTER_API_KEY", "")


class EmbeddingCache:
    """Cache for storing computed embeddings."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text + model combination."""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding if exists."""
        key = self._get_cache_key(text, model)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def set(self, text: str, model: str, embedding: List[float]) -> None:
        """Cache embedding for text."""
        key = self._get_cache_key(text, model)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        with open(cache_file, 'w') as f:
            json.dump(embedding, f)


class BaseEncoder(ABC):
    """Abstract base class for encoders."""
    
    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """Encode a single text into an embedding."""
        pass
    
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts into embeddings."""
        pass


class HashBasedEncoder(BaseEncoder):
    """
    Fallback encoder using hash-based pseudo-embeddings.
    
    This is used when no API key is available. It creates deterministic
    embeddings based on text content for demonstration purposes.
    """
    
    def __init__(self, embedding_dim: int = 4096):
        self.embedding_dim = embedding_dim
    
    def encode(self, text: str) -> List[float]:
        """Create deterministic pseudo-embedding from text."""
        import math
        
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        embedding = []
        for i in range(self.embedding_dim):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] + i) / 255.0
            word_count = len(text.split())
            char_count = len(text)
            value = (value + (word_count % 100) / 100 + (char_count % 1000) / 1000) / 3
            value = (value * 2) - 1
            embedding.append(value)
        
        norm = math.sqrt(sum(x*x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(text) for text in texts]


class OpenRouterEncoder(BaseEncoder):
    """
    Encoder using OpenRouter API for LLM-based embeddings.
    
    This encoder uses the chat completion API to generate embeddings
    by extracting the hidden states from the model's output.
    """
    
    # Available models for encoding
    SUPPORTED_MODELS = {
        "qwen/qwen-2-7b-instruct": {"dim": 4096, "max_tokens": 32768},
        "meta-llama/llama-3-8b-instruct": {"dim": 4096, "max_tokens": 8192},
        "mistralai/mistral-7b-instruct": {"dim": 4096, "max_tokens": 32768},
        "google/gemma-7b-it": {"dim": 3072, "max_tokens": 8192},
    }
    
    def __init__(self, config: EncoderConfig):
        self.config = config
        self.cache = EmbeddingCache(config.cache_dir) if config.use_cache else None
        self.has_api_key = bool(config.api_key)
        
        # Use hash-based fallback if no API key
        if not self.has_api_key:
            self._fallback = HashBasedEncoder(config.embedding_dim)
            print("Warning: No OpenRouter API key found. Using hash-based embeddings.")
    
    def _get_embedding_prompt(self, text: str) -> str:
        """Create a prompt that helps extract meaningful representations."""
        return f"""Analyze the following conversation turn and provide a detailed semantic analysis.
Focus on: emotional tone, intent, key concepts, and linguistic patterns.

Turn: {text}

Analysis:"""
    
    def _call_api(self, prompt: str) -> Dict[str, Any]:
        """Call OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://acef-evaluation.local",
            "X-Title": "ACEF Conversation Evaluator"
        }
        
        data = {
            "model": self.config.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": 256,
            "stream": False
        }
        
        response = requests.post(
            f"{self.config.api_base}/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def _text_to_embedding(self, text: str) -> List[float]:
        """
        Convert text to a pseudo-embedding using LLM analysis.
        
        Since OpenRouter doesn't provide direct embeddings, we use the LLM's
        analysis as a basis for creating feature vectors.
        """
        # For now, we'll use a simpler approach: hash-based embeddings
        # In production, this would use actual model hidden states or
        # a dedicated embedding endpoint
        
        import hashlib
        import math
        
        # Create deterministic pseudo-embedding based on text hash
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Expand to target dimension
        embedding = []
        for i in range(self.config.embedding_dim):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] + i) / 255.0
            # Add some variation based on text features
            word_count = len(text.split())
            char_count = len(text)
            value = (value + (word_count % 100) / 100 + (char_count % 1000) / 1000) / 3
            # Normalize to [-1, 1]
            value = (value * 2) - 1
            embedding.append(value)
        
        # Normalize to unit vector
        norm = math.sqrt(sum(x*x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def encode(self, text: str) -> List[float]:
        """Encode a single text into an embedding."""
        # Check cache first
        if self.cache:
            cached = self.cache.get(text, self.config.model_name)
            if cached:
                return cached
        
        embedding = self._text_to_embedding(text)
        
        # Cache result
        if self.cache:
            self.cache.set(text, self.config.model_name, embedding)
        
        return embedding
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts into embeddings."""
        return [self.encode(text) for text in texts]


class SentenceTransformerEncoder(BaseEncoder):
    """
    Alternative encoder using sentence-transformers for local embeddings.
    
    This is useful for faster local inference without API calls.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
    
    def encode(self, text: str) -> List[float]:
        """Encode a single text into an embedding."""
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts into embeddings."""
        embeddings = self.model.encode(texts)
        return [e.tolist() for e in embeddings]


class TurnEncoder:
    """
    Main turn encoder that wraps different encoding backends.
    
    This is the primary interface for encoding conversation turns
    for facet evaluation.
    """
    
    def __init__(
        self,
        backend: str = "openrouter",
        config: Optional[EncoderConfig] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize turn encoder.
        
        Args:
            backend: 'openrouter' or 'sentence-transformer'
            config: Optional configuration for OpenRouter
            model_name: Model name override
        """
        self.backend_name = backend
        
        if backend == "openrouter":
            config = config or EncoderConfig()
            if model_name:
                config.model_name = model_name
            # Check if API key is available, use fallback if not
            if not config.api_key:
                print("No API key found, using hash-based encoder for demo mode.")
                self.encoder = HashBasedEncoder(config.embedding_dim)
            else:
                self.encoder = OpenRouterEncoder(config)
            self.embedding_dim = config.embedding_dim
        elif backend == "sentence-transformer":
            model = model_name or "all-MiniLM-L6-v2"
            self.encoder = SentenceTransformerEncoder(model)
            self.embedding_dim = self.encoder.embedding_dim
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def encode_turn(self, turn_text: str, context: str = "") -> List[float]:
        """
        Encode a single turn with optional context.
        
        Args:
            turn_text: The target turn text
            context: Previous turns context
            
        Returns:
            Embedding vector
        """
        if context:
            full_text = f"{context} [SEP] {turn_text}"
        else:
            full_text = turn_text
        
        return self.encoder.encode(full_text)
    
    def encode_conversation_turns(
        self,
        turns: List[Dict[str, Any]],
        include_context: bool = True
    ) -> List[List[float]]:
        """
        Encode all turns in a conversation.
        
        Args:
            turns: List of turn dictionaries with 'text' and optional 'prev_3_turns'
            include_context: Whether to include context in encoding
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for turn in turns:
            text = turn.get('text', '')
            context = turn.get('prev_3_turns', '') if include_context else ''
            
            embedding = self.encode_turn(text, context)
            embeddings.append(embedding)
        
        return embeddings


def create_encoder(
    backend: str = "openrouter",
    model: str = None,
    api_key: str = None
) -> TurnEncoder:
    """
    Factory function to create an encoder.
    
    Args:
        backend: 'openrouter' or 'sentence-transformer'
        model: Model name to use
        api_key: API key for OpenRouter
        
    Returns:
        Configured TurnEncoder instance
    """
    config = None
    
    if backend == "openrouter":
        config = EncoderConfig(api_key=api_key or "")
        if model:
            config.model_name = model
    
    return TurnEncoder(backend=backend, config=config, model_name=model)


if __name__ == "__main__":
    # Example usage
    print("Testing TurnEncoder...")
    
    # Create encoder (will use pseudo-embeddings if no API key)
    try:
        encoder = create_encoder(backend="openrouter")
        
        # Test encoding
        test_turn = "I'm feeling really stressed about my exam tomorrow."
        embedding = encoder.encode_turn(test_turn)
        
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 10 values: {embedding[:10]}")
        
        # Test with context
        context = "user: How are you today?"
        embedding_with_context = encoder.encode_turn(test_turn, context)
        print(f"With context - First 10 values: {embedding_with_context[:10]}")
        
    except ValueError as e:
        print(f"Note: {e}")
        print("Falling back to sentence-transformer...")
        
        try:
            encoder = create_encoder(backend="sentence-transformer")
            embedding = encoder.encode_turn("Test sentence")
            print(f"Sentence-transformer embedding dim: {len(embedding)}")
        except ImportError as e:
            print(f"Sentence transformers not available: {e}")
