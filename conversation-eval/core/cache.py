"""Caching layer for embeddings."""

import json
import hashlib
import os
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod


class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        pass
    
    @abstractmethod
    def set(self, key: str, value: str, ttl: int = 3600):
        pass
    
    @abstractmethod
    def delete(self, key: str):
        pass


class MemoryCache(CacheBackend):
    """Simple in-memory cache."""
    
    def __init__(self, max_size: int = 10000):
        self._cache = {}
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[str]:
        return self._cache.get(key)
    
    def set(self, key: str, value: str, ttl: int = 3600):
        if len(self._cache) >= self._max_size:
            # Evict oldest 10%
            for k in list(self._cache.keys())[:self._max_size // 10]:
                del self._cache[k]
        self._cache[key] = value
    
    def delete(self, key: str):
        self._cache.pop(key, None)
    
    def clear(self):
        self._cache.clear()


class FileCache(CacheBackend):
    """File-based persistent cache."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _path(self, key: str) -> str:
        h = hashlib.sha256(key.encode()).hexdigest()[:32]
        return os.path.join(self.cache_dir, f"{h}.json")
    
    def get(self, key: str) -> Optional[str]:
        path = self._path(key)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()
        return None
    
    def set(self, key: str, value: str, ttl: int = 3600):
        with open(self._path(key), 'w') as f:
            f.write(value)
    
    def delete(self, key: str):
        path = self._path(key)
        if os.path.exists(path):
            os.remove(path)


class EmbeddingCache:
    """Cache wrapper for text embeddings."""
    
    def __init__(self, backend: CacheBackend = None, prefix: str = "emb"):
        self.backend = backend or MemoryCache()
        self.prefix = prefix
        self._hits = 0
        self._misses = 0
    
    def _key(self, text: str, model: str) -> str:
        h = hashlib.sha256(f"{model}:{text}".encode()).hexdigest()[:16]
        return f"{self.prefix}:{h}"
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        cached = self.backend.get(self._key(text, model))
        if cached:
            self._hits += 1
            return json.loads(cached)
        self._misses += 1
        return None
    
    def set(self, text: str, model: str, embedding: List[float], ttl: int = 86400):
        self.backend.set(self._key(text, model), json.dumps(embedding), ttl)
    
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {"hits": self._hits, "misses": self._misses, "hit_rate": self._hits / total if total else 0}


_cache = None

def get_cache() -> EmbeddingCache:
    global _cache
    if _cache is None:
        _cache = EmbeddingCache(MemoryCache())
    return _cache
