"""Retry utilities with exponential backoff."""

import time
import logging
from functools import wraps
from typing import Type, Tuple, Callable, Any
import random

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    jitter: bool = True
) -> Callable:
    """Retry decorator with exponential backoff."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        if jitter:
                            delay *= (0.5 + random.random())
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                        time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


class RetryConfig:
    """Retry configuration."""
    
    def __init__(self, max_retries=3, base_delay=1.0, max_delay=60.0, jitter=True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
