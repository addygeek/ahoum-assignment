"""Logging configuration for ACEF."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None, json_format: bool = False):
    """Configure application logging."""
    
    if json_format:
        fmt = '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
    else:
        fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    
    formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
    
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    root.handlers = []
    
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    
    # Reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    return root


def get_logger(name: str):
    return logging.getLogger(name)
