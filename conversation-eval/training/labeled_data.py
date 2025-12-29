"""Labeled data handling for training."""

import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class LabeledExample:
    conversation_id: str
    turn_id: int
    text: str
    context: str
    facet_id: int
    facet_name: str
    cluster: str
    ground_truth_score: int
    annotator_id: str = ""
    confidence: float = 1.0


class LabeledDataset:
    def __init__(self, examples: List[LabeledExample] = None):
        self.examples = examples or []
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def add(self, example: LabeledExample):
        self.examples.append(example)
    
    def save(self, path: str):
        data = {"version": "1.0", "count": len(self.examples), 
                "examples": [asdict(e) for e in self.examples]}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls([LabeledExample(**e) for e in data["examples"]])
    
    def split(self, ratio=0.8, seed=42) -> Tuple["LabeledDataset", "LabeledDataset"]:
        import random
        random.seed(seed)
        examples = self.examples.copy()
        random.shuffle(examples)
        idx = int(len(examples) * ratio)
        return LabeledDataset(examples[:idx]), LabeledDataset(examples[idx:])
    
    def stats(self) -> Dict:
        from collections import Counter
        if not self.examples:
            return {"total": 0}
        return {
            "total": len(self.examples),
            "scores": dict(Counter(e.ground_truth_score for e in self.examples)),
            "facets": len(set(e.facet_id for e in self.examples)),
            "conversations": len(set(e.conversation_id for e in self.examples))
        }
