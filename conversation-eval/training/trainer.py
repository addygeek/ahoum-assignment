"""Training pipeline for ACEF."""

import os
import json
import logging
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    eval_every: int = 100
    early_stopping: int = 5
    device: str = "cpu"


class TrainingMetrics:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({"train": self.train_losses, "val": self.val_losses, "acc": self.val_accuracies}, f)


class SimpleTrainer:
    """Trainer for facet scoring models."""
    
    def __init__(self, scorer, encoder, config: TrainingConfig = None):
        self.scorer = scorer
        self.encoder = encoder
        self.config = config or TrainingConfig()
        self.metrics = TrainingMetrics()
        self.best_loss = float("inf")
        self.patience = 0
    
    def _loss(self, preds, labels):
        import math
        total = 0
        for p, gt in zip(preds, labels):
            total -= math.log(max(p["probabilities"][gt], 1e-10))
        return total / len(preds) if preds else 0
    
    def _accuracy(self, preds, labels):
        correct = sum(1 for p, gt in zip(preds, labels) if p["score"] == gt)
        return correct / len(preds) if preds else 0
    
    def evaluate(self, data: List[Dict]) -> Dict:
        preds, labels = [], []
        for item in data:
            text = f"{item.get('context', '')} [SEP] {item['text']}" if item.get('context') else item['text']
            emb = self.encoder.encode_turn(text)
            pred = self.scorer.score_facet(emb, item["facet_id"], item["cluster"])
            preds.append(pred)
            labels.append(item["ground_truth_score"])
        return {"loss": self._loss(preds, labels), "accuracy": self._accuracy(preds, labels)}
    
    def train(self, train_data, val_data, output_dir="checkpoints", callback=None):
        Path(output_dir).mkdir(exist_ok=True)
        logger.info(f"Training: {len(train_data)} examples, val: {len(val_data)}")
        
        step = 0
        for epoch in range(self.config.epochs):
            for i in range(0, len(train_data), self.config.batch_size):
                batch = train_data[i:i+self.config.batch_size]
                step += 1
                
                if step % self.config.eval_every == 0:
                    metrics = self.evaluate(val_data)
                    logger.info(f"Step {step}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")
                    
                    if metrics["loss"] < self.best_loss:
                        self.best_loss = metrics["loss"]
                        self.patience = 0
                    else:
                        self.patience += 1
                    
                    if self.patience >= self.config.early_stopping:
                        logger.info("Early stopping")
                        return
                
                if callback:
                    callback(step)
        
        self.metrics.save(os.path.join(output_dir, "metrics.json"))
