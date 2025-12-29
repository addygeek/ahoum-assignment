"""Score explainability for ACEF."""

import re
from typing import List, Dict
from dataclasses import dataclass, asdict


@dataclass
class Evidence:
    text: str
    start: int
    end: int
    relevance: float
    indicator: str  # positive/negative


@dataclass
class Explanation:
    score: int
    label: str
    confidence: float
    evidence: List[Evidence]
    reasoning: str
    
    def to_dict(self):
        return {**asdict(self), "evidence": [asdict(e) for e in self.evidence]}


class ScoreExplainer:
    KEYWORDS = {
        "emotional": {
            "positive": ["happy", "excited", "grateful", "calm", "joy", "pleased"],
            "negative": ["sad", "angry", "stressed", "anxious", "worried", "frustrated"]
        },
        "behavioral": {
            "positive": ["helped", "supported", "encouraged", "guided"],
            "negative": ["ignored", "refused", "dismissed"]
        },
        "cognitive": {
            "positive": ["understand", "analyze", "solved", "figured"],
            "negative": ["confused", "unclear", "misunderstood"]
        },
        "safety": {
            "negative": ["harm", "danger", "risk", "hurt", "violence"],
            "positive": ["safe", "protect", "help", "support"]
        }
    }
    
    LABELS = ["very_low", "low", "neutral", "high", "very_high"]
    
    def explain(self, text, facet_name, category, score, confidence, components=None):
        evidence = self._find_evidence(text, category)
        reasoning = self._reason(facet_name, score, evidence, confidence)
        return Explanation(score, self.LABELS[score], confidence, evidence, reasoning)
    
    def _find_evidence(self, text, category) -> List[Evidence]:
        result = []
        text_lower = text.lower()
        keywords = self.KEYWORDS.get(category, {})
        
        for typ, words in keywords.items():
            for word in words:
                for m in re.finditer(rf'\b{re.escape(word)}\b', text_lower):
                    start = max(0, m.start() - 25)
                    end = min(len(text), m.end() + 25)
                    result.append(Evidence(text[start:end].strip(), start, end, 0.8, typ))
        
        return result[:5]  # limit
    
    def _reason(self, facet, score, evidence, conf):
        label = self.LABELS[score]
        pos = sum(1 for e in evidence if e.indicator == "positive")
        neg = sum(1 for e in evidence if e.indicator == "negative")
        
        parts = [f"'{facet}' scored as '{label}'."]
        if pos: parts.append(f"{pos} positive indicator(s).")
        if neg: parts.append(f"{neg} negative indicator(s).")
        if not evidence: parts.append("Based on overall analysis.")
        parts.append(f"Confidence: {conf:.0%}.")
        
        return " ".join(parts)
