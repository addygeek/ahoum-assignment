"""
Facet Registry - Core scalability mechanism for ACEF.

This module parses the Facets Assignment CSV and creates a structured registry
where facets are treated as data, not logic. Adding facets never requires code changes.
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class FacetCategory(str, Enum):
    """Categories for grouping related facets."""
    BEHAVIORAL = "behavioral"
    COGNITIVE = "cognitive"
    EMOTIONAL = "emotional"
    SAFETY = "safety"
    SPIRITUAL = "spiritual"
    PHYSIOLOGICAL = "physiological"
    SOCIAL = "social"
    PERSONALITY = "personality"
    LINGUISTIC = "linguistic"
    OTHER = "other"


class Observability(str, Enum):
    """How the facet can be observed in conversation."""
    EXPLICIT_ONLY = "explicit_only"      # Score only if explicitly mentioned
    IMPLICIT_ALLOWED = "implicit_allowed" # Can infer from language patterns
    NOT_OBSERVABLE = "not_observable"     # Always neutral (e.g., physiological)


class SignalType(str, Enum):
    """Type of signal the facet represents."""
    LATENT_TRAIT = "latent_trait"        # Underlying characteristic
    STATE = "state"                       # Current condition
    BEHAVIOR = "behavior"                 # Observable action
    SKILL = "skill"                       # Ability or competence
    PREFERENCE = "preference"             # Choice or inclination


class Scope(str, Enum):
    """Scope of turns needed to evaluate the facet."""
    SINGLE_TURN = "single_turn"          # Can be evaluated from one turn
    MULTI_TURN = "multi_turn"            # Requires context from multiple turns


@dataclass
class Facet:
    """A single facet in the evaluation registry."""
    facet_id: int
    name: str                            # snake_case normalized name
    original_name: str                   # Original name from CSV
    category: FacetCategory
    signal_type: SignalType
    scope: Scope
    observability: Observability
    score_type: str = "ordinal"          # Always ordinal (5 levels)
    description: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['category'] = self.category.value
        result['signal_type'] = self.signal_type.value
        result['scope'] = self.scope.value
        result['observability'] = self.observability.value
        return result


class FacetRegistry:
    """
    Registry for all facets in the evaluation system.
    
    This is the core scalability mechanism - adding facets is just adding data rows.
    """
    
    # Keywords for automatic category classification
    CATEGORY_KEYWORDS = {
        FacetCategory.EMOTIONAL: [
            'emotion', 'feeling', 'mood', 'affect', 'happiness', 'sadness', 'anger',
            'joy', 'fear', 'stress', 'anxiety', 'depression', 'merriness', 'bliss',
            'contentment', 'enthusiasm', 'irritability', 'hostility', 'warmth'
        ],
        FacetCategory.COGNITIVE: [
            'reasoning', 'thinking', 'intelligence', 'memory', 'attention', 'learning',
            'problem', 'decision', 'judgment', 'comprehension', 'understanding', 'iq',
            'cognitive', 'mental', 'analytical', 'logical', 'arithmetic', 'numerical'
        ],
        FacetCategory.BEHAVIORAL: [
            'behavior', 'action', 'habit', 'tendency', 'practice', 'activity',
            'risk-taking', 'leadership', 'assertive', 'impulsive', 'compulsive'
        ],
        FacetCategory.SAFETY: [
            'safety', 'harm', 'danger', 'violence', 'abuse', 'drug', 'risk',
            'toxicity', 'hate', 'threat', 'harmful'
        ],
        FacetCategory.SPIRITUAL: [
            'spiritual', 'religious', 'faith', 'prayer', 'meditation', 'church',
            'mosque', 'temple', 'hindu', 'muslim', 'christian', 'buddhist', 'jewish',
            'sikh', 'kabbalah', 'sufi', 'yoga', 'mantra', 'scripture', 'quran',
            'bible', 'torah', 'dharma', 'karma', 'chakra'
        ],
        FacetCategory.PHYSIOLOGICAL: [
            'hormone', 'level', 'blood', 'heart', 'metabolic', 'gene', 'dna',
            'sleep', 'caffeine', 'diet', 'weight', 'bmi', 'physical', 'health',
            'fsh', 'serotonin', 'cortisol', 'immune'
        ],
        FacetCategory.SOCIAL: [
            'social', 'relationship', 'communication', 'interpersonal', 'community',
            'collaboration', 'cooperation', 'teamwork', 'network', 'peer'
        ],
        FacetCategory.PERSONALITY: [
            'personality', 'trait', 'character', 'hexaco', 'big five', 'openness',
            'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism',
            'honest', 'humble'
        ],
        FacetCategory.LINGUISTIC: [
            'language', 'speech', 'verbal', 'vocabulary', 'grammar', 'sentence',
            'spelling', 'writing', 'reading', 'comprehension', 'articulation'
        ]
    }
    
    # Keywords for observability classification
    NOT_OBSERVABLE_KEYWORDS = [
        'hormone', 'level', 'gene', 'dna', 'blood', 'fsh', 'metabolic',
        'serotonin', 'cortisol', 'chromatin', 'polygenic', 'immune-response',
        'basophil', 'parathyroid', 'caffeine sensitivity gene'
    ]
    
    def __init__(self):
        self.facets: Dict[int, Facet] = {}
        self.name_to_id: Dict[str, int] = {}
        
    def normalize_name(self, name: str) -> str:
        """Convert facet name to snake_case."""
        # Remove leading numbers and dots (e.g., "800. Sufi practice:")
        name = re.sub(r'^\d+\.\s*', '', name)
        
        # Remove trailing colons
        name = name.rstrip(':')
        
        # Replace special characters with spaces
        name = re.sub(r'[^a-zA-Z0-9\s]', ' ', name)
        
        # Convert to lowercase and replace spaces with underscores
        name = name.lower().strip()
        name = re.sub(r'\s+', '_', name)
        
        # Remove consecutive underscores
        name = re.sub(r'_+', '_', name)
        
        return name
    
    def generate_facet_id(self, name: str, index: int) -> int:
        """Generate unique facet ID based on name hash and index."""
        # Use index as base, but ensure uniqueness
        return index + 1
    
    def classify_category(self, name: str) -> FacetCategory:
        """Automatically classify facet into a category based on keywords."""
        name_lower = name.lower()
        
        # Check each category's keywords
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in name_lower:
                    return category
        
        return FacetCategory.OTHER
    
    def classify_observability(self, name: str) -> Observability:
        """Determine if facet can be observed from conversation text."""
        name_lower = name.lower()
        
        # Check for non-observable (physiological, genetic, etc.)
        for keyword in self.NOT_OBSERVABLE_KEYWORDS:
            if keyword in name_lower:
                return Observability.NOT_OBSERVABLE
        
        # Most behavioral/emotional facets can be implicit
        category = self.classify_category(name)
        if category in [FacetCategory.EMOTIONAL, FacetCategory.BEHAVIORAL, 
                        FacetCategory.PERSONALITY, FacetCategory.COGNITIVE]:
            return Observability.IMPLICIT_ALLOWED
        
        # Spiritual/religious might need explicit mentions
        if category == FacetCategory.SPIRITUAL:
            return Observability.EXPLICIT_ONLY
        
        return Observability.IMPLICIT_ALLOWED
    
    def classify_signal_type(self, name: str, category: FacetCategory) -> SignalType:
        """Determine the signal type for a facet."""
        name_lower = name.lower()
        
        # Skills
        skill_keywords = ['skill', 'ability', 'reasoning', 'intelligence', 'comprehension']
        for kw in skill_keywords:
            if kw in name_lower:
                return SignalType.SKILL
        
        # States (temporary conditions)
        state_keywords = ['mood', 'feeling', 'stress', 'happiness', 'depression symptom']
        for kw in state_keywords:
            if kw in name_lower:
                return SignalType.STATE
        
        # Behaviors
        if category == FacetCategory.BEHAVIORAL:
            return SignalType.BEHAVIOR
        
        # Preferences
        preference_keywords = ['preference', 'style', 'orientation']
        for kw in preference_keywords:
            if kw in name_lower:
                return SignalType.PREFERENCE
        
        # Default to latent trait
        return SignalType.LATENT_TRAIT
    
    def classify_scope(self, category: FacetCategory, signal_type: SignalType) -> Scope:
        """Determine if single or multi-turn analysis is needed."""
        # States and immediate behaviors can often be single-turn
        if signal_type in [SignalType.STATE, SignalType.BEHAVIOR]:
            return Scope.SINGLE_TURN
        
        # Traits and skills typically need more context
        return Scope.MULTI_TURN
    
    def load_from_csv(self, csv_path: str) -> None:
        """Load facets from the CSV file and create registry."""
        path = Path(csv_path).resolve()
        
        if not path.exists():
            # Try to resolve relative to current file if not found
            fallback = list(Path.cwd().rglob(path.name))
            if fallback:
                path = fallback[0]
            else:
                # Try finding in typical locations
                potential_roots = [
                    Path.cwd(),
                    Path.cwd().parent,
                    Path(__file__).parent.parent.parent
                ]
                for root in potential_roots:
                    candidate = root / path.name
                    if candidate.exists():
                        path = candidate
                        break
        
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path} (Resolved to: {path})")
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header row
        facet_names = [line.strip() for line in lines[1:] if line.strip()]
        
        for idx, original_name in enumerate(facet_names):
            normalized_name = self.normalize_name(original_name)
            
            # Skip if empty after normalization
            if not normalized_name:
                continue
            
            # Handle duplicates by appending index
            base_name = normalized_name
            counter = 1
            while normalized_name in self.name_to_id:
                normalized_name = f"{base_name}_{counter}"
                counter += 1
            
            # Classify the facet
            category = self.classify_category(original_name)
            observability = self.classify_observability(original_name)
            signal_type = self.classify_signal_type(original_name, category)
            scope = self.classify_scope(category, signal_type)
            
            # Create facet
            facet_id = self.generate_facet_id(normalized_name, idx)
            facet = Facet(
                facet_id=facet_id,
                name=normalized_name,
                original_name=original_name,
                category=category,
                signal_type=signal_type,
                scope=scope,
                observability=observability,
                description=f"Measures {original_name.lower().rstrip(':')} in conversation."
            )
            
            self.facets[facet_id] = facet
            self.name_to_id[normalized_name] = facet_id
    
    def get_by_id(self, facet_id: int) -> Optional[Facet]:
        """Get facet by ID."""
        return self.facets.get(facet_id)
    
    def get_by_name(self, name: str) -> Optional[Facet]:
        """Get facet by normalized name."""
        facet_id = self.name_to_id.get(name)
        return self.facets.get(facet_id) if facet_id else None
    
    def get_by_category(self, category: FacetCategory) -> List[Facet]:
        """Get all facets in a category."""
        return [f for f in self.facets.values() if f.category == category]
    
    def get_observable_facets(self) -> List[Facet]:
        """Get all facets that can be observed from conversation."""
        return [
            f for f in self.facets.values() 
            if f.observability != Observability.NOT_OBSERVABLE
        ]
    
    def to_json(self, output_path: str) -> None:
        """Export registry to JSON file."""
        data = {
            "version": "1.0",
            "total_facets": len(self.facets),
            "facets": [f.to_dict() for f in self.facets.values()]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def summary(self) -> dict:
        """Get summary statistics of the registry."""
        categories = {}
        observability = {}
        
        for facet in self.facets.values():
            cat = facet.category.value
            obs = facet.observability.value
            
            categories[cat] = categories.get(cat, 0) + 1
            observability[obs] = observability.get(obs, 0) + 1
        
        return {
            "total_facets": len(self.facets),
            "by_category": categories,
            "by_observability": observability
        }


def create_registry_from_csv(csv_path: str, output_path: str = None) -> FacetRegistry:
    """
    Create a facet registry from the CSV file.
    
    Args:
        csv_path: Path to Facets Assignment.csv
        output_path: Optional path to save JSON registry
    
    Returns:
        Populated FacetRegistry instance
    """
    registry = FacetRegistry()
    registry.load_from_csv(csv_path)
    
    if output_path:
        registry.to_json(output_path)
    
    return registry


if __name__ == "__main__":
    # Generate registry from CSV
    import sys
    
    # Default paths
    csv_path = Path(__file__).parent.parent.parent / "Facets Assignment.csv"
    output_path = Path(__file__).parent / "facets" / "facets_registry.json"
    
    # Create facets directory if not exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading facets from: {csv_path}")
    registry = create_registry_from_csv(str(csv_path), str(output_path))
    
    print(f"\nRegistry Summary:")
    summary = registry.summary()
    print(f"Total Facets: {summary['total_facets']}")
    print(f"\nBy Category:")
    for cat, count in sorted(summary['by_category'].items()):
        print(f"  {cat}: {count}")
    print(f"\nBy Observability:")
    for obs, count in sorted(summary['by_observability'].items()):
        print(f"  {obs}: {count}")
    
    print(f"\nRegistry saved to: {output_path}")
