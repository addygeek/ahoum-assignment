"""
Conversation Preprocessor for ACEF.

Handles parsing and preprocessing of conversations for evaluation,
including context window generation and turn segmentation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json


class Speaker(str, Enum):
    """Speaker types in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Turn:
    """A single turn in a conversation."""
    turn_id: int
    speaker: Speaker
    text: str
    conversation_id: str = ""
    
    # Computed fields (filled during preprocessing)
    prev_turn_text: str = ""
    prev_3_turns: str = ""
    turn_position_ratio: float = 0.0
    word_count: int = 0
    
    def __post_init__(self):
        if isinstance(self.speaker, str):
            self.speaker = Speaker(self.speaker.lower())
        self.word_count = len(self.text.split())
    
    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "speaker": self.speaker.value,
            "text": self.text,
            "conversation_id": self.conversation_id,
            "prev_turn_text": self.prev_turn_text,
            "prev_3_turns": self.prev_3_turns,
            "turn_position_ratio": self.turn_position_ratio,
            "word_count": self.word_count
        }


@dataclass
class Conversation:
    """A complete conversation with metadata."""
    conversation_id: str
    turns: List[Turn] = field(default_factory=list)
    domain_hint: str = ""
    risk_profile: str = "normal"  # normal, elevated, high
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_turns(self) -> int:
        return len(self.turns)
    
    @property
    def user_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.speaker == Speaker.USER]
    
    @property
    def assistant_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.speaker == Speaker.ASSISTANT]
    
    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "total_turns": self.total_turns,
            "domain_hint": self.domain_hint,
            "risk_profile": self.risk_profile,
            "metadata": self.metadata,
            "turns": [t.to_dict() for t in self.turns]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        """Create Conversation from dictionary."""
        turns = []
        for turn_data in data.get("turns", []):
            turn = Turn(
                turn_id=turn_data["turn_id"],
                speaker=turn_data["speaker"],
                text=turn_data["text"],
                conversation_id=data.get("conversation_id", "")
            )
            turns.append(turn)
        
        return cls(
            conversation_id=data.get("conversation_id", "unknown"),
            turns=turns,
            domain_hint=data.get("domain_hint", ""),
            risk_profile=data.get("risk_profile", "normal"),
            metadata=data.get("metadata", {})
        )


class ConversationPreprocessor:
    """
    Preprocessor for conversation data.
    
    Handles:
    - Turn sorting and validation
    - Context window generation
    - Position ratio calculation
    - Risk profile detection
    """
    
    # Keywords that might indicate elevated risk
    RISK_KEYWORDS = {
        "elevated": [
            "stressed", "anxious", "worried", "upset", "frustrated",
            "struggling", "difficult", "hard time", "can't cope"
        ],
        "high": [
            "hurt myself", "end it", "suicide", "kill", "self-harm",
            "hopeless", "worthless", "no point", "want to die",
            "abuse", "violence", "threat"
        ]
    }
    
    def __init__(self, context_window_size: int = 3):
        """
        Initialize preprocessor.
        
        Args:
            context_window_size: Number of previous turns to include in context
        """
        self.context_window_size = context_window_size
    
    def preprocess(self, conversation: Conversation) -> Conversation:
        """
        Preprocess a conversation for evaluation.
        
        Args:
            conversation: Raw conversation to process
            
        Returns:
            Processed conversation with computed fields
        """
        # Sort turns by turn_id
        conversation.turns.sort(key=lambda t: t.turn_id)
        
        total_turns = len(conversation.turns)
        
        for idx, turn in enumerate(conversation.turns):
            # Set conversation ID reference
            turn.conversation_id = conversation.conversation_id
            
            # Calculate turn position ratio (0.0 to 1.0)
            turn.turn_position_ratio = (idx + 1) / total_turns if total_turns > 0 else 0.0
            
            # Get previous turn text
            if idx > 0:
                turn.prev_turn_text = conversation.turns[idx - 1].text
            
            # Get previous N turns as context
            start_idx = max(0, idx - self.context_window_size)
            prev_turns = conversation.turns[start_idx:idx]
            turn.prev_3_turns = " [SEP] ".join([
                f"{t.speaker.value}: {t.text}" for t in prev_turns
            ])
        
        # Detect risk profile
        conversation.risk_profile = self._detect_risk_profile(conversation)
        
        return conversation
    
    def _detect_risk_profile(self, conversation: Conversation) -> str:
        """Detect risk level based on conversation content."""
        all_text = " ".join([t.text.lower() for t in conversation.turns])
        
        # Check for high risk first
        for keyword in self.RISK_KEYWORDS["high"]:
            if keyword in all_text:
                return "high"
        
        # Check for elevated risk
        for keyword in self.RISK_KEYWORDS["elevated"]:
            if keyword in all_text:
                return "elevated"
        
        return "normal"
    
    def process_batch(self, conversations: List[Conversation]) -> List[Conversation]:
        """Process multiple conversations."""
        return [self.preprocess(conv) for conv in conversations]
    
    @staticmethod
    def load_from_json(json_path: str) -> List[Conversation]:
        """Load conversations from a JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = []
        items = data if isinstance(data, list) else data.get("conversations", [data])
        
        for item in items:
            conv = Conversation.from_dict(item)
            conversations.append(conv)
        
        return conversations
    
    @staticmethod
    def save_to_json(conversations: List[Conversation], json_path: str) -> None:
        """Save processed conversations to JSON."""
        data = {
            "conversations": [c.to_dict() for c in conversations]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


def build_turn_context(turn: Turn, conversation: Conversation) -> str:
    """
    Build the full context string for encoding a turn.
    
    Format:
    [Previous context] [SEP] [Target turn]
    
    Args:
        turn: The target turn to build context for
        conversation: The full conversation
        
    Returns:
        Context string for the encoder
    """
    context_parts = []
    
    if turn.prev_3_turns:
        context_parts.append(turn.prev_3_turns)
    
    context_parts.append(f"{turn.speaker.value}: {turn.text}")
    
    return " [SEP] ".join(context_parts) if len(context_parts) > 1 else context_parts[0]


if __name__ == "__main__":
    # Example usage
    sample_conv = Conversation(
        conversation_id="test_001",
        turns=[
            Turn(turn_id=1, speaker="user", text="I'm feeling really stressed about my exam tomorrow."),
            Turn(turn_id=2, speaker="assistant", text="I understand exam stress can be overwhelming. What subject is forcing you?"),
            Turn(turn_id=3, speaker="user", text="It's calculus. I've been studying for weeks but still feel unprepared."),
            Turn(turn_id=4, speaker="assistant", text="It sounds like you've put in significant effort. Let's focus on the key concepts.")
        ]
    )
    
    preprocessor = ConversationPreprocessor()
    processed = preprocessor.preprocess(sample_conv)
    
    print("Processed Conversation:")
    print(json.dumps(processed.to_dict(), indent=2))
