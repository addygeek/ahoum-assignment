# ACEF - Ahoum Conversation Evaluation Framework

A **production-ready conversation evaluation benchmark** that automatically scores conversation turns across **300+ facets** covering linguistic quality, pragmatics, safety, and emotional/behavioral signals.

## Features

- **Scalable to 5000+ facets** - Adding facets is just adding data rows
- **No one-shot prompting** - All scoring is model-driven and programmatic
- **Confidence estimation** - Every score includes uncertainty quantification
- **Open-weight LLMs** - Uses OpenRouter API (Qwen2-7B, Llama-3-8B, etc.)
- **REST API** - FastAPI endpoints for evaluation
- **Streaming support** - Real-time turn-by-turn evaluation

## Architecture

```
Conversation
    ↓
Turn Segmentation (preprocessor.py)
    ↓
Contextual Encoding (encoder.py - LLM via OpenRouter)
    ↓
Shared Turn Representation
    ↓
Facet Router (evaluator.py)
    ↓
Facet Scoring Heads (scoring_heads.py)
    ↓
Score + Confidence Output (confidence.py)
```

## Project Structure

```
conversation-eval/
├── data/
│   ├── facet_registry.py    # Facet parsing and registry
│   ├── preprocessor.py      # Conversation preprocessing
│   └── sample_conversations.json
├── models/
│   ├── encoder.py           # LLM-based turn encoding
│   ├── scoring_heads.py     # Facet scoring MLPs
│   ├── confidence.py        # Uncertainty estimation
│   └── evaluator.py         # Main evaluation pipeline
├── inference/
│   └── pipeline.py          # Batch/streaming inference
├── api/
│   ├── main.py              # FastAPI endpoints
│   └── schemas.py           # Request/response models
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 3. Generate Facet Registry

```bash
python data/facet_registry.py
```

### 4. Run API Server

```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Evaluate a Conversation

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": {
      "conversation_id": "test_001",
      "turns": [
        {"turn_id": 1, "speaker": "user", "text": "I am stressed"},
        {"turn_id": 2, "speaker": "assistant", "text": "I understand. Let me help."}
      ]
    }
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/facets` | GET | List all facets |
| `/facets/{id}` | GET | Get facet details |
| `/facets/summary` | GET | Registry statistics |
| `/evaluate` | POST | Evaluate conversation |
| `/evaluate/batch` | POST | Batch evaluation |
| `/stream/start` | POST | Start streaming session |
| `/stream/turn` | POST | Add turn for evaluation |
| `/stream/end` | POST | End streaming session |

## Scoring System

### Score Levels (0-4)

| Level | Label | Description |
|-------|-------|-------------|
| 0 | Very Low | Minimal presence |
| 1 | Low | Below average |
| 2 | Neutral | Average/not applicable |
| 3 | High | Above average |
| 4 | Very High | Strong presence |

### Confidence Estimation

Confidence scores (0-1) are computed using:
- Softmax margin (top-2 probability difference)
- Prediction entropy
- Optional MC Dropout variance

### Observability

| Type | Behavior |
|------|----------|
| `implicit_allowed` | Infer from language patterns |
| `explicit_only` | Score only if explicitly mentioned |
| `not_observable` | Always neutral (e.g., physiological) |

## Facet Categories

- **Behavioral** - Actions and habits
- **Cognitive** - Reasoning and thinking
- **Emotional** - Feelings and moods
- **Safety** - Harm and risk detection
- **Spiritual** - Religious and spiritual themes
- **Physiological** - Physical/biological markers
- **Social** - Interpersonal dynamics
- **Personality** - Character traits
- **Linguistic** - Language patterns

## Requirements

- Python 3.10+
- FastAPI
- Pydantic
- Requests
- Optional: sentence-transformers (local embeddings)

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | API key for OpenRouter |
| `FACETS_CSV_PATH` | Path to Facets Assignment.csv |

## License

MIT License
