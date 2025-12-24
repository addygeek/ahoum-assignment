# ACEF - Ahoum Conversation Evaluation Framework

A production-ready conversation evaluation benchmark that scores conversation turns across 300+ facets covering linguistic quality, pragmatics, safety, and emotional/behavioral signals.

## Features

- Scalable to 5000+ facets without code changes
- Model-driven scoring (no one-shot prompting)
- Confidence estimation for all scores
- Open-weight LLM support via OpenRouter (Qwen2-7B, Llama-3-8B, etc.)
- REST API with FastAPI
- Streaming evaluation support
- Visual Streamlit interface

## Architecture

```
Conversation
    |
    v
Turn Segmentation (preprocessor.py)
    |
    v
Contextual Encoding (encoder.py)
    |
    v
Shared Turn Representation
    |
    v
Facet Router (evaluator.py)
    |
    v
Facet Scoring Heads (scoring_heads.py)
    |
    v
Score + Confidence Output (confidence.py)
```

## Project Structure

```
conversation-eval/
├── data/
│   ├── facet_registry.py         # Facet parsing and registry
│   ├── preprocessor.py           # Conversation preprocessing
│   └── sample_conversations.json # Test data
├── models/
│   ├── encoder.py                # LLM-based turn encoding
│   ├── scoring_heads.py          # Facet scoring MLPs
│   ├── confidence.py             # Uncertainty estimation
│   └── evaluator.py              # Main evaluation pipeline
├── inference/
│   └── pipeline.py               # Batch/streaming inference
├── api/
│   ├── main.py                   # FastAPI endpoints
│   └── schemas.py                # Request/response models
├── ui/
│   └── app.py                    # Streamlit interface
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
# Add your OPENROUTER_API_KEY to .env
```

### 3. Generate Facet Registry

```bash
python data/facet_registry.py
```

### 4. Run API Server

```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Run Streamlit UI

```bash
streamlit run ui/app.py --server.port 8501
```

### 6. Evaluate a Conversation (via API)

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": {
      "conversation_id": "test_001",
      "turns": [
        {"turn_id": 1, "speaker": "user", "text": "I am stressed"},
        {"turn_id": 2, "speaker": "assistant", "text": "Let me help."}
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
| 0 | very_low | Minimal presence |
| 1 | low | Below average |
| 2 | neutral | Average/not applicable |
| 3 | high | Above average |
| 4 | very_high | Strong presence |

### Confidence Estimation

Confidence scores (0-1) are computed using:
- Softmax margin (difference between top-2 predictions)
- Prediction entropy (uncertainty in distribution)
- Optional MC Dropout variance

### Observability Types

| Type | Behavior |
|------|----------|
| `implicit_allowed` | Infer from language patterns |
| `explicit_only` | Score only if explicitly mentioned |
| `not_observable` | Always neutral (e.g., physiological) |

## Facet Categories

- Behavioral - Actions and habits
- Cognitive - Reasoning and thinking
- Emotional - Feelings and moods
- Safety - Harm and risk detection
- Spiritual - Religious and spiritual themes
- Physiological - Physical/biological markers
- Social - Interpersonal dynamics
- Personality - Character traits
- Linguistic - Language patterns

## Requirements

- Python 3.10+
- FastAPI
- Pydantic
- Streamlit
- Plotly
- Requests

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | API key for OpenRouter |
| `FACETS_CSV_PATH` | Path to Facets Assignment.csv |

## License

MIT License
