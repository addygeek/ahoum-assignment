# Getting Started with ACEF

This guide walks you through setting up and running the Ahoum Conversation Evaluation Framework locally.

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git (optional, for cloning)

## Installation

### Step 1: Clone or Download the Repository

```bash
git clone https://github.com/addygeek/ahoum-assignment.git
cd ahoum-assignment
```

Or download and extract the ZIP file.

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate the virtual environment:

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
cd conversation-eval
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the `conversation-eval` directory:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key:

```
OPENROUTER_API_KEY=your_api_key_here
```

You can get an API key from [OpenRouter](https://openrouter.ai/).

Note: The system will work without an API key using hash-based embeddings for demonstration purposes.

## Running the System

### Option 1: Run the API Server

Start the FastAPI server:

```bash
cd conversation-eval
$env:OPENROUTER_API_KEY="your_api_key_here"  # Windows PowerShell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Access the API documentation at: http://localhost:8000/docs

### Option 2: Run the Streamlit UI

Start the Streamlit interface:

```bash
cd conversation-eval
streamlit run ui/app.py --server.port 8501
```

Access the web interface at: http://localhost:8501

### Option 3: Run Both (Recommended)

Open two terminal windows:

**Terminal 1 - API Server:**
```bash
cd conversation-eval
$env:OPENROUTER_API_KEY="your_api_key_here"
uvicorn api.main:app --port 8000
```

**Terminal 2 - Streamlit UI:**
```bash
cd conversation-eval
streamlit run ui/app.py --server.port 8501
```

## Testing the Installation

### Run Integration Test

```bash
cd conversation-eval
python test_integration.py
```

Expected output:
```
============================================================
ACEF - Complete System Integration Test
============================================================

[1] Loading Facets from CSV...
    Loaded 399 facets
...
COMPLETE SYSTEM INTEGRATION TEST PASSED
============================================================
```

### Test via API

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": {
      "conversation_id": "test_001",
      "turns": [
        {"turn_id": 1, "speaker": "user", "text": "Hello, I need help."},
        {"turn_id": 2, "speaker": "assistant", "text": "I am here to help you."}
      ]
    },
    "facet_ids": [1, 2, 3]
  }'
```

### Test via Streamlit

1. Open http://localhost:8501
2. Select a template conversation or enter custom text
3. Select facets to evaluate
4. Click "Evaluate Conversation"
5. View results in the Heatmap, Charts, or Table tabs

## Project Structure

```
ahoum-assignment/
├── Facets Assignment.csv      # Input facet definitions
├── README.md                  # Project overview
├── start.md                   # This file
└── conversation-eval/
    ├── data/                  # Data processing modules
    ├── models/                # ML model components
    ├── inference/             # Inference pipeline
    ├── api/                   # FastAPI REST API
    ├── ui/                    # Streamlit interface
    ├── requirements.txt       # Python dependencies
    └── test_integration.py    # Integration test
```

## Troubleshooting

### "CSV file not found" Error

Ensure the `Facets Assignment.csv` file is in the root project directory (same level as `conversation-eval` folder).

### "OPENROUTER_API_KEY not set" Warning

This is expected if you haven't set an API key. The system will use hash-based embeddings for demonstration. For full LLM-based evaluation, set your OpenRouter API key.

### Port Already in Use

If ports 8000 or 8501 are in use, specify different ports:

```bash
uvicorn api.main:app --port 8001
streamlit run ui/app.py --server.port 8502
```

## Next Steps

- Review the [README.md](README.md) for architecture details
- Explore the API documentation at http://localhost:8000/docs
- Try different facet combinations in the Streamlit UI
- Check `data/sample_conversations.json` for example conversations
