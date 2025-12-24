Alright. Now we do this **properly**.

I’ll write this as a **full Product Requirements Document (PRD)** exactly the way a strong AI/ML engineer or research engineer would submit it.
This is not an explanation anymore — this is **the blueprint you execute**.

I’ll cover **everything**:

* What the product is
* What problems it solves
* What exactly to build
* How to architect it
* How to approach data, facets, models, scaling
* How to meet every constraint
* How to deliver it cleanly

You can almost copy-paste this into your GitHub README / design doc.

---

# 📘 PRD: Conversation Evaluation Benchmark System

**Project Name:** Ahoum Conversation Evaluation Framework (ACEF)

---

## 1. Overview

### 1.1 Objective

Design and implement a **production-ready conversation evaluation benchmark** that automatically scores **each turn in a conversation** across **300 distinct facets**, covering:

* Linguistic quality
* Pragmatics
* Safety
* Emotional and behavioral signals

The system must be **architecturally scalable to 5000+ facets** without redesign and must use **open-weight LLMs (≤16B parameters)**. The benchmark must output **discrete ordinal scores** with **confidence estimates**, and be reproducible, extensible, and deployable.

---

## 2. Problem Statement

Current LLM evaluation approaches suffer from:

* Prompt-based, brittle scoring
* Poor scalability across hundreds of dimensions
* Lack of confidence estimation
* Monolithic designs that collapse at scale

This project addresses these issues by designing a **facet-driven, modular, ML-based evaluation pipeline** where facets are treated as **data**, not prompts.

---

## 3. Core Design Principles

1. **Facet-as-Data, not Logic**

   * Facets are defined declaratively in a registry
   * Adding facets never requires code changes

2. **Shared Representation, Multiple Judgments**

   * One encoding per turn
   * Thousands of facet evaluations reuse it

3. **No One-Shot Prompting**

   * No “Score this conversation” prompts
   * All scoring is model-driven and programmatic

4. **Observability-Aware Scoring**

   * Some facets are implicit
   * Some require explicit mention
   * Neutral/unknown is a valid outcome

5. **Production-First Thinking**

   * Deterministic outputs
   * Confidence scores
   * Dockerized, reproducible pipeline

---

## 4. Functional Requirements

### 4.1 Input

* A conversation consisting of:

  * `conversation_id`
  * Ordered turns:

    * `turn_id`
    * `speaker` (user / assistant)
    * `text`

### 4.2 Output

For **every turn × every facet**:

* Ordinal score (5 ordered integers)
* Confidence score (0–1)
* Optional `not_observable` flag

---

## 5. Dataset Handling

### 5.1 Facet Dataset (Facets Assignment.csv)

The provided dataset contains ~400 facets such as:

* Risk-taking
* Naivety
* Merriness
* Statistical Reasoning
* Compassion Fatigue
* Religious practices
* Emotional states
* Cognitive skills

These facets vary greatly in **abstraction, observability, and scope**.

---

## 6. Data Cleaning & Preprocessing

### 6.1 Facet Normalization

Each raw facet name is normalized into:

* `snake_case`
* Unique `facet_id`
* Canonical category

Example:

```
"Democratic Leadership:" → democratic_leadership
```

### 6.2 Conversation Preprocessing

For each conversation:

* Sort turns by `turn_id`
* Generate rolling context windows

### 6.3 Final Processed Dataset Schema

#### Conversation Table

```
conversation_id
total_turns
domain_hint
risk_profile
```

#### Turn Table

```
conversation_id
turn_id
speaker
text
prev_turn_text
prev_3_turns
turn_position_ratio
```

---

## 7. Facet Registry (Critical Artifact)

### 7.1 Purpose

The facet registry is the **core scalability mechanism**.
It defines *what* is evaluated, not *how*.

### 7.2 Facet Registry Schema

```json
{
  "facet_id": 214,
  "name": "risk_taking",
  "category": "behavioral",
  "signal_type": "latent_trait",
  "scope": "multi_turn",
  "observability": "implicit_allowed",
  "score_type": "ordinal",
  "description": "Degree to which the speaker expresses willingness to take risks."
}
```

### 7.3 Observability Rules

| Observability    | Behavior                           |
| ---------------- | ---------------------------------- |
| explicit_only    | Score only if explicitly mentioned |
| implicit_allowed | Infer from language                |
| not_observable   | Always neutral                     |

This prevents hallucinated judgments.

---

## 8. Model Architecture

### 8.1 Backbone Model

* **Qwen2-7B** or **Llama-3-8B (Instruct) make it swithcabel using oenrouter api**
* Open-weights
* ≤16B parameters
* Used as a **representation encoder**, not a judge

---

## 9. System Architecture

### 9.1 High-Level Pipeline

```
Conversation
  ↓
Turn Segmentation
  ↓
Contextual Encoding (LLM)
  ↓
Shared Turn Representation
  ↓
Facet Router
  ↓
Facet Scoring Heads
  ↓
Score + Confidence Output
```

---

## 10. Representation Layer

For each turn:

* Input: `[context + target_turn]`
* Output:

  * Dense vector embedding
  * Auxiliary signals (sentiment, toxicity, uncertainty)

This step runs **once per turn**, regardless of facet count.

---

## 11. Facet Routing Layer

A lightweight classifier maps:

```
(turn_embedding + facet_metadata) → facet_cluster
```

Clusters may include:

* Linguistic quality
* Emotional affect
* Safety
* Behavioral traits
* Cognitive reasoning

This prevents running thousands of independent heads.

---

## 12. Facet Scoring Layer (Scalable Core)

Each cluster has a shared scoring head:

```
score_logits = MLP([turn_embedding ; facet_embedding])
```

* Facet embeddings are learned or initialized
* Adding facets = adding rows
* No architectural changes needed for 5000+ facets

---

## 13. Scoring System

### 13.1 Score Scale

Five ordered integers:

```
Very Low → Low → Neutral → High → Very High
```

Stored as integers, meaning defined externally.

---

## 14. Confidence Estimation

Confidence is computed using:

1. Softmax margin
2. Prediction entropy
3. Optional MC Dropout variance

Final confidence:

```
confidence = 1 − normalized_uncertainty
```

---

## 15. Safety & Emotion Handling

### Safety Facets

* Intent detection
* Response appropriateness
* De-escalation quality
* Harm avoidance

### Emotional Facets

* Emotion expressed
* Emotional alignment
* Emotional regulation

Each is scored independently.

---

## 16. Sample Conversations Dataset

### 16.1 Requirements

* ≥50 conversations
* 5–10 turns each
* Covers:

  * Emotional distress
  * Safety edge cases
  * Sarcasm
  * Reasoning
  * Cultural references

### 16.2 Format

```json
{
  "conversation_id": "conv_001",
  "turns": [...],
  "scores": {
    "turn_3": {
      "risk_taking": { "score": 5, "confidence": 0.91 }
    }
  }
}
```

Manual scoring is acceptable.

---

## 17. Dockerized Baseline

### Stack

* Python 3.10
* PyTorch
* HuggingFace Transformers
* FastAPI
* SQLite / DuckDB

### Command

```bash
docker compose up
```

---

## 18. Sample UI (Optional)

Features:

* Upload conversation JSON
* Select facet groups
* Heatmap visualization
* Confidence overlays

Built using:

* Streamlit or Next.js

---

## 19. Repository Structure

```
conversation-eval/
├── data/
├── facets/
├── models/
├── inference/
├── api/
├── ui/
├── docker/
├── README.md
└── PRD.md
```

---

## 20. Deliverables Mapping

| Requirement      | Covered          |
| ---------------- | ---------------- |
| 300 facets       | Facet Registry   |
| Scales to 5000   | Facet embeddings |
| No prompting     | Model heads      |
| Open-weights     | Llama/Qwen       |
| Confidence       | Entropy + margin |
| Docker           | Included         |
| UI               | Optional         |
| 50 conversations | Included         |

---

## 21. Final Mental Model

> This system does **not judge people**.
> It scores **signals expressed in language**, across reusable facets, using a shared representation.

That sentence alone shows senior-level understanding.

---
