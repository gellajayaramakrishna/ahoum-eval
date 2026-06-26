# Ahoum — Conversation Evaluation Benchmark

A production-ready benchmark system that evaluates conversations across **399 distinct facets** covering:

- Linguistic Quality
- Pragmatics
- Safety
- Emotion

The system uses **open-weight LLMs (Llama 3.1 8B)** through the Groq API and is architected to scale from **399 → 5000+ facets without redesign**.

---

# Overview

This project evaluates conversations by scoring each facet on a **1–5 scale** along with a **confidence score (0.0–1.0)**.

Instead of using one-shot prompting, the system processes facets in configurable batches, making it scalable, modular, and production-friendly.

---

# System Architecture

```text
Input Conversation
        ↓
Facet Loader (facets.py)
        ↓
Facet Batching Engine
        ↓
Llama 3.1 8B via Groq API
        ↓
Structured JSON Scores
        ↓
Confidence Calculation
        ↓
Flask Dashboard UI
```

---

# Why This Scales to 5000+ Facets

The architecture uses configurable batch processing.

Instead of evaluating all facets in a single prompt, facets are processed in groups (default: 30 facets per batch).

This means scaling from:

- 399 facets
to
- 5000+ facets

requires **no architectural redesign** — only additional batch iterations.

```python
score_conversation(
    conversation,
    facets,
    batch_size=30
)
```

---

# Hard Constraints Met

| Constraint | Implementation |
|---|---|
| No one-shot prompting | Batched facet evaluation |
| Open-weights ≤16B | Llama 3.1 8B |
| Scales to 5000+ facets | Configurable batching architecture |
| Ordered score scale | 1–5 scoring system |

---

# Brownie Point Features

| Feature | Status |
|---|---|
| Confidence outputs | ✅ Included per facet |
| Sample UI | ✅ Flask dashboard |
| Dockerised baseline | ✅ docker-compose support |

---

# Tech Stack

| Component | Technology |
|---|---|
| Model | Llama 3.1 8B |
| Inference | Groq API |
| Backend | Python |
| UI | Flask |
| Data Processing | Pandas |
| Containerization | Docker |

---

# Project Structure

```text
ahoum-eval/
│
├── app.py                      # Flask dashboard UI
├── scorer.py                   # Core batched scoring engine
├── score_all.py                # Scores all generated conversations
├── facets.py                   # Loads and manages facets
├── preprocess.py               # CSV preprocessing + feature engineering
├── generate_conversations.py   # Generates sample conversations
│
├── conversations/
│   └── conversations.json
│
├── results/
│   └── all_scores.json
│
├── templates/
│   └── index.html
│
├── Facets_Assignment.csv       # Original dataset
├── Facets_Cleaned.csv          # Processed dataset
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── PROMPT_LOG.md
```

---

# Setup

## 1. Clone Repository

```bash
git clone https://github.com/gellajayaramakrishna/ahoum-eval.git
cd ahoum-eval
```

---

## 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Configure Environment Variables

```bash
cp .env.example .env
```

Add your Groq API key inside `.env`:

```env
GROQ_API_KEY=your_api_key_here
```

---

# Running the Project

## Run Full Evaluation

```bash
python score_all.py
```

---

## Launch Flask UI

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

---

# Docker Setup

## Start Entire System

```bash
docker-compose up --build
```

---

# Confidence Scoring

Each facet score includes:

- score → integer from 1–5
- confidence → float from 0.0–1.0

Example:

```json
{
  "facet": "empathy",
  "score": 4,
  "confidence": 0.91
}
```

---

# Example Evaluation Categories

The benchmark evaluates conversations across areas like:

- Toxicity
- Emotional Support
- Manipulation
- Clarity
- Empathy
- Instruction Following
- Hallucination Risk
- Persuasiveness
- Helpfulness
- Professionalism

---

# Production Features

- Resume-safe evaluation pipeline
- Automatic batch retry handling
- JSON parse recovery
- Deterministic scoring (`temperature=0`)
- Configurable batch sizes
- Docker support
- Modular architecture

---

# Limitations

LLM-based evaluation still contains slight non-determinism (~2–3% variance) even at `temperature=0`.

For production environments:
- scores should be cached
- or averaged across multiple runs

---

# Prompt Logging

All prompts and responses used during evaluation are documented in:

```text
PROMPT_LOG.md
```

---

# Model Information

| Field | Value |
|---|---|
| Model | llama-3.1-8b-instant |
| Provider | Groq |
| License | Meta Open Weights |
| Parameter Size | 8B |

---

# Author

**Gella Jaya Rama Krishna**  
AI/ML Internship Assignment  
IIT Gandhinagar
