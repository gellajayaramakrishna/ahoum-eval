# Ahoum — Conversation Evaluation Benchmark

A production-ready benchmark system that evaluates conversations across **399 distinct facets** covering:

- Linguistic Quality
- Pragmatics
- Safety
- Emotion

The system uses **open-weight LLMs (Llama 3.1 8B)** through the Groq API and is architected to scale from **399 → 5000+ facets without redesign**.

---

## 🌐 Live Demo

**[https://ahoum-eval.onrender.com](https://ahoum-eval.onrender.com)**

No setup required — paste or build a conversation and evaluate it instantly.

---

## Overview

This project evaluates conversations by scoring each facet on a **1–5 scale** along with a **confidence score (0.0–1.0)**.

Instead of using one-shot prompting, the system processes facets in configurable batches, making it scalable, modular, and production-friendly.

---

## System Architecture

```text
Input Conversation
        ↓
Facet Loader (facets.py)
        ↓
Facet Batching Engine (concurrent ThreadPoolExecutor)
        ↓
Llama 3.1 8B via Groq API
        ↓
Structured JSON Scores
        ↓
Confidence Calculation
        ↓
Async Job Polling (Flask + background threads)
        ↓
Live Dashboard UI
```

---

## Why This Scales to 5000+ Facets

The architecture uses configurable concurrent batch processing.

Instead of evaluating all facets in a single prompt, facets are processed in parallel batches. This means scaling from 399 to 5000+ facets requires **no architectural redesign** — only additional batch iterations.

```python
score_conversation(
    conversation,
    facets,
    batch_size=25,
    max_workers=2
)
```

---

## Hard Constraints Met

| Constraint | Implementation |
|---|---|
| No one-shot prompting | Batched facet evaluation |
| Open-weights ≤16B | Llama 3.1 8B |
| Scales to 5000+ facets | Concurrent batching architecture |
| Ordered score scale | 1–5 scoring system |

---

## Brownie Point Features

| Feature | Status |
|---|---|
| Confidence outputs | ✅ Included per facet |
| Sample UI | ✅ Flask dashboard |
| Dockerised baseline | ✅ docker-compose support |
| Live hosted demo | ✅ Render deployment |
| Guided conversation builder | ✅ Build turn-by-turn in UI |
| Async scoring with live progress | ✅ Background jobs + polling |

---

## What's New (v2)

### Async Job System
Scoring no longer blocks the HTTP request. The `/score` endpoint immediately returns a `job_id`, and the frontend polls `/status/<job_id>` every 1.5 seconds for live batch progress.

### Concurrent Batch Scoring
Replaced sequential batching (with fixed 12s sleep) with `ThreadPoolExecutor`. Batches run concurrently, cutting total scoring time significantly.

### Smarter Retry Logic
- Rate limit errors (429): exponential backoff instead of fixed wait
- Malformed/truncated JSON responses: automatic retry up to 4 times
- Missing facets silently skipped by model: filled with neutral defaults

### Robust Score Parsing
- Canonical facet name lookup — handles model returning mangled casing/punctuation
- Hard clamping: scores always land in 1–5, confidence in 0.0–1.0

### Guided Conversation Builder
A new UI mode lets you build conversations turn-by-turn with named speakers instead of pasting raw text. Supports 2–5 speakers with editable/deletable turns.

### Render Deployment Ready
- Binds to `$PORT` from environment
- `Procfile` and `render.yaml` included
- `gunicorn` added to requirements

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | Llama 3.1 8B |
| Inference | Groq API |
| Backend | Python + Flask |
| Async | Python threading + ThreadPoolExecutor |
| UI | Vanilla JS + HTML/CSS |
| Data Processing | Pandas |
| Deployment | Render |
| Containerization | Docker |

---

## Project Structure

```text
ahoum-eval/
│
├── app.py                      # Flask app + async job endpoints
├── scorer.py                   # Concurrent batched scoring engine
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
│   └── index.html              # Dashboard UI with conversation builder
│
├── Facets_Assignment.csv       # Original dataset
├── Facets_Cleaned.csv          # Processed dataset
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Procfile                    # For Render/Heroku deployment
├── render.yaml                 # Render configuration
├── .env.example
└── PROMPT_LOG.md
```

---

## Setup (Local)

### 1. Clone Repository

```bash
git clone https://github.com/gellajayaramakrishna/ahoum-eval.git
cd ahoum-eval
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Add your Groq API key inside `.env`:

```env
GROQ_API_KEY=your_api_key_here
```

### 4. Launch Flask UI

```bash
python app.py
```

Open: `http://localhost:5000`

---

## Docker Setup

```bash
docker-compose up --build
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard UI |
| `/score` | POST | Start a scoring job, returns `job_id` |
| `/status/<job_id>` | GET | Poll job progress and results |

---

## Confidence Scoring

Each facet score includes:

- `score` → integer from 1–5
- `confidence` → float from 0.0–1.0

Example:

```json
{
  "empathy": {
    "score": 4,
    "confidence": 0.91
  }
}
```

---

## Example Evaluation Categories

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

## Production Features

- Async job system — no request timeouts on hosted servers
- Concurrent batch processing
- Exponential backoff on rate limits
- JSON parse error recovery with retry
- Canonical facet name normalization
- Hard score/confidence clamping
- Deterministic scoring (`temperature=0`)
- Configurable batch sizes and worker counts
- Docker support
- Modular architecture

---

## Limitations

LLM-based evaluation still contains slight non-determinism (~2–3% variance) even at `temperature=0`. For production environments, scores should be cached or averaged across multiple runs.

---

## Prompt Logging

All prompts and responses used during evaluation are documented in `PROMPT_LOG.md`.

---

## Model Information

| Field | Value |
|---|---|
| Model | llama-3.1-8b-instant |
| Provider | Groq |
| License | Meta Open Weights |
| Parameter Size | 8B |

---

## Author

**Gella Jaya Rama Krishna**  
AI/ML Internship Assignment  
IIT Gandhinagar
