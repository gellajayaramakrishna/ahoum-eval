# Ahoum — Conversation Evaluation Benchmark

A production-ready benchmark system that scores every conversation turn across **399 distinct facets** covering linguistic quality, pragmatics, safety, and emotion — using open-weights LLMs via Groq API.

## Overview

This system takes any conversation as input and scores it on 399 psychological, linguistic, and behavioral facets using **Llama 3.1 8B** (open-weights, ≤16B). Each facet receives a score (1–5) and a confidence value (0.0–1.0).

The architecture is designed to scale to **5000+ facets without any redesign** — by processing facets in configurable batches.

## Architecture

Input Conversation
↓
facets.py — loads 399 facets from CSV
↓
scorer.py — splits into batches of 30
↓
Groq API (Llama 3.1 8B) — scores each batch
↓
JSON output — score + confidence per facet
↓
Flask UI — visual results dashboard

## Why This Architecture Scales to 5000 Facets

Facets are processed in **configurable batches** (default: 30 per API call).
To scale from 399 → 5000 facets, nothing changes architecturally — the system simply runs more batches automatically.

```python
# This handles 399 facets or 5000 facets identically
score_conversation(conversation, facets, batch_size=30)
```

## Hard Constraints Met

| Constraint | How We Meet It |
|---|---|
| No one-shot prompting | Batches of 30 facets per API call |
| Open-weights ≤16B | Llama 3.1 8B via Groq |
| Scales to 5000 facets | Batch architecture, no redesign needed |
| 5-point score scale | 1=Not present → 5=Strongly present |

## Brownie Points

| Feature | Status |
|---|---|
| Confidence outputs | ✅ Each score has 0.0–1.0 confidence |
| Sample UI | ✅ Flask dashboard at localhost:5000 |
| Dockerised baseline | ✅ See Docker section below |

## Project Structure

ahoum-eval/
├── facets.py                 # Load and clean facets from CSV
├── scorer.py                 # Core scoring engine (batched)
├── score_all.py              # Score all 50 conversations
├── generate_conversations.py # Generate 50 sample conversations
├── preprocess.py             # Clean CSV + add extra columns
├── app.py                    # Flask web UI
├── templates/
│   └── index.html            # UI frontend
├── conversations/
│   └── conversations.json    # 50 generated conversations
├── results/
│   └── all_scores.json       # Scores for all conversations
├── Facets_Assignment.csv     # Original facets dataset
├── Facets_Cleaned.csv        # Preprocessed facets with categories
├── requirements.txt
└── .env.example

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/gellajayaramakrishna/ahoum-eval.git
cd ahoum-eval
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env` and add your Groq API key:

## Limitations
- LLM scoring has slight non-determinism (~2-3% variance) even at temperature=0 
  due to floating point differences. For production, scores should be averaged 
  over 3 runs or cached after first evaluation.