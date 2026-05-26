# Prompt Log — Ahoum AI/ML Assignment
**Student:** Gella Jaya Rama Krishna  
**Role:** AI/ML Internship  
**Institute:** IIT Gandhinagar  
**Date:** May 26, 2026

## Overview
I used Claude (Anthropic) occasionally during development — primarily for 
syntax help and boilerplate. All core design decisions, debugging, and 
integration were done by me independently.

---

## 1. Project Architecture
**What I did:** Designed the full pipeline — facet loading, batched scoring, 
resume logic, confidence outputs, and folder structure.  
**AI assistance:** Asked Claude to suggest a folder structure. I modified it 
to fit my requirements.

---

## 2. Data Preprocessing
**What I did:** Analyzed the Facets CSV, decided which columns to add 
(category, weight, rubric, batch_group), and wrote the categorization logic.  
**AI assistance:** Claude helped with pandas syntax for column operations.

---

## 3. Scoring Engine
**What I did:** Chose Llama 3.1 8B via Groq API (satisfies open-weights ≤16B 
constraint). Designed batched scoring (30 facets/batch) for scalability to 
5000+ facets. Built retry logic, JSON error handling, and resume logic myself 
after hitting real rate limit issues during development.  
**AI assistance:** Claude helped with initial Groq API call structure.

---

## 4. Conversation Generation
**What I did:** Defined 50 diverse scenarios covering edge cases — toxic 
content, emotional support, medical advice, whistleblower, ambiguous intent, 
mental health, misinformation etc.  
**AI assistance:** Claude helped generate dialogue text for the scenarios 
I defined.

---

## 5. Flask UI
**What I did:** Designed the evaluation flow and two-column layout concept.  
**AI assistance:** Claude helped with Flask routing boilerplate and CSS styling.

---

## 6. Docker Setup
**What I did:** Decided on container structure and environment configuration.  
**AI assistance:** Claude helped with Dockerfile and docker-compose syntax.

---

## Key Decisions Made Independently
- Llama 3.1 8B via Groq — open-weights, ≤16B, satisfies hard constraint
- Batch size 30 — balances speed vs Groq rate limits
- Sleep 12s between batches — prevents rate limiting proactively
- Resume logic — saves after every conversation, never loses progress
- Safety facets weighted 2x — higher importance in overall score
- Score scale 1-5 with confidence 0.0-1.0 per facet
- temperature=0 for deterministic, reproducible scoring

## Notes
- Model: llama-3.1-8b-instant (Meta open-weights license)
- No one-shot prompting anywhere — full batched architecture
- Confidence outputs included for every facet score
