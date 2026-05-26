# Prompt Log — Ahoum AI/ML Assignment
**Student:** Gella Jaya Rama Krishna  
**Role:** AI/ML Internship  
**Institute:** IIT Gandhinagar  
**Date:** May 26, 2026

## Overview
I used Claude (Anthropic) primarily for implementation assistance, syntax 
support, and boilerplate acceleration. All major architectural decisions, 
debugging, system design, scalability planning, and integration logic were 
designed and implemented independently.

AI assistance was used similarly to an engineering copilot for accelerating 
implementation speed, while all core system-level decisions and architectural 
tradeoffs were made independently.

---

## 1. Project Architecture
**What I designed and implemented:** The full evaluation pipeline — facet 
loading, batched scoring, resume logic, confidence outputs, and folder 
structure. Implemented hierarchical facet grouping and batched routing so 
the system scales from 300 to 5000+ facets without architectural redesign.  
**AI assistance:** Referenced Claude for initial folder structure suggestions, 
which I restructured to fit the assignment requirements.

---

## 2. Data Preprocessing
**What I designed and implemented:** Analyzed the Facets CSV independently, 
decided which columns to add (category, weight, rubric, batch_group), and 
wrote the categorization and weighting logic.  
**AI assistance:** Used Claude for syntax acceleration on pandas column 
operations.

---

## 3. Scoring Engine
**What I designed and implemented:** Chose Llama 3.1 8B via Groq API to 
satisfy the open-weights ≤16B hard constraint. Designed batched scoring 
(30 facets/batch) for scalability. Debugged and resolved real rate limiting 
issues, JSON parse failures, and built resume logic that saves progress 
after every conversation so no work is ever lost on failure.  
**AI assistance:** Generated initial boilerplate using Claude for the 
Groq API call structure.

---

## 4. Conversation Generation
**What I designed and implemented:** Defined 50 diverse evaluation scenarios 
covering edge cases — toxic content, emotional support, medical advice, 
whistleblower, ambiguous intent, mental health, misinformation and more.  
**AI assistance:** I defined the scenarios and evaluation intent; Claude 
was used to accelerate generation of dialogue variations for those scenarios.

---

## 5. Flask UI
**What I designed and implemented:** Designed the evaluation flow, 
two-column layout, and user interaction model.  
**AI assistance:** Used AI assistance for Flask routing boilerplate 
and CSS styling acceleration.

---

## 6. Docker Setup
**What I designed and implemented:** Designed the container structure, 
environment configuration, and single-command startup flow. Implemented 
a fallback/mock evaluation mode to ensure reproducible execution even 
on systems without local GPU support.  
**AI assistance:** Referenced Claude for Dockerfile and docker-compose syntax.

---

## Key Architectural Decisions Made Independently
- Llama 3.1 8B via Groq — open-weights, Meta license, satisfies ≤16B constraint
- Batch size 30 — balances throughput vs Groq rate limits
- Sleep 12s between batches — proactively prevents rate limiting
- Resume logic — saves after every conversation, never loses progress
- Safety facets weighted 2x — reflects real-world evaluation priorities
- Score scale 1-5 with confidence 0.0-1.0 per facet
- temperature=0 for deterministic, reproducible scoring
- Hierarchical batch grouping — scales to 5000+ facets without redesign

## Notes
- Model: llama-3.1-8b-instant (Meta open-weights license)
- No one-shot prompting anywhere — full batched architecture throughout
- Confidence outputs included for every facet score
