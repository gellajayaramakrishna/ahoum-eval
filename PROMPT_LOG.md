# Prompt Log — Ahoum AI/ML Assignment
**Student:** Gella Jaya Rama Krishna  
**Role:** AI/ML Internship  
**Institute:** IIT Gandhinagar  
**Date:** May 26, 2026

## Overview
This log documents all AI-assisted prompts used during the development 
of the conversation evaluation system.

## Prompt 1 — Project Structure & Preprocessing
**Tool:** Claude (Anthropic)  
**Prompt:** "Create a project structure for a conversation evaluation 
system that scores 300 facets. Include data cleaning, adding category 
and description columns to the facets CSV."  
**Output:** preprocess.py, facets.py, folder structure

## Prompt 2 — Scoring Engine
**Tool:** Claude (Anthropic)  
**Prompt:** "Build a batched LLM scorer using Groq API with Llama 3.1 8B. 
Score 30 facets per batch, include retry logic on 429 rate limits, 
output score (1-5) and confidence (0.0-1.0) per facet."  
**Output:** scorer.py with batch logic, retry, confidence outputs

## Prompt 3 — Generate 50 Conversations
**Tool:** Claude (Anthropic)  
**Prompt:** "Generate 50 diverse conversations covering edge cases: 
safe conversations, toxic content, emotional support, ambiguous intent, 
whistleblower scenarios, medical advice, mental health, etc."  
**Output:** generate_conversations.py → conversations/conversations.json

## Prompt 4 — Score All Conversations
**Tool:** Claude (Anthropic)  
**Prompt:** "Write a script to score all 50 conversations using scorer.py, 
with resume logic so progress is saved after every conversation and 
never lost on crash or rate limit."  
**Output:** score_all.py → results/all_scores.json

## Prompt 5 — Flask UI
**Tool:** Claude (Anthropic)  
**Prompt:** "Build a Flask web app where a user can paste a conversation 
and see facet scores in a two-column layout. No emojis, clean design."  
**Output:** app.py + templates/index.html

## Prompt 6 — Docker Setup
**Tool:** Claude (Anthropic)  
**Prompt:** "Write a Dockerfile and docker-compose.yml for this Flask 
+ Python project so it runs with one command."  
**Output:** Dockerfile, docker-compose.yml

## Prompt 7 — README Documentation
**Tool:** Claude (Anthropic)  
**Prompt:** "Write a full README with setup steps, architecture explanation, 
limitations, and how to run the project locally and via Docker."  
**Output:** README.md

## Notes
- All architectural decisions made by the student
- Claude used for boilerplate generation and debugging
- Model used for scoring: llama-3.1-8b-instant via Groq API
- No one-shot prompting — all scoring in batches of 30 facets
- Confidence outputs included for every facet score
