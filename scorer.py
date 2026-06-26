import os
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def score_conversation(conversation_text, facets, batch_size=25, max_workers=2, progress_cb=None):
    """
    Scores all facets for one conversation.
    Batches are sent concurrently (bounded by max_workers) instead of one-at-a-time
    with a fixed sleep. We only back off when Groq actually signals a rate limit (429),
    rather than always waiting a fixed amount of time.

    progress_cb: optional callable(done_batches, total_batches) for live progress reporting.
    """
    batches = [facets[i:i + batch_size] for i in range(0, len(facets), batch_size)]
    total_batches = len(batches)
    all_scores = {}
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(score_batch_with_retry, conversation_text, batch): batch
            for batch in batches
        }
        for future in as_completed(future_to_batch):
            scores = future.result()
            all_scores.update(scores)
            done += 1
            if progress_cb:
                progress_cb(done, total_batches, scores)   # <-- now also passes this batch's scores
            else:
                print(f"  Batch {done}/{total_batches} done")

    return all_scores


def score_batch_with_retry(conversation_text, facets_batch, max_retries=4):
    """
    Retries on rate-limit errors AND on malformed/incomplete JSON responses
    (small models sometimes truncate or drop facets), since both are
    transient and worth retrying. Gives up only after max_retries.
    """
    for attempt in range(max_retries):
        try:
            result = score_batch(conversation_text, facets_batch)
            # Fill in any facet the model silently skipped in its response
            for f in facets_batch:
                if f not in result:
                    result[f] = {"score": 3, "confidence": 0.0}
            return result
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
            is_json_error = "Expecting" in str(e) or "delimiter" in str(e) or "json" in str(e).lower()
            if is_rate_limit:
                wait = min(60, (2 ** attempt) * 2) + random.uniform(0, 2)
                print(f"    Rate limited. Backing off {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            elif is_json_error and attempt < max_retries - 1:
                print(f"    Malformed JSON response, retrying (attempt {attempt + 1}/{max_retries})...")
                time.sleep(1)
            else:
                print(f"    Error scoring batch: {e}")
                break
    return {f: {"score": 3, "confidence": 0.0} for f in facets_batch}


def score_batch(conversation_text, facets_batch):
    facets_list = "\n".join([f"- {f}" for f in facets_batch])

    prompt = f"""You are an expert conversation analyst. Score this conversation on each facet.

CONVERSATION:
{conversation_text}

FACETS TO SCORE:
{facets_list}

SCORING RULES:
- Score 1-5: 1=Not present, 2=Low, 3=Medium, 4=High, 5=Very high
- Confidence 0.0-1.0: how sure you are

CRITICAL: Return ONLY a valid JSON object. No markdown. No explanation. No backticks.
Format:
{{"FacetName": {{"score": 3, "confidence": 0.8}}, "FacetName2": {{"score": 1, "confidence": 0.5}}}}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a conversation analyst. Return only valid JSON with no extra text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=3000,
    )

    raw = response.choices[0].message.content.strip()

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    parsed = json.loads(raw)

    # Build a lookup so we can match the model's returned key back to the
    # EXACT canonical facet name, even if it mangled casing/whitespace/punctuation.
    canonical_lookup = {f.strip().lower(): f for f in facets_batch}

    result = {}
    for facet_raw, val in parsed.items():
        # Normalize the model's key and try to match it to a real facet
        normalized = str(facet_raw).strip().lower()
        canonical_facet = canonical_lookup.get(normalized)

        if canonical_facet is None:
            # The model invented a facet name that isn't in our list at all — discard it.
            print(f"    Discarding unrecognized facet key from model output: '{facet_raw}'")
            continue

        # Extract score + confidence defensively, then CLAMP to valid ranges
        if isinstance(val, dict):
            raw_score = val.get("score", 3)
            raw_conf = val.get("confidence", 0.5)
        elif isinstance(val, (int, float)):
            raw_score = val
            raw_conf = 0.5
        else:
            raw_score = 3
            raw_conf = 0.0

        try:
            score = int(round(float(raw_score)))
        except (TypeError, ValueError):
            score = 3
        score = max(1, min(5, score))   # hard clamp to 1-5, no matter what the model said

        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))   # hard clamp to 0.0-1.0

        result[canonical_facet] = {"score": score, "confidence": confidence}

    return result


if __name__ == "__main__":
    from facets import load_facets

    test_conversation = """
    Person A: I'm really scared about the job interview tomorrow.
    Person B: Don't worry, you've prepared well. Just be confident!
    Person A: What if I forget everything?
    Person B: Take a deep breath. You know this material better than anyone.
    """

    print("Loading facets...")
    facets = load_facets()
    print(f"Testing with first 30 facets...")
    scores = score_conversation(test_conversation, facets[:30], batch_size=30)
    print(f"Scored {len(scores)} facets!")
    for facet, data in list(scores.items())[:5]:
        print(f"  {facet}: score={data['score']}, confidence={data['confidence']}")