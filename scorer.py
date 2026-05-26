import os
import json
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def score_conversation(conversation_text, facets, batch_size=30):
    all_scores = {}
    total_batches = (len(facets) + batch_size - 1) // batch_size

    for i in range(0, len(facets), batch_size):
        batch = facets[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"  Scoring batch {batch_num}/{total_batches}...")

        scores = score_batch_with_retry(conversation_text, batch)
        all_scores.update(scores)

        time.sleep(12)

    return all_scores


def score_batch_with_retry(conversation_text, facets_batch, max_retries=3):
    for attempt in range(max_retries):
        try:
            return score_batch(conversation_text, facets_batch)
        except Exception as e:
            if "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"    Rate limited. Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"    Error: {e}")
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
        max_tokens=2000,
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

    result = {}
    for facet, val in parsed.items():
        if isinstance(val, dict):
            result[facet] = {
                "score": int(val.get("score", 3)),
                "confidence": float(val.get("confidence", 0.5))
            }
        elif isinstance(val, (int, float)):
            result[facet] = {"score": int(val), "confidence": 0.5}
        else:
            result[facet] = {"score": 3, "confidence": 0.0}

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