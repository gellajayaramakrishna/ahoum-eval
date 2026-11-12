"""
LLM judge: Llama 3.1 8B (open weights) via Groq, batched + concurrent.

v3 changes (each one fixes a failure measured in reports/teacher_audit.json):
  * facet names are cleaned before prompting (no trailing ':' headings, which the model merged
    with neighbours) and every returned key goes through FacetMatcher (8.5% of keys were mangled)
  * JSON mode (response_format=json_object), with automatic fallback if the endpoint rejects it
  * facets the model silently drops get ONE targeted re-ask before falling back to neutral
  * fallbacks are explicit: {"score": 3, "confidence": 0.0, "source": "fallback"}
  * Groq client is created lazily, so the app (and the CPU-only judge) runs without an API key
  * in-memory LRU cache keyed by (prompt version, model, conversation, facet batch)
  * per-call stats (calls, retries, fallbacks, tokens) are returned for monitoring
"""
import hashlib
import json
import os
import random
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from facets import FacetMatcher

load_dotenv()

MODEL = os.getenv("AHOUM_LLM_MODEL", "llama-3.1-8b-instant")
PROMPT_VERSION = "v3.0"

SYSTEM_PROMPT = "You are a careful conversation analyst. You return only valid JSON."

USER_TEMPLATE = """Score the conversation below on each facet.

CONVERSATION:
{conversation}

FACETS (score every one, use the names exactly as written):
{facets}

SCALE
1 = not present / no evidence in the conversation
2 = weak or indirect evidence
3 = clearly present but moderate
4 = strong, repeated evidence
5 = dominant feature of the conversation
If a facet cannot be observed from a conversation at all (e.g. a lab value or a yearly count),
give score 1 with low confidence.

confidence (0.0-1.0) = how certain you are, given the evidence in the text.

Return ONE JSON object mapping each facet name to {{"score": int, "confidence": float}}, e.g.
{{"Empathy": {{"score": 4, "confidence": 0.8}}, "Basophil count": {{"score": 1, "confidence": 0.3}}}}"""

_client = None
_client_lock = threading.Lock()
_json_mode = True
_cache = OrderedDict()
_cache_lock = threading.Lock()
CACHE_SIZE = 512


def llm_available():
    return bool(os.getenv("GROQ_API_KEY"))


def get_client():
    global _client
    with _client_lock:
        if _client is None:
            if not llm_available():
                raise RuntimeError("GROQ_API_KEY is not set - LLM judge unavailable (use mode='lite').")
            from groq import Groq
            _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        return _client


def set_client(client):
    """Inject a client (tests use a fake one)."""
    global _client
    _client = client


def _fallback(facets):
    return {f: {"score": 3, "confidence": 0.0, "source": "fallback"} for f in facets}


def _cache_key(conversation, facets):
    h = hashlib.sha256()
    h.update(f"{PROMPT_VERSION}|{MODEL}|".encode())
    h.update(conversation.encode())
    h.update("|".join(facets).encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------- public API
def score_conversation(conversation_text, facets, batch_size=25, max_workers=2, progress_cb=None,
                       return_stats=False):
    """Scores `facets` for one conversation with the LLM judge.
    progress_cb(done_batches, total_batches, batch_scores) is called as batches finish."""
    batches = [facets[i:i + batch_size] for i in range(0, len(facets), batch_size)]
    total, done = len(batches), 0
    all_scores = {}
    stats = {"llm_calls": 0, "retries": 0, "reasked": 0, "fallbacks": 0, "cache_hits": 0,
             "prompt_tokens": 0, "completion_tokens": 0}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(score_batch_with_retry, conversation_text, b, stats, lock): b for b in batches}
        for fut in as_completed(futs):
            scores = fut.result()
            all_scores.update(scores)
            done += 1
            if progress_cb:
                progress_cb(done, total, scores)
    stats["fallbacks"] = sum(1 for v in all_scores.values() if v.get("source") == "fallback")
    return (all_scores, stats) if return_stats else all_scores


def score_batch_with_retry(conversation_text, facets_batch, stats=None, lock=None, max_retries=4):
    stats = stats if stats is not None else {}
    lock = lock or threading.Lock()

    def bump(k, n=1):
        with lock:
            stats[k] = stats.get(k, 0) + n

    key = _cache_key(conversation_text, facets_batch)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            bump("cache_hits")
            return json.loads(_cache[key])

    result = None
    for attempt in range(max_retries):
        try:
            result = score_batch(conversation_text, facets_batch, bump)
            break
        except Exception as e:  # noqa: BLE001 - network/parse errors are all retried the same way
            msg = str(e).lower()
            bump("retries")
            if "429" in msg or "rate" in msg:
                time.sleep(min(60, (2 ** attempt) * 2) + random.uniform(0, 2))
            elif attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                print(f"    LLM batch failed after {max_retries} attempts: {e}")
    if result is None:
        return _fallback(facets_batch)

    missing = [f for f in facets_batch if f not in result]
    if missing:  # one targeted re-ask for facets the model skipped
        bump("reasked", len(missing))
        try:
            result.update(score_batch(conversation_text, missing, bump))
        except Exception:  # noqa: BLE001
            pass
    for f in facets_batch:
        result.setdefault(f, {"score": 3, "confidence": 0.0, "source": "fallback"})

    if all(v.get("source") != "fallback" for v in result.values()):
        with _cache_lock:
            _cache[key] = json.dumps(result)
            if len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)
    return result


def _parse_json(raw):
    raw = raw.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    s, e = raw.find("{"), raw.rfind("}") + 1
    if s != -1 and e > s:
        raw = raw[s:e]
    return json.loads(raw)


def parse_scores(parsed, facets_batch):
    """Model output dict -> {canonical_facet: {"score": int 1-5, "confidence": float 0-1, "source": "llm"}}"""
    matcher = FacetMatcher(facets_batch)
    out = {}
    for k, val in parsed.items():
        f = matcher.match(k)
        if f is None or f in out:
            continue
        if isinstance(val, dict):
            rs, rc = val.get("score", 3), val.get("confidence", 0.5)
        elif isinstance(val, (int, float)):
            rs, rc = val, 0.5
        else:
            continue
        try:
            score = max(1, min(5, int(round(float(rs)))))
        except (TypeError, ValueError):
            continue
        try:
            conf = max(0.0, min(1.0, float(rc)))
        except (TypeError, ValueError):
            conf = 0.5
        out[f] = {"score": score, "confidence": round(conf, 3), "source": "llm"}
    return out


def score_batch(conversation_text, facets_batch, bump=lambda *a, **k: None):
    global _json_mode
    prompt = USER_TEMPLATE.format(conversation=conversation_text,
                                  facets="\n".join(f"- {f}" for f in facets_batch))
    kwargs = dict(model=MODEL, temperature=0, max_tokens=60 + 28 * len(facets_batch),
                  messages=[{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}])
    client = get_client()
    bump("llm_calls")
    try:
        resp = client.chat.completions.create(**kwargs, **({"response_format": {"type": "json_object"}}
                                                           if _json_mode else {}))
    except Exception as e:  # noqa: BLE001
        if _json_mode and "response_format" in str(e):
            _json_mode = False
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    usage = getattr(resp, "usage", None)
    if usage is not None:
        bump("prompt_tokens", getattr(usage, "prompt_tokens", 0) or 0)
        bump("completion_tokens", getattr(usage, "completion_tokens", 0) or 0)
    return parse_scores(_parse_json(resp.choices[0].message.content), facets_batch)


if __name__ == "__main__":
    from facets import load_facets

    conv = """Person A: I'm really scared about the job interview tomorrow.
Person B: Don't worry, you've prepared well. Just be confident!
Person A: What if I forget everything?
Person B: Take a deep breath. You know this material better than anyone."""
    scores, st = score_conversation(conv, load_facets()[:30], batch_size=30, return_stats=True)
    print(st)
    for f, d in list(scores.items())[:5]:
        print(f, d)
