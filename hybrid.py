"""
Hybrid scoring: the distilled CPU judge scores every facet in ~40 ms, then only the facets it is
least sure about are sent to the LLM judge.

Modes
  lite    - LiteJudge only (no API key needed, deterministic)
  hybrid  - LiteJudge + LLM on the `llm_share` most uncertain facets (default 30%)
  llm     - LLM judge on every facet (original behaviour)

Offline evaluation (reports/lite_metrics.json, routing_A): with llm_share=0.5 the facets kept by the
student have MAE 0.27 vs the teacher, against 0.47 for a random half.
"""
import os
import time

from scorer import llm_available, score_conversation

_judge = None


def get_lite_judge():
    global _judge
    if _judge is None:
        from lite.model import LiteJudge
        _judge = LiteJudge.load()
    return _judge


def score(conversation, facets, mode="hybrid", llm_share=0.3, progress_cb=None, batch_size=25, max_workers=2):
    t0 = time.perf_counter()
    info = {"mode": mode, "llm_share": 0.0, "notes": []}
    if mode in ("hybrid", "llm") and not llm_available():
        info["notes"].append("GROQ_API_KEY not set - fell back to lite mode")
        mode = info["mode"] = "lite"

    if mode == "llm":
        scores, st = score_conversation(conversation, facets, batch_size, max_workers, progress_cb, return_stats=True)
        info.update(llm_stats=st, llm_share=1.0)
    else:
        lite = get_lite_judge().predict(conversation, facets)
        info["lite_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        scores = lite
        if mode == "hybrid" and llm_share > 0:
            n = max(1, int(round(len(facets) * llm_share)))
            routed = sorted(facets, key=lambda f: -lite[f]["uncertainty"])[:n]
            if progress_cb:
                progress_cb(0, -(-n // batch_size), {})
            llm, st = score_conversation(conversation, routed, batch_size, max_workers, progress_cb, return_stats=True)
            for f, v in llm.items():
                if v.get("source") == "fallback":     # keep the student's answer, not a neutral 3
                    continue
                v["lite_score"] = lite[f]["score"]
                scores[f] = v
            info.update(llm_stats=st, llm_share=round(n / len(facets), 3), routed=len(routed))
        elif progress_cb:
            progress_cb(1, 1, scores)
    info["total_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    return scores, info


def default_mode():
    return os.getenv("AHOUM_DEFAULT_MODE", "hybrid" if llm_available() else "lite")
