"""
Human-label check: do the judges' emotion facets agree with human annotators?

Data : GoEmotions TEST split (held out from everything; the emotion encoder trains on the train split).
Task : each comment is wrapped as a one-turn conversation and scored on 6 facets; the score is
       compared with the human label for the matching emotions (AUROC: 0.5 = chance, 1.0 = perfect).

  python scripts/validate_against_humans.py            # lite judge only (no API key needed)
  python scripts/validate_against_humans.py --llm      # also the Llama 3.1 8B judge (needs GROQ_API_KEY,
                                                       #   ~N/1 API calls, cached + resumable)
Writes reports/human_validation.json
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lite.emotion import load_goemotions  # noqa: E402

FACETS = {
    "Joyfulness": ["joy", "amusement", "excitement"],
    "Hostility": ["anger", "annoyance", "disgust"],
    "Fearfulness: Fear of physical dangers": ["fear", "nervousness"],
    "Depression: Feelings of sadness and hopelessness": ["sadness", "grief"],
    "Compassion": ["caring"],
    "Disrespect": ["disapproval", "disgust"],
}


def sample(n, seed=0):
    texts, Y, names = load_goemotions("test")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(texts))[:n]
    targets = {f: Y[idx][:, [names.index(e) for e in em]].max(1) for f, em in FACETS.items()}
    return [texts[i] for i in idx], targets


def auroc_table(scores, targets):
    out = {}
    for f, y in targets.items():
        s = np.array([sc[f] for sc in scores], float)
        out[f] = {"auroc": round(float(roc_auc_score(y, s)), 3), "positives": int(y.sum())}
    out["_macro_auroc"] = round(float(np.mean([v["auroc"] for v in out.values()])), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    texts, targets = sample(args.n)
    convs = [f"Person A: {t}" for t in texts]
    report_path = os.path.join(ROOT, "reports", "human_validation.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {}
    report.update({"dataset": "GoEmotions test split", "n_comments": len(texts),
                   "facet_to_human_labels": FACETS})

    from hybrid import get_lite_judge
    judge = get_lite_judge()
    lite = [{f: v["raw"] for f, v in judge.predict(c, list(FACETS)).items()} for c in convs]
    report["lite_judge"] = auroc_table(lite, targets)
    # in-domain reference: the GoEmotions encoder itself (trained on the train split of this dataset)
    enc = judge.emo
    P = enc.predict_proba(texts)
    ref = [{f: float(P[i, [enc.names.index(e) for e in em]].sum()) for f, em in FACETS.items()} for i in range(len(texts))]
    report["reference_emotion_encoder_in_domain"] = auroc_table(ref, targets)
    print("lite judge:", json.dumps(report["lite_judge"], indent=1))

    if args.llm:
        from scorer import score_conversation
        cache_p = os.path.join(ROOT, "data", "llm_goemotions_cache.json")
        cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
        llm = []
        for i, c in enumerate(convs):
            if c not in cache:
                s = score_conversation(c, list(FACETS), batch_size=len(FACETS), max_workers=1)
                cache[c] = {f: v["score"] + 0.01 * v["confidence"] for f, v in s.items()}
                if i % 20 == 0:
                    json.dump(cache, open(cache_p, "w"))
                    print(f"  {i}/{len(convs)}")
            llm.append(cache[c])
        json.dump(cache, open(cache_p, "w"))
        report["llm_judge"] = auroc_table(llm, targets)
        print("llm judge:", json.dumps(report["llm_judge"], indent=1))

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
