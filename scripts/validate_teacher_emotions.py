"""
Construct-validity check of the LLM judge on emotion facets.

For facets that have a clear GoEmotions counterpart, compare across the 50 conversations:
   LLM judge score (Llama 3.1 8B)   vs   human-label emotion encoder (GoEmotions, independent model)
If the judge tracks real emotional content, the two should rank conversations similarly.

Writes reports/teacher_emotion_validity.json
Run:  python scripts/validate_teacher_emotions.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lite.emotion import EmotionEncoder  # noqa: E402
from lite.text import split_turns  # noqa: E402

FACET_TO_EMOTIONS = {
    "Happiness": ["joy", "amusement", "excitement"],
    "Joyfulness": ["joy", "amusement", "excitement"],
    "Merriness": ["joy", "amusement"],
    "Enthusiasm": ["excitement", "joy"],
    "Hostility": ["anger", "annoyance", "disgust"],
    "Irritability": ["annoyance", "anger"],
    "Fearfulness: Fear of physical dangers": ["fear", "nervousness"],
    "Depression: Feelings of sadness and hopelessness": ["sadness", "grief", "disappointment"],
    "Moroseness": ["sadness", "disappointment", "grief"],
    "Desperation": ["fear", "sadness", "nervousness"],
    "Compassion": ["caring", "love"],
    "Warmheartedness": ["caring", "love", "gratitude"],
    "Discontentment": ["disappointment", "disapproval", "annoyance"],
    "Disrespect": ["disapproval", "disgust", "anger"],
    "Psychological construct: Hope scale": ["optimism"],
}


def main():
    enc = EmotionEncoder.load()
    convs = json.load(open(os.path.join(ROOT, "conversations", "conversations.json")))
    labels = pd.read_csv(os.path.join(ROOT, "data", "teacher_labels.csv"))
    labels = labels[labels.confidence > 0]
    names = enc.names

    prof = {}
    for c in convs:
        turns = [u for _, u in split_turns(c["conversation"])] or [c["conversation"]]
        prof[c["id"]] = enc.predict_proba(turns).mean(0)

    rows, null = [], []
    rng = np.random.default_rng(0)
    for facet, emos in FACET_TO_EMOTIONS.items():
        g = labels[labels.facet == facet]
        if len(g) < 20 or g.score.nunique() < 2:
            continue
        cols = [names.index(e) for e in emos]
        ref = np.array([prof[c][cols].sum() for c in g.conv_id])
        rho = spearmanr(g.score, ref)[0]
        # permutation null: same facet, shuffled conversations
        perm = [spearmanr(rng.permutation(g.score.values), ref)[0] for _ in range(500)]
        p = float(np.mean(np.abs(perm) >= abs(rho)))
        rows.append({"facet": facet, "emotions": emos, "n_conversations": int(len(g)),
                     "spearman": round(float(rho), 3), "perm_p": round(p, 3)})
    df = pd.DataFrame(rows)
    report = {
        "what": "Spearman across conversations between LLM-judge facet score and GoEmotions-encoder probability",
        "facets_tested": int(len(df)),
        "mean_spearman": round(float(df.spearman.mean()), 3),
        "median_spearman": round(float(df.spearman.median()), 3),
        "facets_significant_p_lt_0.05": int((df.perm_p < 0.05).sum()),
        "per_facet": rows,
    }
    with open(os.path.join(ROOT, "reports", "teacher_emotion_validity.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(df.to_string(index=False))
    print({k: v for k, v in report.items() if k != "per_facet"})


if __name__ == "__main__":
    main()
