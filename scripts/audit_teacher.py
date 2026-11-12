"""
Audit of the original LLM judge (Llama 3.1 8B via Groq) outputs in results/all_scores.json.

Answers:
  1. How often did the judge return a key that isn't one of our facets (mangled / merged / hallucinated)?
  2. How often did it silently fail (facet missing, or neutral fallback with confidence 0.0)?
  3. How many facets carry no signal (near-constant across 50 very different conversations)?

Also writes data/teacher_labels.csv: the cleaned (conversation, facet, score, confidence) table
used to train the distilled judge.

Run:  python scripts/audit_teacher.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from facets import FacetMatcher, canonicalize_scores, load_facets  # noqa: E402


def main():
    facets = load_facets()
    matcher = FacetMatcher(facets)
    results = json.load(open(os.path.join(ROOT, "results", "all_scores.json")))
    n_conv, n_fac = len(results), len(facets)
    expected = n_conv * n_fac

    raw_keys = sum(len(r["scores"]) for r in results)
    tot = Counter()
    rows = []
    for r in results:
        clean, st = canonicalize_scores(r["scores"], matcher)
        tot.update(st)
        for f, v in clean.items():
            rows.append({"conv_id": r["id"], "facet": f, "score": int(v["score"]),
                         "confidence": float(v["confidence"])})
    df = pd.DataFrame(rows)
    covered = len(df)
    fallback = int((df.confidence == 0.0).sum())
    valid = df[df.confidence > 0].copy()

    g = valid.groupby("facet")["score"]
    stds = g.std(ddof=0)[g.count() >= 10]
    means = g.mean()[g.count() >= 10]

    report = {
        "conversations": n_conv,
        "facets": n_fac,
        "expected_labels": expected,
        "raw_keys_returned": raw_keys,
        "keys_exact_match": tot["exact"],
        "keys_repaired_by_canonicalizer": tot["repaired"],
        "keys_unmatched_hallucinated": tot["unmatched"],
        "key_mismatch_rate_of_returned": round((tot["repaired"] + tot["unmatched"]) / raw_keys, 4),
        "labels_covered_after_repair": covered,
        "missing_labels": expected - covered,
        "missing_rate": round((expected - covered) / expected, 4),
        "fallback_labels_conf0": fallback,
        "fallback_rate": round(fallback / expected, 4),
        "valid_labels": len(valid),
        "valid_rate": round(len(valid) / expected, 4),
        "score_distribution_valid": {int(k): int(v) for k, v in valid.score.value_counts().sort_index().items()},
        "confidence_distinct_values": int(valid.confidence.round(2).nunique()),
        "facets_with_>=10_valid_labels": int(len(stds)),
        "near_constant_facets_std_lt_0.3": int((stds < 0.3).sum()),
        "always_score_1_facets_mean_le_1.1": int((means <= 1.1).sum()),
        "examples_always_1": sorted(means[means <= 1.1].index.tolist())[:12],
        "median_facet_std": round(float(np.median(stds)), 3),
    }

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "teacher_audit.json"), "w") as f:
        json.dump(report, f, indent=2)
    df.to_csv(os.path.join(ROOT, "data", "teacher_labels.csv"), index=False)
    print(json.dumps(report, indent=2))
    print(f"\nwrote data/teacher_labels.csv ({len(df)} rows, {len(valid)} valid)")


if __name__ == "__main__":
    main()
