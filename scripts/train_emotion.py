"""
Train + evaluate the human-label emotion encoder on GoEmotions (official train/dev/test splits).
Writes models/emotion_encoder.joblib and reports/emotion_metrics.json
Run:  python scripts/train_emotion.py
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lite.emotion import EMO_PATH, EmotionEncoder, load_goemotions  # noqa: E402


def main():
    t0 = time.time()
    tr_x, tr_y, names = load_goemotions("train")
    dv_x, dv_y, _ = load_goemotions("dev")
    te_x, te_y, _ = load_goemotions("test")
    print(f"train {len(tr_x)}  dev {len(dv_x)}  test {len(te_x)}  labels {len(names)}")

    enc = EmotionEncoder().fit(tr_x, tr_y, names, dv_x, dv_y)
    P = enc.predict_proba(te_x)
    pred = (P >= enc.thresholds).astype(int)
    per = {n: {"f1": round(float(f1_score(te_y[:, j], pred[:, j], zero_division=0)), 3),
               "auroc": round(float(roc_auc_score(te_y[:, j], P[:, j])), 3),
               "support": int(te_y[:, j].sum())} for j, n in enumerate(names)}
    report = {
        "dataset": "GoEmotions (Demszky et al., 2020) official splits",
        "n_train": len(tr_x), "n_test": len(te_x), "labels": len(names),
        "test_macro_f1": round(float(f1_score(te_y, pred, average="macro", zero_division=0)), 4),
        "test_micro_f1": round(float(f1_score(te_y, pred, average="micro", zero_division=0)), 4),
        "test_macro_auroc": round(float(np.mean([v["auroc"] for v in per.values()])), 4),
        "reference_bert_base_macro_f1_from_paper": 0.46,
        "per_label": per,
        "train_seconds": round(time.time() - t0, 1),
    }
    enc.save(EMO_PATH)
    report["model_MB"] = round(os.path.getsize(EMO_PATH) / 1e6, 2)
    with open(os.path.join(ROOT, "reports", "emotion_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "per_label"}, indent=2))


if __name__ == "__main__":
    main()
