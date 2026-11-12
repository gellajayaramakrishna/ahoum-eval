"""
Train + evaluate the distilled judge (LiteJudge) against the Llama-3.1-8B teacher labels.

Evaluation protocols (all cross-validated, nothing is scored on data it was trained on):
  A. unseen conversations          GroupKFold(5) over conversations, facets seen in training
  B. unseen conversations + facets 5x5 grid split: train on (train convs x train facets),
                                   test on (held-out convs x held-out facets)  -> zero-shot facets
Baselines: global mean, per-facet teacher mean (A) / kNN facet prior (B).

Also measures: uncertainty-based routing (how much of the work can skip the LLM),
CPU latency for all 399 facets.

Writes reports/lite_metrics.json, reports/oof_predictions_A.csv, models/lite_judge.joblib
Run:  python scripts/train_lite.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from facets import load_facets  # noqa: E402
from lite.features import FeatureBuilder, knn_prior  # noqa: E402
from lite.model import MODEL_PATH, LiteJudge  # noqa: E402

N_FOLDS = 5
QUICK = "--quick" in sys.argv


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    pr = np.clip(np.rint(p), 1, 5).astype(int)
    return {
        "n": int(len(y)),
        "MAE": round(float(np.mean(np.abs(y - p))), 4),
        "RMSE": round(float(np.sqrt(np.mean((y - p) ** 2))), 4),
        "pearson": round(float(pearsonr(y, p)[0]), 4) if np.std(p) > 0 else 0.0,
        "spearman": round(float(spearmanr(y, p)[0]), 4) if np.std(p) > 0 else 0.0,
        "QWK": round(float(cohen_kappa_score(y.astype(int), pr, weights="quadratic")), 4),
        "exact_acc": round(float(np.mean(pr == y)), 4),
        "within1_acc": round(float(np.mean(np.abs(pr - y) <= 1)), 4),
    }


def within_facet_spearman(df, col):
    """Mean Spearman across facets of (prediction vs teacher) over conversations.
    Measures whether the model *ranks conversations* correctly for a given facet,
    which a constant per-facet prior cannot do (it scores 0 by construction)."""
    vals = []
    for _, g in df.groupby("facet"):
        if len(g) >= 8 and g.score.nunique() > 1 and g[col].nunique() > 1:
            vals.append(spearmanr(g.score, g[col])[0])
    return round(float(np.nanmean(vals)), 4), len(vals)


def within_conv_spearman(df, col):
    vals = [spearmanr(g.score, g[col])[0] for _, g in df.groupby("conv_id") if g[col].nunique() > 1]
    return round(float(np.nanmean(vals)), 4)


def main():
    t_all = time.time()
    convs = json.load(open(os.path.join(ROOT, "conversations", "conversations.json")))
    texts = {c["id"]: c["conversation"] for c in convs}
    labels = pd.read_csv(os.path.join(ROOT, "data", "teacher_labels.csv"))
    labels = labels[labels.confidence > 0].reset_index(drop=True)      # drop neutral fallbacks
    print(f"{len(labels)} valid teacher labels, {labels.conv_id.nunique()} conversations, "
          f"{labels.facet.nunique()} facets")
    fb = FeatureBuilder()
    report = {"data": {"valid_labels": int(len(labels)), "conversations": int(labels.conv_id.nunique()),
                       "facets": int(labels.facet.nunique())}}

    # ---------------------------------------------------------------- A: unseen conversations
    oof = []
    for k, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(labels, groups=labels.conv_id)):
        L_tr, L_te = labels.iloc[tr], labels.iloc[te]
        judge = LiteJudge(seed=k).fit(texts, L_tr, fb=fb, verbose=False)
        fmean = L_tr.groupby("facet").score.mean()
        for cid, g in L_te.groupby("conv_id"):
            mu, err = judge.predict_matrix(texts[cid], g.facet.tolist())
            oof.append(g.assign(pred=mu, unc=err,
                                base_global=L_tr.score.mean(),
                                base_facet=g.facet.map(fmean).fillna(L_tr.score.mean()).values,
                                fold=k))
        print(f"  A fold {k+1}/{N_FOLDS} done")
    A = pd.concat(oof)
    A.to_csv(os.path.join(ROOT, "reports", "oof_predictions_A.csv"), index=False)

    resA = {}
    for name, col in [("global_mean_baseline", "base_global"), ("facet_mean_baseline", "base_facet"),
                      ("lite_judge", "pred")]:
        m = metrics(A.score, A[col])
        # baselines are constant per facet -> they cannot rank conversations (metric undefined, report null)
        if col == "pred":
            m["within_facet_spearman"], nf = within_facet_spearman(A, col)
        else:
            m["within_facet_spearman"] = None
        m["within_conv_spearman"] = within_conv_spearman(A, col) if col != "base_global" else None
        resA[name] = m
    resA["facets_in_within_facet_metric"] = nf
    # facets where conversation-level signal is externally validated (GoEmotions counterpart)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from validate_teacher_emotions import FACET_TO_EMOTIONS
    emoA = A[A.facet.isin(FACET_TO_EMOTIONS)]
    resA["within_facet_spearman_emotion_facets"] = {
        "facets": int(emoA.facet.nunique()),
        "lite_judge": within_facet_spearman(emoA, "pred")[0],
    }
    hi = A.groupby("facet").score.std()
    hiA = A[A.facet.isin(hi[hi >= 1.0].index)]
    resA["within_facet_spearman_high_variance_facets"] = {
        "facets": int(hiA.facet.nunique()), "teacher_std_threshold": 1.0,
        "lite_judge": within_facet_spearman(hiA, "pred")[0],
    }
    report["A_unseen_conversations"] = resA

    # routing: keep the most-certain x% with the student, send the rest to the LLM
    A_sorted = A.sort_values("unc")
    rng = np.random.default_rng(0)
    routing = []
    for keep in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        n = int(len(A) * keep)
        kept = A_sorted.iloc[:n]
        rand = A.iloc[rng.permutation(len(A))[:n]]
        routing.append({
            "student_share": keep,
            "llm_share": round(1 - keep, 2),
            "MAE_on_student_rows": round(float(np.mean(np.abs(kept.score - kept.pred))), 4),
            "MAE_random_rows_same_size": round(float(np.mean(np.abs(rand.score - rand.pred))), 4),
            "within1_on_student_rows": round(float(np.mean(np.abs(np.rint(kept.pred) - kept.score) <= 1)), 4),
            # hybrid vs teacher: LLM rows match teacher by definition, so overall MAE = kept error * share
            "hybrid_MAE_vs_teacher": round(float(np.sum(np.abs(kept.score - kept.pred)) / len(A)), 4),
        })
    report["routing_A"] = routing
    unc_corr = spearmanr(A.unc, np.abs(A.score - A.pred))[0]
    report["uncertainty_vs_error_spearman"] = round(float(unc_corr), 4)

    if QUICK:
        print(json.dumps({k: report[k] for k in ("A_unseen_conversations", "uncertainty_vs_error_spearman")}, indent=1))
        return

    # ---------------------------------------------------------------- B: unseen convs AND unseen facets
    facet_list = np.array(sorted(labels.facet.unique()))
    conv_list = np.array(sorted(labels.conv_id.unique()))
    rng = np.random.default_rng(42)
    f_fold = dict(zip(rng.permutation(facet_list), np.arange(len(facet_list)) % N_FOLDS))
    c_fold = dict(zip(rng.permutation(conv_list), np.arange(len(conv_list)) % N_FOLDS))
    lf, lc = labels.facet.map(f_fold), labels.conv_id.map(c_fold)
    oofB = []
    for k in range(N_FOLDS):
        L_tr = labels[(lf != k) & (lc != k)]
        L_te = labels[(lf == k) & (lc == k)]
        judge = LiteJudge(seed=k).fit(texts, L_tr, fb=fb, verbose=False)
        trf = L_tr.groupby("facet").score
        kv = fb.facet_vecs(list(trf.mean().index))
        for cid, g in L_te.groupby("conv_id"):
            fl = g.facet.tolist()
            mu, _ = judge.predict_matrix(texts[cid], fl)
            pm, _ = knn_prior(fb.facet_vecs(fl), kv, trf.mean().values, trf.std(ddof=0).fillna(0).values)
            oofB.append(g.assign(pred=mu, base_global=L_tr.score.mean(), base_knn=pm, fold=k))
        print(f"  B fold {k+1}/{N_FOLDS} done ({len(L_tr)} train rows, {len(L_te)} test rows)")
    B = pd.concat(oofB)
    resB = {}
    for name, col in [("global_mean_baseline", "base_global"), ("knn_facet_prior_baseline", "base_knn"),
                      ("lite_judge_zero_shot", "pred")]:
        m = metrics(B.score, B[col])
        m["within_conv_spearman"] = within_conv_spearman(B, col) if col != "base_global" else None
        resB[name] = m
    report["B_unseen_conversations_and_facets"] = resB

    # ---------------------------------------------------------------- final model + latency
    judge = LiteJudge(seed=0).fit(texts, labels, fb=fb)
    judge.save(MODEL_PATH)
    report["final_model"] = judge.meta
    report["final_model"]["file_MB"] = round(os.path.getsize(MODEL_PATH) / 1e6, 2)

    facets = load_facets()
    sample = convs[0]["conversation"]
    judge.predict(sample, facets)  # warm-up
    times = []
    for c in convs[:20]:
        t0 = time.perf_counter()
        judge.predict(c["conversation"], facets)
        times.append(time.perf_counter() - t0)
    report["latency_399_facets_cpu_ms"] = {"median": round(1000 * float(np.median(times)), 1),
                                           "p95": round(1000 * float(np.percentile(times, 95)), 1),
                                           "cpu_threads": 2}
    report["total_runtime_s"] = round(time.time() - t_all, 1)

    with open(os.path.join(ROOT, "reports", "lite_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
