"""
LiteJudge - a distilled, CPU-only facet scorer.

Teacher : Llama 3.1 8B (Groq) labels from results/all_scores.json
Student : LightGBM on GloVe-based facet-conversation match features (lite/features.py)
Output  : per facet score in [1,5] + an uncertainty estimate (predicted absolute error)

Uncertainty is a second LightGBM trained on *out-of-fold* residuals of the first one,
so it learns where the student tends to disagree with the teacher. The hybrid scorer
uses it to decide which facets are worth an LLM call.
"""
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

from .emotion import get_emotion_encoder
from .emotion_head import EmotionFacetHead
from .features import FeatureBuilder, knn_prior

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "lite_judge.joblib")

SCORE_PARAMS = dict(objective="regression", learning_rate=0.04, num_leaves=31, min_child_samples=30,
                    feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                    n_estimators=500, verbose=-1, n_jobs=2)
ERR_PARAMS = dict(objective="regression_l1", learning_rate=0.05, num_leaves=15, min_child_samples=50,
                  feature_fraction=0.5, n_estimators=250, verbose=-1, n_jobs=2)


class LiteJudge:
    def __init__(self, zero_shot_frac=0.3, knn_k=10, seed=0, residual=True):
        self.fb = None
        self.zero_shot_frac = zero_shot_frac
        self.knn_k = knn_k
        self.seed = seed
        self.residual = residual          # learn (score - facet prior) instead of the raw score
        self.ref_texts = []
        self.facet_stats = {}     # facet -> (mean, std, n)
        self.known_facets = []
        self.known_vecs = None
        self.model = None
        self.err_model = None
        self.meta = {}

    # ------------------------------------------------------------------ priors
    def _fit_facet_stats(self, labels):
        g = labels.groupby("facet")["score"]
        self.facet_stats = {f: (float(m), float(s), int(n))
                            for f, m, s, n in zip(g.mean().index, g.mean(), g.std(ddof=0).fillna(0), g.count())}
        self.known_facets = sorted(self.facet_stats)
        self.known_vecs = self.fb.facet_vecs(self.known_facets)
        self._known_mean = np.array([self.facet_stats[f][0] for f in self.known_facets])
        self._known_std = np.array([self.facet_stats[f][1] for f in self.known_facets])

    def priors_for(self, facets, facet_vecs, force_zero_shot=False):
        """(F,3) [mean, std, is_own]. Own stats if the facet was in training, else kNN prior."""
        out = np.zeros((len(facets), 3), np.float32)
        unknown = [i for i, f in enumerate(facets) if force_zero_shot or f not in self.facet_stats]
        for i, f in enumerate(facets):
            if i not in unknown:
                m, s, _ = self.facet_stats[f]
                out[i] = (m, s, 1.0)
        if unknown:
            V = facet_vecs[unknown]
            if force_zero_shot:
                # exclude the facet itself from its own neighbourhood
                S_self = [self.known_facets.index(facets[i]) if facets[i] in self.facet_stats else -1 for i in unknown]
                S = V @ self.known_vecs.T
                for r, j in enumerate(S_self):
                    if j >= 0:
                        S[r, j] = -np.inf
                k = self.knn_k
                idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
                sims = np.take_along_axis(S, idx, axis=1)
                w = np.exp(sims * 8.0)
                w /= w.sum(1, keepdims=True)
                m, s = (w * self._known_mean[idx]).sum(1), (w * self._known_std[idx]).sum(1)
            else:
                m, s = knn_prior(V, self.known_vecs, self._known_mean, self._known_std, k=self.knn_k)
            out[unknown, 0], out[unknown, 1], out[unknown, 2] = m, s, 0.0
        return out

    # ------------------------------------------------------------------ design matrix
    def _design(self, conv_texts, labels, train_mode):
        """Builds X for every labelled row. In train mode a random subset of facets gets a
        zero-shot (kNN) prior so a single model serves both seen and unseen facets; seen facets
        get a leave-one-out mean so the row's own label never leaks into its prior."""
        rng = np.random.default_rng(self.seed)
        facets = sorted(labels.facet.unique())
        fvec = dict(zip(facets, self.fb.facet_vecs(facets)))
        zs = set(rng.choice(facets, int(len(facets) * self.zero_shot_frac), replace=False)) if train_mode else set()
        self._zs = zs
        sums = labels.groupby("facet")["score"].agg(["sum", "count"])
        sq = labels.assign(s2=labels.score ** 2).groupby("facet")["s2"].sum()

        X, rows = [], []
        for cid, grp in labels.groupby("conv_id", sort=False):
            enc = self.fb.encode(conv_texts[cid])
            fl = grp.facet.tolist()
            V = np.vstack([fvec[f] for f in fl])
            P = self.priors_for(fl, V)
            if train_mode:
                for i, (f, y) in enumerate(zip(fl, grp.score.values)):
                    if f in zs:
                        continue
                    n = sums.loc[f, "count"]
                    if n > 1:
                        m = (sums.loc[f, "sum"] - y) / (n - 1)
                        v = (sq.loc[f] - y * y) / (n - 1) - m * m
                        P[i] = (m, np.sqrt(max(v, 0)), 1.0)
                zi = [i for i, f in enumerate(fl) if f in zs]
                if zi:
                    P[zi] = self.priors_for([fl[i] for i in zi], V[zi], force_zero_shot=True)
            X.append(self.fb.build(enc, V, P, fl))
            rows.append(grp.index.values)
        return np.vstack(X), np.concatenate(rows)

    # ------------------------------------------------------------------ fit / predict
    def fit(self, conv_texts, labels, fb=None, inner_folds=4, verbose=True):
        t0 = time.time()
        labels = labels.reset_index(drop=True)
        self.fb = fb or FeatureBuilder()
        self.ref_texts = [conv_texts[c] for c in sorted(labels.conv_id.unique())]
        self.fb.set_reference(self.ref_texts)
        self._fit_facet_stats(labels)
        X, idx = self._design(conv_texts, labels, train_mode=True)
        y = labels.score.values[idx].astype(float)
        w = np.clip(labels.confidence.values[idx], 0.2, 1.0)
        groups = labels.conv_id.values[idx]
        X = np.hstack([X, self._emotion_block_train(conv_texts, labels, idx, inner_folds)])
        PRIOR_COL = self.fb.feature_names().index("prior_mean")
        base = X[:, PRIOR_COL] if self.residual else np.zeros_like(y)
        self._prior_col = PRIOR_COL
        self._own_col = self.fb.feature_names().index("prior_is_own")
        self._emo_col = len(self.fb.feature_names())        # first column of the emotion block
        t = y - base
        if getattr(self, "keep_design", False):
            self._Xtr, self._ttr, self._wtr, self._ytr, self._gtr = X, t, w, y, groups

        # out-of-fold predictions -> error model (learns where the student disagrees with the teacher)
        oof = np.zeros_like(y)
        for tr, te in GroupKFold(inner_folds).split(X, y, groups):
            ms, mz = self._fit_heads(X[tr], t[tr], w[tr])
            oof[te] = self._combine(X[te], ms, mz) + base[te]
        self.err_model = lgb.LGBMRegressor(**ERR_PARAMS, random_state=self.seed).fit(
            np.column_stack([X, oof]), np.abs(np.clip(oof, 1, 5) - y))
        self.model, self.model_zs = self._fit_heads(X, t, w)
        self.meta = {"n_train_rows": int(len(y)), "n_conversations": int(len(set(groups))),
                     "n_facets": len(self.known_facets), "n_features": int(X.shape[1]),
                     "fit_seconds": round(time.time() - t0, 1)}
        if verbose:
            print("fit:", self.meta)
        return self

    # ------------------------------------------------------------------ prediction heads
    EMO_WEIGHT = 0.5   # final residual = 0.5 * emotion-head (linear) + 0.5 * LightGBM

    def _fit_heads(self, X, t, w):
        """Two GBMs: one trained on rows whose facet prior is the facet's own teacher mean (seen
        facets), one on all rows incl. the simulated zero-shot ones (used for unseen facets)."""
        own = X[:, self._own_col] == 1
        ms = lgb.LGBMRegressor(**SCORE_PARAMS, random_state=self.seed).fit(X[own], t[own], sample_weight=w[own])
        mz = lgb.LGBMRegressor(**SCORE_PARAMS, random_state=self.seed).fit(X, t, sample_weight=w)
        return ms, mz

    def _combine(self, X, ms, mz):
        own = X[:, self._own_col] == 1
        g = np.where(own, ms.predict(X), mz.predict(X))
        return self.EMO_WEIGHT * X[:, self._emo_col] + (1 - self.EMO_WEIGHT) * g

    # ------------------------------------------------------------------ emotion features
    N_EMO_Z = 28

    def _emotion_block(self, profile, facets, V, exclude_self=False, head=None):
        head = head or self.emo_head
        f = head.predict(profile, facets, V, exclude_self)
        z = np.tile(head.z(profile)[: self.N_EMO_Z], (len(facets), 1))
        return np.column_stack([f, z])

    def _emotion_block_train(self, conv_texts, labels, idx, inner_folds):
        """Out-of-fold emotion-head feature for every training row (no label leakage)."""
        self.emo = get_emotion_encoder()
        if self.emo is None:
            raise RuntimeError("models/emotion_encoder.joblib missing - run scripts/train_emotion.py first")
        convs = np.array(sorted(labels.conv_id.unique()))
        self._profiles = {c: self.emo.conversation_profile(conv_texts[c]) for c in convs}
        pos = {r: i for i, r in enumerate(idx)}
        out = np.zeros((len(idx), 1 + self.N_EMO_Z), np.float32)
        for tr, te in GroupKFold(min(inner_folds, len(convs))).split(convs, groups=convs):
            head = EmotionFacetHead().fit(self._profiles, labels[labels.conv_id.isin(convs[tr])], self.fb.facet_vecs,
                                           self.emo.names)
            for c in convs[te]:
                g = labels[labels.conv_id == c]
                fl = g.facet.tolist()
                V = self.fb.facet_vecs(fl)
                blk = self._emotion_block(self._profiles[c], fl, V, head=head)
                zi = [i for i, f in enumerate(fl) if f in self._zs]
                if zi:
                    blk[zi] = self._emotion_block(self._profiles[c], [fl[i] for i in zi], V[zi],
                                                  exclude_self=True, head=head)
                out[[pos[r] for r in g.index.values]] = blk
        self.emo_head = EmotionFacetHead().fit(self._profiles, labels, self.fb.facet_vecs, self.emo.names)
        return out

    def design_matrix(self, text, facets, force_zero_shot=False):
        enc = self.fb.encode(text)
        V = self.fb.facet_vecs(facets)
        P = self.priors_for(facets, V, force_zero_shot=force_zero_shot)
        X = self.fb.build(enc, V, P, list(facets))
        X = np.hstack([X, self._emotion_block(self.emo.conversation_profile(text), list(facets), V,
                                              exclude_self=force_zero_shot)])
        return X

    def predict_matrix(self, text, facets, force_zero_shot=False):
        X = self.design_matrix(text, facets, force_zero_shot)
        mu = self._combine(X, self.model, self.model_zs) + (X[:, self._prior_col] if self.residual else 0.0)
        err = self.err_model.predict(np.column_stack([X, mu]))
        return np.clip(mu, 1, 5), np.maximum(err, 0)

    def predict(self, text, facets):
        mu, err = self.predict_matrix(text, facets)
        out = {}
        for f, m, e in zip(facets, mu, err):
            out[f] = {"score": int(np.clip(np.rint(m), 1, 5)), "raw": round(float(m), 3),
                      "uncertainty": round(float(e), 3),
                      "confidence": round(float(np.clip(1 - e / 2.0, 0, 1)), 3),
                      "source": "lite"}
        return out

    # ------------------------------------------------------------------ io
    def save(self, path=MODEL_PATH):
        keep = (self.fb, self.emo, getattr(self, "_profiles", None))
        self.fb, self.emo, self._profiles = None, None, None
        try:
            joblib.dump(self, path, compress=3)
        finally:
            self.fb, self.emo, self._profiles = keep

    @staticmethod
    def load(path=MODEL_PATH):
        m = joblib.load(path)
        m.fb = FeatureBuilder()
        m.fb.set_reference(m.ref_texts)
        m.emo = get_emotion_encoder()
        return m

    def feature_names(self):
        return self.fb.feature_names() + ["emo_head"] + [f"emo_z_{n}" for n in self.emo.names[: self.N_EMO_Z]]
