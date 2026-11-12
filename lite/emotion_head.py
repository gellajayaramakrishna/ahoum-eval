"""
EmotionFacetHead - links the human-label emotion encoder to arbitrary facets.

For every facet f we fit a ridge regression on the conversation's mean GoEmotions profile p(c):
      teacher_score(c, f) - mean_f  ~  W_f . (p(c) - mu) / s
shrunk towards a *semantic prior* W0_f instead of towards zero:
      W0_f[e] = beta * max(0, cos(glove(f), glove(emotion e)) - 0.3)
so "Joyfulness" starts out tied to `joy`, and 40-50 training conversations only adjust it.
Without the prior, the per-facet ridge picked up spurious emotions and failed out of domain
(see README, "Human-label check").

A facet never seen in training gets W0_f plus the neighbours' learned correction (W - W0).
"""
import numpy as np


class EmotionFacetHead:
    def __init__(self, alpha=30.0, beta=6.0, sim_floor=0.3, knn_k=10, n_emotions=28):
        self.alpha, self.beta, self.sim_floor = alpha, beta, sim_floor
        self.knn_k = knn_k
        self.E = n_emotions

    def _prior(self, V):
        S = V @ self.EV.T                                      # (F, E) facet-emotion cosine
        return self.beta * np.maximum(S - self.sim_floor, 0) * self.s

    def fit(self, profiles, labels, facet_vec_fn, emotion_names):
        convs = sorted(labels.conv_id.unique())
        Z = np.vstack([profiles[c][: self.E] for c in convs])
        self.mu, self.s = Z.mean(0), float(Z.std()) + 1e-6
        Zs = (Z - self.mu) / self.s
        ci = {c: i for i, c in enumerate(convs)}
        names = list(emotion_names)[: self.E]
        ev = facet_vec_fn(names)
        if "neutral" in names:
            ev[names.index("neutral")] = 0.0
        self.EV = ev
        self.facets = sorted(labels.facet.unique())
        self.index = {f: j for j, f in enumerate(self.facets)}
        self.V = facet_vec_fn(self.facets)
        W0 = self._prior(self.V)
        W = np.zeros_like(W0, dtype=np.float32)
        I = self.alpha * np.eye(self.E)
        for j, (f, g) in enumerate(labels.groupby("facet", sort=True)):
            X = Zs[[ci[c] for c in g.conv_id]]
            y = g.score.values - g.score.mean()
            W[j] = np.linalg.solve(X.T @ X + I, X.T @ y + self.alpha * W0[j])
        self.W, self.D = W, W - W0                              # D = learned correction to the prior
        return self

    def coefs(self, facets, V, exclude_self=False):
        out = self._prior(V).astype(np.float32)
        need = []
        for i, f in enumerate(facets):
            if f in self.index and not exclude_self:
                out[i] = self.W[self.index[f]]
            else:
                need.append(i)
        if need:
            S = V[need] @ self.V.T
            if exclude_self:
                for r, i in enumerate(need):
                    j = self.index.get(facets[i])
                    if j is not None:
                        S[r, j] = -np.inf
            k = min(self.knn_k, S.shape[1] - 1)
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
            w = np.exp(np.take_along_axis(S, idx, axis=1) * 8.0)
            w /= w.sum(1, keepdims=True)
            out[need] += np.einsum("nk,nkd->nd", w, self.D[idx])
        return out

    def z(self, profile):
        return (profile[: self.E] - self.mu) / self.s

    def predict(self, profile, facets, V, exclude_self=False):
        return self.coefs(facets, V, exclude_self) @ self.z(profile)
