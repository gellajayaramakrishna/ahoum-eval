"""
Feature builder for the distilled judge.

For each (conversation, facet) pair the model sees:
  * semantic match   - how close the facet's meaning is to the conversation, overall, per turn and per word
  * interaction      - elementwise product of conversation and facet vectors (lets the trees learn
                       "this direction in conversation space matters for facets in that direction")
  * facet identity   - the facet vector itself (generalises to facets never seen in training)
  * facet prior      - teacher mean/std for this facet if known, else a similarity-weighted prior
                       from the k nearest known facets
  * conversation shape - turns, length, speakers, questions, exclamations, negations, 2nd-person usage
"""
import re

import numpy as np

from .text import STOP, get_vectors, split_turns, tokenize

_NEG = {"not", "no", "never", "don't", "can't", "won't", "isn't", "wasn't", "didn't", "nothing", "nobody"}


class ConversationEncoding:
    """All per-conversation tensors, computed once and reused for every facet."""

    def __init__(self, text, wv):
        self.text = text
        turns = split_turns(text) or [("?", text)]
        self.turns = turns
        tok_all = tokenize(text)
        ids_all = wv.ids(tok_all)
        self.vec = wv.embed_ids(ids_all)
        self.turn_vecs = np.vstack([wv.embed(u) for _, u in turns])
        uniq = sorted(set(ids_all))
        self.word_vecs = wv.M[uniq] if uniq else np.zeros((1, wv.dim), np.float32)
        n_words = max(1, len(tok_all))
        self.shape = np.array([
            len(turns),
            np.log1p(n_words),
            len({s for s, _ in turns}),
            text.count("?") / len(turns),
            text.count("!") / len(turns),
            sum(t in _NEG for t in tok_all) / n_words,
            sum(t in ("you", "your", "you're") for t in tok_all) / n_words,
            sum(t in ("i", "me", "my", "i'm") for t in tok_all) / n_words,
            len(re.findall(r"\*[^*]+\*", text)) / len(turns),   # stage directions *sighs*
            len(set(tok_all) - STOP) / n_words,                  # lexical diversity
        ], dtype=np.float32)


SHAPE_NAMES = ["n_turns", "log_words", "n_speakers", "q_per_turn", "excl_per_turn", "neg_rate",
               "you_rate", "i_rate", "stage_dir_per_turn", "lex_div"]


class FeatureBuilder:
    def __init__(self, wv=None):
        self.wv = wv or get_vectors()
        self.dim = self.wv.dim
        self.ref = []
        self._ref_cache = {}

    # ------------------------------------------------------------------ reference corpus
    def set_reference(self, texts):
        """Reference conversations used to normalise similarities per facet. Raw cosine scores
        are dominated by the facet's wording (some facet names are just 'closer' to everything);
        z-scoring against a reference corpus isolates what is specific to THIS conversation."""
        self.ref = [self.encode(t) for t in texts]
        self._ref_cache = {}

    @staticmethod
    def _sims(enc, V):
        tsims = V @ enc.turn_vecs.T
        wsims = V @ enc.word_vecs.T
        k = min(5, wsims.shape[1])
        return np.column_stack([V @ enc.vec, tsims.max(1), -np.sort(-wsims, axis=1)[:, :k].mean(1)])

    def ref_stats(self, facets, V):
        """(F, 6): mean and std of [sim_conv, sim_turn_max, sim_word_top5] over the reference corpus."""
        todo = [i for i, f in enumerate(facets) if f not in self._ref_cache]
        if todo and self.ref:
            Vt = V[todo]
            S = np.stack([self._sims(e, Vt) for e in self.ref], 0)       # (R, F', 3)
            st = np.hstack([S.mean(0), S.std(0) + 1e-3])
            for j, i in enumerate(todo):
                self._ref_cache[facets[i]] = st[j]
        if not self.ref:
            return np.tile(np.array([0, 0, 0, 1, 1, 1], np.float32), (len(facets), 1))
        return np.vstack([self._ref_cache[f] for f in facets])

    def facet_vecs(self, facets):
        return np.vstack([self.wv.embed(f) for f in facets])

    def encode(self, text):
        return ConversationEncoding(text, self.wv)

    def build(self, enc: ConversationEncoding, facet_vecs: np.ndarray, priors: np.ndarray, facets=None):
        """
        enc: one conversation; facet_vecs: (F, d); priors: (F, 3) = [prior_mean, prior_std, prior_is_own]
        facets: names (needed for the reference-normalised features)
        Returns (F, n_features)
        """
        F = facet_vecs.shape[0]
        conv_sim = facet_vecs @ enc.vec                         # (F,)
        tsims = facet_vecs @ enc.turn_vecs.T                    # (F, T)
        wsims = facet_vecs @ enc.word_vecs.T                    # (F, W)
        k = min(5, wsims.shape[1])
        top_w = -np.sort(-wsims, axis=1)[:, :k].mean(1)
        sim_block = np.column_stack([
            conv_sim,
            tsims.max(1), tsims.mean(1), tsims.std(1),
            wsims.max(1), top_w,
            (wsims > 0.6).sum(1), (wsims > 0.5).mean(1),
            tsims.argmax(1) / max(1, tsims.shape[1] - 1),     # where in the conversation it peaks
        ])
        rs = self.ref_stats(facets if facets is not None else [str(i) for i in range(F)], facet_vecs)
        raw = np.column_stack([conv_sim, tsims.max(1), top_w])
        z = (raw - rs[:, :3]) / rs[:, 3:]
        inter = facet_vecs * (enc.vec - self._ref_mean_vec())[None, :]
        shape = np.repeat(enc.shape[None, :], F, axis=0)
        return np.hstack([sim_block, z, raw - rs[:, :3], priors, shape, inter, facet_vecs]).astype(np.float32)

    def _ref_mean_vec(self):
        if not self.ref:
            return np.zeros(self.dim, np.float32)
        if not hasattr(self, "_rmv") or self._rmv_n != len(self.ref):
            self._rmv = np.mean([e.vec for e in self.ref], 0)
            self._rmv_n = len(self.ref)
        return self._rmv

    def feature_names(self):
        sim = ["sim_conv", "sim_turn_max", "sim_turn_mean", "sim_turn_std", "sim_word_max", "sim_word_top5",
               "n_words_sim_gt_0.6", "frac_words_sim_gt_0.5", "peak_turn_pos"]
        z = ["z_sim_conv", "z_sim_turn_max", "z_sim_word_top5", "d_sim_conv", "d_sim_turn_max", "d_sim_word_top5"]
        pri = ["prior_mean", "prior_std", "prior_is_own"]
        return sim + z + pri + SHAPE_NAMES + [f"inter_{i}" for i in range(self.dim)] + \
            [f"facet_{i}" for i in range(self.dim)]


def knn_prior(target_vecs, known_vecs, known_mean, known_std, k=10, exclude_self=False):
    """Similarity-weighted mean/std of the k most similar known facets (zero-shot facet prior)."""
    S = target_vecs @ known_vecs.T
    if exclude_self:
        np.fill_diagonal(S, -np.inf)
    k = min(k, S.shape[1] - (1 if exclude_self else 0))
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    sims = np.take_along_axis(S, idx, axis=1)
    w = np.exp(sims * 8.0)
    w /= w.sum(1, keepdims=True)
    return (w * known_mean[idx]).sum(1), (w * known_std[idx]).sum(1)
