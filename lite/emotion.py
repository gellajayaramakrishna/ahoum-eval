"""
Emotion encoder trained on human labels (GoEmotions, Demszky et al. 2020: 58k Reddit comments,
27 emotions + neutral, labelled by 3+ raters).

Model: TF-IDF (word 1-2 grams + char 2-5 grams) -> one-vs-rest logistic regression,
per-label decision thresholds tuned on the dev split.

Used two ways in Ahoum:
  1. as conversation features for the distilled judge (per-turn emotion profile)
  2. as an independent, human-grounded reference to sanity-check the LLM judge's emotion facets
"""
import os

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from .text import split_turns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMO_PATH = os.path.join(ROOT, "models", "emotion_encoder.joblib")
GOEMO_DIR = os.path.join(ROOT, "data", "external", "goemotions")


def load_goemotions(split):
    texts, labels = [], []
    with open(os.path.join(GOEMO_DIR, f"{split}.tsv"), encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            texts.append(parts[0])
            labels.append([int(x) for x in parts[1].split(",")])
    names = open(os.path.join(GOEMO_DIR, "emotions.txt")).read().split()
    Y = np.zeros((len(texts), len(names)), np.int8)
    for i, ls in enumerate(labels):
        Y[i, ls] = 1
    return texts, Y, names


class EmotionEncoder:
    def __init__(self, C=4.0):
        self.C = C
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=80000,
                                    sublinear_tf=True)
        self.clf = None
        self.names = None
        self.thresholds = None

    def _X(self, texts, fit=False):
        if fit:
            return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def fit(self, texts, Y, names, dev_texts=None, dev_Y=None):
        self.names = list(names)
        X = self._X(texts, fit=True)
        self.clf = OneVsRestClassifier(
            LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced", solver="liblinear"), n_jobs=1
        ).fit(X, Y)
        self.thresholds = np.full(len(names), 0.5)
        if dev_texts is not None:
            P = self.predict_proba(dev_texts)
            grid = np.linspace(0.1, 0.95, 35)
            for j in range(len(names)):
                f1s = []
                for t in grid:
                    p = P[:, j] >= t
                    tp = np.sum(p & (dev_Y[:, j] == 1))
                    f1s.append(2 * tp / max(1, p.sum() + dev_Y[:, j].sum()))
                self.thresholds[j] = grid[int(np.argmax(f1s))]
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._X(texts))

    def predict(self, texts):
        return (self.predict_proba(texts) >= self.thresholds).astype(int)

    # ---------------------------------------------------------------- conversation-level
    def conversation_profile(self, conversation):
        """(2*E,) = [mean over turns, max over turns] of per-turn emotion probabilities."""
        turns = [u for _, u in split_turns(conversation)] or [conversation]
        P = self.predict_proba(turns)
        return np.concatenate([P.mean(0), P.max(0)]).astype(np.float32)

    def save(self, path=EMO_PATH):
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path=EMO_PATH):
        return joblib.load(path)


_cache = {}


def get_emotion_encoder(path=EMO_PATH):
    if path not in _cache:
        _cache[path] = EmotionEncoder.load(path) if os.path.exists(path) else None
    return _cache[path]
