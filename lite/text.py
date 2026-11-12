"""Tokenisation + word-vector utilities shared by training and inference."""
import os
import re

import numpy as np

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")

# Speaker tags like "Interviewer:" / "Person A:" at line start
_SPEAKER = re.compile(r"^\s*([A-Z][\w .'-]{0,30}):", re.M)

STOP = set("""a an the and or of to in on at for with by from as is are was were be been being it its this
that these those i you he she we they me him her us them my your his our their not no do does did
have has had so if then than but just also very can could would should will shall may might must
about into over under up down out off again once here there when where why how all any both each few
more most other some such only own same too s t don now""".split())


def tokenize(text: str):
    """Lowercase word tokens. Splits CamelCase ("SelfEsteem" -> self, esteem) and hyphens."""
    text = _CAMEL.sub(" ", str(text))
    return _TOKEN.findall(text.lower().replace("-", " ").replace("_", " "))


def split_turns(conversation: str):
    """Returns list of (speaker, utterance). Falls back to non-empty lines."""
    turns = []
    for line in str(conversation).splitlines():
        line = line.strip()
        if not line:
            continue
        m = _SPEAKER.match(line)
        if m:
            turns.append((m.group(1).strip(), line[m.end():].strip()))
        elif turns:
            sp, u = turns[-1]
            turns[-1] = (sp, (u + " " + line).strip())
        else:
            turns.append(("?", line))
    return turns


class WordVectors:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.words = z["words"]
        self.M = z["vectors"].astype(np.float32)
        self.M /= np.linalg.norm(self.M, axis=1, keepdims=True) + 1e-8
        self.index = {w: i for i, w in enumerate(self.words.tolist())}
        self.dim = self.M.shape[1]
        # rank-based IDF proxy: GloVe is ordered by corpus frequency (Zipf),
        # so log(rank) behaves like an IDF weight without needing a corpus.
        self.idf = np.log(np.arange(len(self.words)) + 10.0).astype(np.float32)

    _SUFFIXES = ("fulness", "iveness", "ousness", "ness", "ity", "ment", "ism", "ing", "ed", "s")

    def lookup(self, tok):
        """Index of a token; backs off by stripping suffixes ("courageousness" -> "courageous")."""
        if tok in self.index:
            return self.index[tok]
        for suf in self._SUFFIXES:
            if tok.endswith(suf) and len(tok) - len(suf) >= 4:
                root = tok[: -len(suf)]
                for cand in (root, root + "e", root + "ous", root + "y"):
                    if cand in self.index:
                        return self.index[cand]
        return None

    def ids(self, tokens, drop_stop=True):
        out = []
        for t in tokens:
            if drop_stop and t in STOP:
                continue
            i = self.lookup(t)
            if i is not None:
                out.append(i)
        return out

    def embed_ids(self, ids):
        if not ids:
            return np.zeros(self.dim, np.float32)
        w = self.idf[ids]
        v = (self.M[ids] * w[:, None]).sum(0) / w.sum()
        return v / (np.linalg.norm(v) + 1e-8)

    def embed(self, text):
        return self.embed_ids(self.ids(tokenize(text)))


_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "glove_50k_100d.npz")
_cache = {}


def get_vectors(path=_DEFAULT):
    if path not in _cache:
        _cache[path] = WordVectors(path)
    return _cache[path]
