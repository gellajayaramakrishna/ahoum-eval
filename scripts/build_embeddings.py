"""
Builds the compact word-vector table the distilled judge ships with.

Source: GloVe 6B 100d (Wikipedia + Gigaword, Pennington et al. 2014), via the gensim-data GitHub release:
  https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz

Keeps the 50k most frequent words + every word appearing in a facet name, stored as float16
(~10 MB) so the whole judge runs on a free 512 MB Render instance with no GPU and no torch.

Run:  python scripts/build_embeddings.py path/to/glove-wiki-gigaword-100.gz
"""
import gzip
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from facets import load_facets  # noqa: E402
from lite.text import tokenize  # noqa: E402

TOP_N = 50_000


def main(src):
    facet_vocab = {w for f in load_facets() for w in tokenize(f)}
    words, vecs = [], []
    with gzip.open(src, "rt", encoding="utf-8") as fh:
        first = fh.readline().split()
        if len(first) != 2:  # no word2vec header line
            fh.seek(0)
        for i, line in enumerate(fh):
            parts = line.rstrip().split(" ")
            w = parts[0]
            if i < TOP_N or w in facet_vocab:
                words.append(w)
                vecs.append(np.asarray(parts[1:], dtype=np.float32))
    M = np.vstack(vecs).astype(np.float16)
    out = os.path.join(ROOT, "models", "glove_50k_100d.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, words=np.array(words), vectors=M)
    missing = sorted(facet_vocab - set(words))
    print(f"saved {M.shape} -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")
    print(f"facet words without a vector: {len(missing)} / {len(facet_vocab)}  e.g. {missing[:15]}")


if __name__ == "__main__":
    main(sys.argv[1])
