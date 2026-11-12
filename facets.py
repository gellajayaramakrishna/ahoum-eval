"""
Facet loading + name canonicalisation.

The raw CSV has two quirks that hurt the original LLM pipeline:
  * leading numbering ("793. Sufi practice")
  * 30 rows that end in ':' ("Democratic Leadership:"). In the prompt these look like
    section headings, so Llama often merged them with the next facet
    ("Moral and Ethical Parameters: Frankness") or dropped them (92% drop rate).

We keep all 399 facets but present a clean name (no numbering, no trailing colon),
and provide `canonicalize_key` to map whatever the model returns back onto a real facet.
"""
import os
import re

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(_HERE, "Facets Assignment.csv")


def _clean(name: str) -> str:
    name = re.sub(r"^\d+\.\s*", "", str(name)).strip()
    return name.rstrip(":").strip()


def load_raw_facets(csv_path=DEFAULT_CSV):
    """Facet names exactly as in the CSV (minus numbering). Kept for backwards compat."""
    df = pd.read_csv(csv_path)
    facets = df["Facets"].dropna().astype(str).str.strip().tolist()
    return [re.sub(r"^\d+\.\s*", "", f).strip() for f in facets if f]


def load_facets(csv_path=DEFAULT_CSV):
    """Clean, de-duplicated facet names (what every scorer in this repo uses)."""
    seen, out = set(), []
    for f in load_raw_facets(csv_path):
        c = _clean(f)
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


class FacetMatcher:
    """Maps model-returned keys (mangled casing, merged headings, numbering) to canonical facets."""

    def __init__(self, facets):
        self.facets = list(facets)
        self.by_norm = {_norm(f): f for f in self.facets}

    def match(self, key):
        k = _clean(key)
        n = _norm(k)
        if not n:
            return None
        if n in self.by_norm:
            return self.by_norm[n]
        # "Heading: Facet" or "Heading: 516. Facet" -> try the part after each colon
        parts = [p for p in re.split(r":\s*", k) if p.strip()]
        for i in range(len(parts) - 1, 0, -1):
            tail = _norm(_clean(": ".join(parts[i:])))
            if tail in self.by_norm:
                return self.by_norm[tail]
        # unique containment (e.g. "Operant-Learning rate" -> "Psychological construct: Operant-Learning rate")
        cands = [f for nf, f in self.by_norm.items() if len(n) >= 6 and (n in nf or nf in n)]
        if len(cands) == 1:
            return cands[0]
        return None


def canonicalize_scores(scores: dict, matcher: FacetMatcher):
    """Returns (clean_scores, stats). On collisions the first label wins."""
    out, stats = {}, {"exact": 0, "repaired": 0, "unmatched": 0}
    for key, val in scores.items():
        m = matcher.match(key)
        if m is None:
            stats["unmatched"] += 1
            continue
        stats["exact" if _norm(key) == _norm(m) else "repaired"] += 1
        out.setdefault(m, val)
    return out, stats


if __name__ == "__main__":
    fs = load_facets()
    print(f"Total facets loaded: {len(fs)}")
    print("First 5:", fs[:5])
