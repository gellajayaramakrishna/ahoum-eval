from facets import FacetMatcher, canonicalize_scores, load_facets


def test_load_facets_clean_and_unique():
    fs = load_facets()
    assert len(fs) == 399
    assert len({f.lower() for f in fs}) == 399
    assert not any(f.endswith(":") for f in fs)
    assert "Democratic Leadership" in fs


def test_matcher_repairs_known_failure_modes():
    m = FacetMatcher(load_facets())
    assert m.match("risktaking") == "Risktaking"                          # casing
    assert m.match("Democratic Leadership:") == "Democratic Leadership"   # trailing colon
    assert m.match("Moral and Ethical Parameters: Frankness") == "Frankness"   # merged heading
    assert m.match("Operant-Learning rate") == "Psychological construct: Operant-Learning rate"
    assert m.match("Therapist Self-Awareness") is None                    # hallucinated facet


def test_canonicalize_counts():
    m = FacetMatcher(["Empathy", "Anger"])
    out, st = canonicalize_scores({"empathy": {"score": 4}, "ANGER ": {"score": 2}, "Joy": {"score": 5}}, m)
    assert set(out) == {"Empathy", "Anger"}
    assert st == {"exact": 2, "repaired": 0, "unmatched": 1}
