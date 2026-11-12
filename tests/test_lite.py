import time

import numpy as np
import pytest

from conftest import SAMPLE
from facets import load_facets
from hybrid import get_lite_judge


@pytest.fixture(scope="module")
def judge():
    return get_lite_judge()


def test_scores_every_facet_in_range(judge):
    fs = load_facets()
    out = judge.predict(SAMPLE, fs)
    assert set(out) == set(fs)
    for v in out.values():
        assert 1 <= v["score"] <= 5 and 1 <= v["raw"] <= 5
        assert 0 <= v["confidence"] <= 1 and v["uncertainty"] >= 0
        assert v["source"] == "lite"


def test_deterministic_and_fast(judge):
    fs = load_facets()
    a = judge.predict(SAMPLE, fs)
    t0 = time.perf_counter()
    b = judge.predict(SAMPLE, fs)
    assert time.perf_counter() - t0 < 1.0
    assert a == b


def test_unseen_facet_zero_shot(judge):
    out = judge.predict(SAMPLE, ["Fear of failure", "Love of cooking pasta"])
    assert set(out) == {"Fear of failure", "Love of cooking pasta"}


def test_emotion_sensitivity(judge):
    happy = "A: We won the championship!!! I'm so happy and proud of you all.\nB: Best day ever, thank you everyone!"
    angry = "A: You ruined everything, I'm furious. Get out.\nB: Don't yell at me, you're being a jerk and I hate this."
    f = ["Joyfulness", "Hostility"]
    h, g = judge.predict(happy, f), judge.predict(angry, f)
    assert h["Joyfulness"]["raw"] > g["Joyfulness"]["raw"]
    assert g["Hostility"]["raw"] > h["Hostility"]["raw"]


def test_empty_and_unstructured_text(judge):
    for txt in ["hello", "no speakers here at all, just one line of text"]:
        out = judge.predict(txt, load_facets()[:10])
        assert len(out) == 10 and all(np.isfinite(v["raw"]) for v in out.values())
