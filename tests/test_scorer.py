import pytest

import scorer
from fakes import FakeGroq

FACETS = ["Empathy", "Anger", "Hostility", "Merriness"]


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(scorer.time, "sleep", lambda *_: None)
    scorer._cache.clear()
    scorer._json_mode = True
    yield
    scorer.set_client(None)


def test_happy_path_and_json_mode():
    fake = FakeGroq()
    scorer.set_client(fake)
    out, st = scorer.score_conversation("A: hi\nB: hello", FACETS, batch_size=2, return_stats=True)
    assert set(out) == set(FACETS)
    assert all(v["source"] == "llm" and v["score"] == 4 for v in out.values())
    assert st["llm_calls"] == 2 and st["fallbacks"] == 0
    assert fake.calls[0]["kwargs"]["response_format"] == {"type": "json_object"}


def test_mangled_keys_clamping_and_reask():
    def beh(facets, n):
        if n == 1:   # mangled names, out-of-range values, one facet dropped
            return {"empathy": {"score": 9, "confidence": 3}, "ANGER": {"score": -2, "confidence": 0.5},
                    "Hostility:": 2}
        return {f: {"score": 2, "confidence": 0.6} for f in facets}
    fake = FakeGroq(beh)
    scorer.set_client(fake)
    out, st = scorer.score_conversation("x", FACETS, batch_size=4, return_stats=True)
    assert out["Empathy"] == {"score": 5, "confidence": 1.0, "source": "llm"}
    assert out["Anger"]["score"] == 1
    assert out["Hostility"]["score"] == 2
    assert out["Merriness"]["score"] == 2          # recovered by the targeted re-ask
    assert fake.calls[1]["facets"] == ["Merriness"]
    assert st["reasked"] == 1 and st["fallbacks"] == 0


def test_persistent_failure_falls_back_explicitly():
    scorer.set_client(FakeGroq(lambda f, n: "not json at all"))
    out, st = scorer.score_conversation("x", FACETS, batch_size=4, return_stats=True)
    assert all(v == {"score": 3, "confidence": 0.0, "source": "fallback"} for v in out.values())
    assert st["fallbacks"] == 4 and st["retries"] == 4


def test_cache_hit_skips_llm():
    fake = FakeGroq()
    scorer.set_client(fake)
    scorer.score_conversation("same", FACETS, batch_size=4)
    _, st = scorer.score_conversation("same", FACETS, batch_size=4, return_stats=True)
    assert len(fake.calls) == 1 and st["cache_hits"] == 1


def test_json_mode_rejected_is_disabled():
    def beh(facets, n):
        return Exception("response_format is not supported") if n == 1 else {f: {"score": 3, "confidence": 0.5} for f in facets}
    fake = FakeGroq(beh)
    scorer.set_client(fake)
    out = scorer.score_conversation("x", FACETS, batch_size=4)
    assert all(v["source"] == "llm" for v in out.values())
    assert "response_format" not in fake.calls[-1]["kwargs"]


def test_no_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    scorer.set_client(None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        scorer.get_client()
