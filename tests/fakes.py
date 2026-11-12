import json
import re
from types import SimpleNamespace


class FakeGroq:
    """Mimics groq.Groq().chat.completions.create. `behaviour(facets, call_no)` -> dict or Exception."""

    def __init__(self, behaviour=None):
        self.calls = []
        self.behaviour = behaviour or (lambda facets, n: {f: {"score": 4, "confidence": 0.9} for f in facets})
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        prompt = kw["messages"][-1]["content"]
        block = prompt.split("FACETS (score every one, use the names exactly as written):")[1].split("SCALE")[0]
        facets = [l[2:].strip() for l in block.strip().splitlines() if l.startswith("- ")]
        self.calls.append({"facets": facets, "kwargs": kw})
        out = self.behaviour(facets, len(self.calls))
        if isinstance(out, Exception):
            raise out
        content = out if isinstance(out, str) else json.dumps(out)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))
