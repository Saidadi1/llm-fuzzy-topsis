import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings  # noqa: E402
from src.llm import LLMClient  # noqa: E402


class FakeBackend:
    """Scripted stand-in for an LLM: returns queued answers (or a function of the prompt)."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, system, prompt, **kw):
        self.prompts.append(prompt)
        a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        a = a(prompt) if callable(a) else a
        return a if isinstance(a, str) else json.dumps(a)


@pytest.fixture
def make_client(tmp_path):
    def _make(answers, use_cache=False):
        s = Settings(cache_path=str(tmp_path / "cache.sqlite"))
        return LLMClient(provider="fake", model="fake", settings=s, use_cache=use_cache,
                         backend=FakeBackend(answers))
    return _make
