import pytest
from pydantic import BaseModel

from src.llm import LLMError, extract_json


class Item(BaseModel):
    name: str
    value: int


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('<think>hmm {"x":0}</think> Sure: {"a": 3} done') == {"a": 3}
    assert extract_json('noise {"a": {"b": "}"}} tail') == {"a": {"b": "}"}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_complete_json_retries_with_feedback(make_client):
    client = make_client(["not json", '{"name": "x"}', '{"name": "x", "value": 3}'])
    out = client.complete_json("give item", Item)
    assert out == Item(name="x", value=3)
    prompts = client._backend.prompts
    assert len(prompts) == 3 and "previous answer was invalid" in prompts[2]


def test_complete_json_gives_up(make_client):
    client = make_client(["bad"])
    with pytest.raises(LLMError):
        client.complete_json("give item", Item, max_retries=1)


def test_semantic_validator(make_client):
    client = make_client(['{"name": "x", "value": -1}', '{"name": "x", "value": 1}'])

    def positive(o):
        if o.value < 0:
            raise ValueError("value must be positive")

    assert client.complete_json("p", Item, validator=positive).value == 1


def test_cache_makes_calls_reproducible(make_client):
    client = make_client(['{"name": "a", "value": 1}', '{"name": "b", "value": 2}'], use_cache=True)
    first = client.complete_json("same prompt", Item)
    second = client.complete_json("same prompt", Item)
    assert first == second
    assert client.n_calls == 1 and client.n_cache_hits == 1
