"""Provider-agnostic LLM wrapper with JSON-only outputs validated by pydantic.

    client = LLMClient()                       # provider from LLM_PROVIDER
    profile = client.complete_json(prompt, CandidateProfile)

Design choices
--------------
* One interface, three backends (Ollama by default, OpenAI, Gemini) so that the
  rest of the pipeline never depends on a vendor.
* The JSON schema of the pydantic model is passed to the provider when it
  supports constrained decoding (Ollama `format`, Gemini `responseSchema`)
  and is always repeated in the prompt.
* Invalid JSON / schema violations trigger a retry where the validation error
  is fed back to the model ("self-repair"), up to `max_retries` times.
* Every raw call is cached (see cache.py) -> reproducible results.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from .cache import LLMCache, make_key
from .config import DEFAULT_MODELS, Settings, get_settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMTimeout(LLMError):
    pass


MAX_OUTPUT_TOKENS = 1000  # a cap stops degenerate, endlessly repeating generations
                          # (a full 6-criterion scoring answer is ~550 tokens)


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in `text` (tolerates ```json fences
    and <think> blocks emitted by reasoning models)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON found in model output")
    depth, in_str, esc = 0, False, False
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON in model output")


class LLMClient:
    """Thin client over Ollama / OpenAI / Gemini REST APIs."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        settings: Settings | None = None,
        use_cache: bool = True,
        backend: Callable[..., str] | None = None,
    ):
        self.settings = settings or get_settings()
        self.provider = (provider or self.settings.provider).lower()
        self.model = model or self.settings.model or DEFAULT_MODELS.get(self.provider, "")
        self.cache = LLMCache(self.settings.cache_path) if use_cache else None
        # `backend` lets tests inject a fake model: backend(system, prompt, **kw) -> str
        self._backend = backend
        self.n_calls = 0       # real (non-cached) calls
        self.n_cache_hits = 0

    # ------------------------------------------------------------------ raw
    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        seed: int | None = None,
        schema: dict | None = None,
    ) -> str:
        seed = self.settings.seed if seed is None else seed
        request = dict(provider=self.provider, model=self.model, system=system, prompt=prompt,
                       temperature=temperature, seed=seed, schema=schema)
        key = make_key(**request)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.n_cache_hits += 1
                return hit
        t0 = time.time()
        if self._backend is not None:
            out = self._backend(system=system, prompt=prompt, temperature=temperature,
                                seed=seed, schema=schema)
        elif self.provider == "ollama":
            out = self._ollama(system, prompt, temperature, seed, schema)
        elif self.provider == "openai":
            out = self._openai(system, prompt, temperature, seed)
        elif self.provider == "gemini":
            out = self._gemini(system, prompt, temperature, seed, schema)
        else:
            raise LLMError(f"unknown provider {self.provider}")
        self.n_calls += 1
        log.info("LLM %s/%s call took %.1fs", self.provider, self.model, time.time() - t0)
        if self.cache is not None:
            self.cache.set(key, self.provider, self.model, request, out)
        return out

    # --------------------------------------------------------------- json
    def complete_json(
        self,
        prompt: str,
        model_cls: type[T],
        system: str = "",
        temperature: float = 0.0,
        seed: int | None = None,
        max_retries: int = 3,
        validator: Callable[[T], None] | None = None,
    ) -> T:
        """Ask for JSON matching `model_cls`; validate; retry with feedback.

        `validator` may raise ValueError for semantic checks beyond the schema
        (e.g. "every criterion must be scored"); the message is fed back too.
        """
        schema = model_cls.model_json_schema()
        sys_msg = (system + "\n\n" if system else "") + (
            "Answer ONLY with a single JSON object, no prose, no markdown. "
            "It must validate against this JSON schema:\n" + json.dumps(schema, ensure_ascii=False)
        )
        current = prompt
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                raw = self.generate(current, system=sys_msg, temperature=temperature,
                                    seed=None if seed is None else seed + attempt * 1000, schema=schema)
            except LLMTimeout as e:
                # degenerate generation (e.g. endless repetition): retry with another seed
                last_err = e
                log.warning("LLM timeout on attempt %d", attempt + 1)
                seed = (self.settings.seed if seed is None else seed) + 1
                temperature = max(temperature, 0.3)  # greedy decoding would repeat the same loop
                continue
            try:
                obj = model_cls.model_validate(extract_json(raw))
                if validator is not None:
                    validator(obj)
                return obj
            except (ValueError, ValidationError) as e:
                last_err = e
                log.warning("invalid JSON on attempt %d: %s", attempt + 1, str(e)[:300])
                current = (
                    f"{prompt}\n\nYour previous answer was invalid:\n{raw[:2000]}\n\n"
                    f"Error: {str(e)[:1000]}\nReturn a corrected JSON object only."
                )
        raise LLMError(f"no valid JSON after {max_retries + 1} attempts: {last_err}")

    # ----------------------------------------------------------- backends
    def _post(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=self.settings.timeout)
                if r.status_code in (429, 500, 502, 503):
                    time.sleep(2 ** attempt * 5)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.Timeout as e:
                raise LLMTimeout(f"{url} timed out after {self.settings.timeout}s") from e
            except requests.ConnectionError as e:
                if attempt == 2:
                    raise LLMError(f"cannot reach {url}: {e}") from e
                time.sleep(2)
        raise LLMError(f"{url} failed after retries: {r.status_code} {r.text[:300]}")

    def _ollama(self, system, prompt, temperature, seed, schema) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
            "think": False,  # disable chain-of-thought on reasoning models (qwen3.x)
            "format": schema if schema else "",
            "options": {"temperature": temperature, "seed": seed, "num_ctx": 8192,
                        "num_predict": MAX_OUTPUT_TOKENS},
        }
        data = self._post(f"{self.settings.ollama_host}/api/chat", payload)
        return data["message"]["content"]

    def _openai(self, system, prompt, temperature, seed) -> str:
        if not self.settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": temperature,
            "seed": seed,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        data = self._post(f"{self.settings.openai_base_url}/chat/completions", payload, headers)
        return data["choices"][0]["message"]["content"]

    def _gemini(self, system, prompt, temperature, seed, schema) -> str:
        if not self.settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
               f"?key={self.settings.gemini_api_key}")
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "seed": seed,
                                 "maxOutputTokens": MAX_OUTPUT_TOKENS,
                                 "responseMimeType": "application/json"},
        }
        data = self._post(url, payload)
        return data["candidates"][0]["content"]["parts"][0]["text"]
