"""Persistent cache of LLM calls (SQLite).

The key is a SHA-256 hash of every parameter that can change the answer
(provider, model, system prompt, prompt, temperature, seed, schema). Running
the pipeline twice therefore returns byte-identical answers, which makes the
reported results reproducible and free to re-run.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


def make_key(**params: Any) -> str:
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLMCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS calls ("
                " key TEXT PRIMARY KEY, provider TEXT, model TEXT,"
                " request TEXT, response TEXT, created REAL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def get(self, key: str) -> str | None:
        with self._lock, self._connect() as con:
            row = con.execute("SELECT response FROM calls WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, provider: str, model: str, request: dict, response: str) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO calls VALUES (?, ?, ?, ?, ?, ?)",
                (key, provider, model, json.dumps(request, ensure_ascii=False, default=str),
                 response, time.time()),
            )

    def __len__(self) -> int:
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
