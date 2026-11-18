"""Central configuration, read from environment variables / .env."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
RESULTS_DIR = ROOT / "results"
CONFIG_DIR = ROOT / "config"


@dataclass(frozen=True)
class Settings:
    provider: str = os.getenv("LLM_PROVIDER", "ollama")          # ollama | openai | gemini
    model: str = os.getenv("LLM_MODEL", "")                        # empty -> provider default
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    cache_path: str = os.getenv("LLM_CACHE_PATH", str(ROOT / ".cache" / "llm_cache.sqlite"))
    timeout: int = int(os.getenv("LLM_TIMEOUT", "600"))
    seed: int = int(os.getenv("SEED", "42"))


DEFAULT_MODELS = {
    "ollama": "qwen3.5:4b",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
}


def get_settings() -> Settings:
    return Settings()


def set_global_seed(seed: int = 42) -> None:
    """Seed every source of randomness we control (numpy, random)."""
    random.seed(seed)
    np.random.seed(seed)
