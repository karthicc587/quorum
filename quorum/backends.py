"""LLM backends. Groq's free tier by default, local Ollama when wifi dies."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import Settings, get_secret

TIMEOUT_S = 12


def _post(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.loads(r.read().decode())


class GroqLLM:
    """Free tier, no card. Key lives in .env or the Settings screen."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, settings: Settings):
        self.model = settings.groq_model

    @staticmethod
    def available() -> bool:
        return bool(get_secret("GROQ_API_KEY"))

    def complete(self, system: str, user: str) -> str:
        key = get_secret("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY missing. Add it under Settings.")
        data = _post(self.URL, {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.2,
            "max_tokens": 220,
            "response_format": {"type": "json_object"},
        }, {"Authorization": f"Bearer {key}"})
        return data["choices"][0]["message"]["content"]


class OllamaLLM:
    """Offline fallback. Slower, and noticeably worse at refusing to extrapolate."""

    URL = "http://127.0.0.1:11434/api/chat"

    def __init__(self, settings: Settings):
        self.model = settings.ollama_model

    @staticmethod
    def available() -> bool:
        try:
            with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2):
                return True
        except Exception:
            return False

    def complete(self, system: str, user: str) -> str:
        data = _post(self.URL, {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 220},
        }, {})
        return data["message"]["content"]


class ChainLLM:
    """Try each backend in order. Router escalates if all of them fail."""

    def __init__(self, *backends):
        self.backends = backends
        self.last_used: str | None = None

    def complete(self, system: str, user: str) -> str:
        errors = []
        for b in self.backends:
            try:
                out = b.complete(system, user)
                self.last_used = type(b).__name__
                return out
            except Exception as e:
                errors.append(f"{type(b).__name__}: {e}")
        raise RuntimeError("; ".join(errors) or "no backends configured")


def build_llm(settings: Settings, prefer: str = "groq") -> ChainLLM:
    groq, ollama = GroqLLM(settings), OllamaLLM(settings)
    order = (groq, ollama) if prefer == "groq" else (ollama, groq)
    return ChainLLM(*order)
