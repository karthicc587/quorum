"""LLM backends. Groq's free tier by default, local Ollama when wifi dies."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import Settings, get_secret

TIMEOUT_S = 12

# urllib announces itself as "Python-urllib/3.x", which Cloudflare's bot filter
# blocks outright — the request is refused with "error code: 1010" before it
# ever reaches the API, and the status looks like an auth failure. Identifying
# the application honestly is enough; this is not an attempt to look like a
# browser.
USER_AGENT = "quorum.ai/0.1 (meeting delegate; +https://github.com/quorum-ai)"


_STATUS_HINTS = {
    401: "the API key was rejected — check it is current and not revoked",
    403: "the API key is not permitted to use this model",
    404: "no such model — check the model name in settings",
    429: "rate limited by the free tier; wait a moment",
}


def _post(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": USER_AGENT,
                 "Accept": "application/json",
                 **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode()
        except Exception:
            pass
        try:
            body = json.loads(raw).get("error", {}).get("message", "")
        except Exception:
            # Not JSON — an edge proxy answered, not the API. Keep the text.
            body = " ".join(raw.split())[:200]
        hint = _STATUS_HINTS.get(e.code, "")
        if "1010" in body or "error code: 1010" in body.lower():
            hint = ("blocked by Cloudflare before reaching the API, not an auth "
                    "problem — the HTTP client is being rejected on its "
                    "signature. Check the User-Agent header is being sent")
        raise RuntimeError(
            f"HTTP {e.code}{f' — {hint}' if hint else ''}"
            f"{f': {body}' if body else ''}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"could not reach {url.split('/')[2]}: {e.reason}") from e


# Tried in order when the configured model is gone. A retired name returns 403
# or 404 depending on whether it still exists for enterprise accounts, so both
# are treated as "try the next one".
GROQ_FALLBACKS = ("openai/gpt-oss-120b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b")
_MODEL_GONE = ("403", "404", "model_not_found", "does not exist",
               "not permitted", "decommission")


def _model_is_gone(err: str) -> bool:
    low = err.lower()
    return any(m in low for m in _MODEL_GONE)


def _json_mode_unsupported(err: str) -> bool:
    low = err.lower()
    return "400" in low and ("json" in low or "response_format" in low)


class GroqLLM:
    """Free tier, no card. Key lives in .env or the Settings screen."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, settings: Settings):
        self.model = settings.groq_model
        self.tried: list[str] = []
        self.strict_json = True

    @staticmethod
    def available() -> bool:
        return bool(get_secret("GROQ_API_KEY"))

    def _call(self, model: str, key: str, system: str, user: str,
              strict_json: bool = True) -> str:
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.2,
            # Reasoning models spend tokens before the answer, so the budget
            # has to cover both or the JSON arrives truncated.
            "max_tokens": 220 if strict_json else 700,
        }
        if strict_json:
            body["response_format"] = {"type": "json_object"}
        data = _post(self.URL, body, {"Authorization": f"Bearer {key}"})
        return data["choices"][0]["message"]["content"]

    def _try_model(self, model: str, key: str, system: str, user: str) -> str:
        """Prefer server-side JSON mode, but do not depend on it.

        Reasoning models emit thinking tokens before the answer, which Groq's
        json_object validator rejects with a 400 telling you to adjust the
        prompt. The router's parser already recovers JSON from fenced or
        prose-wrapped output, so dropping strict mode costs nothing.
        """
        try:
            return self._call(model, key, system, user, strict_json=True)
        except Exception as e:
            if not _json_mode_unsupported(str(e)):
                raise
            self.strict_json = False
            return self._call(model, key, system, user, strict_json=False)

    def complete(self, system: str, user: str) -> str:
        key = get_secret("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY missing. Add it under Settings.")

        candidates = [self.model] + [m for m in GROQ_FALLBACKS if m != self.model]
        last = ""
        for model in candidates:
            try:
                out = self._try_model(model, key, system, user)
                if model != self.model:
                    # Stick with what worked for the rest of the session rather
                    # than paying the failed call again on every question.
                    self.model = model
                self.tried.append(model)
                return out
            except Exception as e:
                last = str(e)
                if not _model_is_gone(last):
                    raise           # a real error: key, rate limit, network
        raise RuntimeError(
            f"No usable Groq model. Tried {', '.join(candidates)}. Last error: {last}. "
            "Groq retires model names periodically — check "
            "https://console.groq.com/docs/models and set groq_model in "
            ".quorum/settings.json"
        )


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
        }, {"User-Agent": USER_AGENT})
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
