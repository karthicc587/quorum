"""Wake-phrase detection and self-echo suppression.

Whisper does not transcribe a name the same way twice. "AI Kartik" comes back
as "Hey Cartik", "A.I. Kartek", "I heart it" if the audio is bad. Exact
matching fails constantly, so everything here is fuzzy.
"""
from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass

from rapidfuzz import fuzz

_FILLER = re.compile(r"\b(um+|uh+|er+|hmm+|like|you know)\b", re.I)
_PUNCT = re.compile(r"[^\w\s]")

_QUESTION_CUES = (
    "who", "what", "when", "where", "why", "how", "which", "whose",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "will", "would", "should", "have", "has", "any", "got",
)


def normalize(text: str) -> str:
    t = text.lower()
    t = t.replace(".", " ").replace("-", " ")
    t = _FILLER.sub(" ", t)
    t = _PUNCT.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class WakeHit:
    matched: bool
    score: int
    utterance: str
    is_question: bool


def looks_like_question(text: str) -> bool:
    """Whisper drops question marks constantly, so check the shape too."""
    s = text.strip()
    if s.endswith("?"):
        return True
    words = normalize(s).split()
    if not words:
        return False
    return words[0] in _QUESTION_CUES or any(
        w in _QUESTION_CUES[:8] for w in words[:4]
    )


def strip_wake(text: str, phrase: str) -> str:
    """Remove the wake phrase so the router sees only the actual question."""
    norm_words = normalize(phrase).split()
    words = text.split()
    n = len(norm_words)
    best_i, best_score = None, 0
    for i in range(max(1, len(words) - n + 1)):
        window = " ".join(words[i:i + n])
        sc = fuzz.ratio(normalize(window), normalize(phrase))
        if sc > best_score:
            best_i, best_score = i, sc
    if best_i is not None and best_score >= 70:
        words = words[:best_i] + words[best_i + n:]
    out = " ".join(words).strip()
    return re.sub(r"^[,\-–—:\s]+", "", out)


def score(text: str, phrase: str) -> int:
    """How strongly this transcript looks like it addressed the agent.

    Two signals, take the max:

      window  — de-space the phrase and every 1–3 word window of the text,
                then compare. Survives Whisper splitting or joining the
                initialism ("A I Kartik", "AIKartik").
      token   — compare the bare surname against each single word. Whisper
                drops the "AI" entirely about a third of the time, and this
                catches "Cartik" / "Kartek" / "Karthik".

    partial_ratio over the whole utterance is deliberately not used: it
    scores "artificial" and "artifacts" at 80 against "kartik" and would fire
    the agent mid-conversation.
    """
    target = normalize(phrase).replace(" ", "")
    words = normalize(text).split()
    if not words:
        return 0

    window = 0
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            window = max(window, fuzz.ratio(target, "".join(words[i:i + n])))

    token = 0
    surname = normalize(phrase).split()[-1]
    if len(surname) >= 5:
        token = max(fuzz.ratio(surname, w) for w in words)

    return int(max(window, token))


def detect(text: str, phrase: str, threshold: int = 72) -> WakeHit:
    sc = score(text, phrase)
    if sc < threshold:
        return WakeHit(False, sc, "", False)
    utterance = strip_wake(text, phrase)
    return WakeHit(True, sc, utterance, looks_like_question(utterance or text))


class EchoGuard:
    """Second line of defence against the agent hearing itself.

    The first line is muting STT during playback. That fails when Meet's
    jitter buffer delays our own audio past the tail window, so we also drop
    any transcript that closely matches something we recently said.
    """

    def __init__(self, threshold: int = 85, ttl_s: float = 12.0):
        self.threshold = threshold
        self.ttl_s = ttl_s
        self._spoken: deque[tuple[float, str]] = deque(maxlen=24)

    def remember(self, text: str) -> None:
        self._spoken.append((time.monotonic(), normalize(text)))

    def _live(self) -> list[str]:
        now = time.monotonic()
        return [t for ts, t in self._spoken if now - ts <= self.ttl_s]

    def is_echo(self, text: str) -> bool:
        cand = normalize(text)
        if len(cand) < 8:
            return False
        return any(
            fuzz.partial_ratio(cand, prev) >= self.threshold
            for prev in self._live()
        )
