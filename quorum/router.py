"""Decide whether the agent may answer, or must hand the question back.

Two layers, deliberately:

  1. Deterministic guards. Regex over the utterance. These fire before the
     model is consulted and cannot be argued out of. A language model asked
     "are you sure?" will eventually say yes, so anything that would create a
     commitment is escalated by rule rather than by judgment.

  2. Model classification, for everything the guards let through.

The asymmetry that drives the tuning: a wrong answer invents a commitment in
Kartik's name. A wrong escalation costs three seconds of meeting time.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from .kb import KB


class Tier(str, Enum):
    ANSWER = "answer"
    ESCALATE = "escalate"


@dataclass
class Decision:
    tier: Tier
    confidence: float
    reason: str
    draft: str
    guard: str | None = None        # deterministic rule that forced this
    latency_ms: int = 0
    raw: str = ""
    trace: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# layer 1 — deterministic guards
# ---------------------------------------------------------------------------
# Ordered most-specific first. Categories overlap by nature ("can you have it
# by Monday" is both a commitment and a date), and the first match wins, so
# the broad commitment catch-all sits last. Only the label is affected —
# every one of these escalates either way.
_GUARDS: list[tuple[str, re.Pattern]] = [
    ("opinion", re.compile(
        r"\b(do you think|what do you think|your (take|opinion|view|thoughts)|"
        r"should we|should i|would you rather|prefer|feel about|"
        r"in your view|agree|disagree|better idea|recommend|"
        r"going to be any good|worth it)\b", re.I)),
    ("interpersonal", re.compile(
        r"\b(everyone else|the team think|performance|pulling (his|her|their) weight|"
        r"blame|fault|behind schedule because|conflict|complain|frustrat|"
        r"upset|between us|off the record)\b", re.I)),
    ("resource", re.compile(
        r"\b(budget|cost|spend|hire|headcount|pay|paid|salary|contract|invoice|"
        r"legal|liabilit)\b", re.I)),
    ("commitment", re.compile(
        r"\b(can you|could you|will you|would you|are you able|"
        r"by when|deadline|deliver|"
        r"send me|get me|have it|finish|ship|commit|promise|"
        r"sign off|approve|agree to)\b", re.I)),
    ("scheduling", re.compile(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"tomorrow|tonight|next week|this week|end of (day|week)|eod|eow|"
        r"\d{1,2}\s*(am|pm)|\d{1,2}/\d{1,2})\b", re.I)),
]


def guard_check(utterance: str) -> str | None:
    for name, pat in _GUARDS:
        if pat.search(utterance):
            return name
    return None


# ---------------------------------------------------------------------------
# layer 2 — model
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a meeting delegate speaking on behalf of Kartik. You speak in the \
first person, aloud, in one or two short sentences. No preamble, no lists.

You may ONLY assert what appears in the knowledge base below. You may not \
infer, extrapolate, soften, hedge into, or combine entries into a new claim. \
Restating a POSITION is allowed. Extending one is not.

Reply with strict JSON and nothing else:
{"tier":"answer"|"escalate","confidence":0.0-1.0,"reason":"<8 words max",\
"draft":"<what to say aloud>"}

Set tier to "escalate" if ANY of these hold:
- answering needs a fact that is not literally in the knowledge base
- the question asks for an opinion, preference, or judgment
- answering would commit Kartik to a date, deliverable, or resource
- the question concerns people, performance, or disagreement
- you are not sure

When tier is "escalate", leave draft as an empty string.
Confidence is your probability that an "answer" would be both correct and \
within bounds. Be strict: when torn, go lower.

KNOWLEDGE BASE
{kb}
"""

USER_PROMPT = """\
Recent transcript:
{transcript}

Addressed to you: "{utterance}"
"""


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class Router:
    def __init__(self, llm: LLM, kb: KB, threshold: float = 0.70):
        self.llm = llm
        self.kb = kb
        self.threshold = threshold

    def decide(self, utterance: str, transcript: list[str] | None = None) -> Decision:
        t0 = time.perf_counter()
        trace: list[str] = []

        guard = guard_check(utterance)
        if guard:
            trace.append(f"guard:{guard}")
            return Decision(
                tier=Tier.ESCALATE, confidence=1.0,
                reason=f"{guard} rule", draft="", guard=guard,
                latency_ms=int((time.perf_counter() - t0) * 1000), trace=trace,
            )
        trace.append("guard:clear")

        system = SYSTEM_PROMPT.replace("{kb}", self.kb.as_prompt())
        user = USER_PROMPT.format(
            transcript="\n".join(transcript or []) or "(nothing yet)",
            utterance=utterance,
        )

        try:
            raw = self.llm.complete(system, user)
        except Exception as e:
            trace.append(f"llm-error:{type(e).__name__}")
            return Decision(
                Tier.ESCALATE, 0.0, "model unavailable", "",
                latency_ms=int((time.perf_counter() - t0) * 1000), trace=trace,
            )

        d = _parse(raw)
        d.trace = trace + [f"model:{d.tier.value}@{d.confidence:.2f}"]

        if d.tier is Tier.ANSWER and d.confidence < self.threshold:
            d.tier, d.draft = Tier.ESCALATE, ""
            d.reason = f"low confidence ({d.confidence:.2f})"
            d.trace.append("gate:below-threshold")

        if d.tier is Tier.ANSWER and not d.draft.strip():
            d.tier, d.reason = Tier.ESCALATE, "empty draft"
            d.trace.append("gate:empty-draft")

        d.latency_ms = int((time.perf_counter() - t0) * 1000)
        return d


def _parse(raw: str) -> Decision:
    """Models wrap JSON in fences, prose, or both. Recover what we can."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return Decision(Tier.ESCALATE, 0.0, "unparseable model output", "", raw=raw)
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Decision(Tier.ESCALATE, 0.0, "malformed json", "", raw=raw)

    tier = Tier.ANSWER if str(obj.get("tier", "")).lower() == "answer" else Tier.ESCALATE
    try:
        conf = max(0.0, min(1.0, float(obj.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    return Decision(
        tier=tier,
        confidence=conf,
        reason=str(obj.get("reason", ""))[:64],
        draft=str(obj.get("draft", "")).strip(),
        raw=raw,
    )
