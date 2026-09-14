"""The knowledge base is the agent's entire permitted vocabulary of claims.

Three sections, and the split is the safety property:
  Facts      — may be stated as-is
  Positions  — may be restated, never extended
  Boundaries — always escalate, no matter how confident the model feels
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT

KB_PATH = ROOT / "kb.md"

_SECTION_ALIASES = {
    "facts": "facts",
    "positions": "positions",
    "positions taken": "positions",
    "always escalate": "boundaries",
    "boundaries": "boundaries",
    "hard boundaries": "boundaries",
}

TEMPLATE = """\
# Identity
Kartik — NYU BTE. Placeholder until you swap in the real project.

# Facts
Things the agent may state out loud, exactly as written.
- Competitor analysis: in progress, draft due Wednesday
- Team: Kartik, [partner], [partner]
- Last deliverable: submitted 9/8, no revisions requested
- Meeting cadence: Mondays, 30 minutes

# Positions
Decisions already made. The agent may restate these; it may not extend them.
- Scope locked to two competitors, decided 9/5
- Using the free-tier stack, no paid hosting

# Always escalate
- Any new date, deliverable, or task assignment
- Any opinion, preference, or judgment call
- Anything about people, performance, or disagreement
- Anything not literally written above
"""


@dataclass
class KB:
    identity: str = ""
    facts: list[str] = field(default_factory=list)
    positions: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)
    raw: str = ""

    def as_prompt(self) -> str:
        def block(title: str, items: list[str]) -> str:
            body = "\n".join(f"- {i}" for i in items) if items else "- (none)"
            return f"{title}:\n{body}"
        return "\n\n".join([
            f"Identity: {self.identity or '(unset)'}",
            block("FACTS (may state verbatim)", self.facts),
            block("POSITIONS (may restate, never extend)", self.positions),
            block("ALWAYS ESCALATE", self.boundaries),
        ])

    def token_estimate(self) -> int:
        return len(self.raw) // 4

    def warnings(self) -> list[str]:
        w = []
        if self.token_estimate() > 3000:
            w.append("Knowledge base is over 3k tokens. Trim it — long context "
                     "makes the router hedge and slows every turn.")
        if not self.facts:
            w.append("No facts listed, so the agent will escalate everything.")
        if not self.boundaries:
            w.append("No escalation rules listed. Add them or the agent will "
                     "improvise on questions it should hand back to you.")
        return w


def parse(text: str) -> KB:
    kb = KB(raw=text)
    current: str | None = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            head = s.lstrip("#").strip().lower()
            if head == "identity":
                current = "identity"
            else:
                current = _SECTION_ALIASES.get(head)
            continue
        if not s:
            continue
        if current == "identity":
            kb.identity = (kb.identity + " " + s).strip()
        elif current and s.startswith(("-", "*")):
            item = re.sub(r"^[-*]\s*", "", s)
            if item:
                getattr(kb, current).append(item)
    return kb


def load(path: Path | None = None) -> KB:
    p = path or KB_PATH
    if not p.exists():
        p.write_text(TEMPLATE)
    return parse(p.read_text())


def save(text: str, path: Path | None = None) -> KB:
    p = path or KB_PATH
    p.write_text(text)
    return parse(text)
