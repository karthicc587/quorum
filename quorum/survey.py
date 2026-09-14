"""The participant survey — the part that actually tests the hypothesis.

The technical metrics answer "does the pipeline work". They cannot answer
"is a disclosed delegate with hard limits acceptable to the people in the
room", which is the claim. Only the people in the room can answer that.

Design notes worth defending in the writeup:

  Paired pre/post. Attitudes to AI in meetings are all over the place before
  anyone has seen one, so a post-only measure tells you about your sample, not
  about your system. Six items are asked twice, word for word, and the shift
  is the finding.

  Reverse-scored items. A3, B2 and C2 are worded so that agreement is the
  negative response. Someone straight-lining down the column shows up as
  incoherent rather than enthusiastic, and `straight_lining` flags it.

  Escalation is asked about separately from the agent overall. The whole
  design argument is that handing questions back is a feature; if participants
  read it as the agent being useless, that is a finding and it should not be
  buried inside a general satisfaction score.

  Free text last, and only three of them. Long instruments filled in after a
  meeting get abandoned halfway.
"""
from __future__ import annotations

import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import ROOT

SURVEY_DIR = ROOT / "surveys"

LIKERT = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]


@dataclass
class Item:
    key: str
    text: str
    construct: str
    reverse: bool = False       # agreement is the negative response
    paired: bool = False        # asked identically before and after
    kind: str = "likert"        # likert | text | choice
    options: list[str] = field(default_factory=list)


# --------------------------------------------------------------------- pre
PRE = [
    Item("consent",
         "I understand this session involves an AI delegate attending on "
         "someone's behalf, and I agree to take part.",
         "consent", kind="choice", options=["Yes", "No"]),

    Item("A1", "I would be comfortable if a colleague sent an AI delegate to a "
               "meeting like this instead of attending.",
         "acceptability", paired=True),
    Item("A2", "An AI delegate could represent someone accurately in a routine "
               "meeting.",
         "trust", paired=True),
    Item("A3", "An AI delegate in a meeting would mostly get in the way.",
         "friction", reverse=True, paired=True),
    Item("A4", "I would trust what an AI delegate says about its owner's work.",
         "trust", paired=True),
    Item("A5", "It matters to me whether I am told a participant is an AI.",
         "disclosure", paired=True),
    Item("A6", "I would rather someone skip a meeting entirely than send an AI "
               "delegate.",
         "acceptability", reverse=True, paired=True),

    Item("A7", "How often do you attend meetings where you contribute little or "
               "nothing?",
         "context", kind="choice",
         options=["Never", "Rarely", "Sometimes", "Often", "Most of them"]),
]

# -------------------------------------------------------------------- post
POST = [
    # the six paired items, identical wording
    Item("A1", "I would be comfortable if a colleague sent an AI delegate to a "
               "meeting like this instead of attending.",
         "acceptability", paired=True),
    Item("A2", "An AI delegate could represent someone accurately in a routine "
               "meeting.",
         "trust", paired=True),
    Item("A3", "An AI delegate in a meeting would mostly get in the way.",
         "friction", reverse=True, paired=True),
    Item("A4", "I would trust what an AI delegate says about its owner's work.",
         "trust", paired=True),
    Item("A5", "It matters to me whether I am told a participant is an AI.",
         "disclosure", paired=True),
    Item("A6", "I would rather someone skip a meeting entirely than send an AI "
               "delegate.",
         "acceptability", reverse=True, paired=True),

    # escalation, asked on its own
    Item("B1", "When the agent said it would check with Kartik, that felt like "
               "the right call.",
         "escalation"),
    Item("B2", "Handing questions back to Kartik made the meeting drag.",
         "escalation", reverse=True),
    Item("B3", "I would rather the agent guess than hand a question back.",
         "escalation", reverse=True),
    Item("B4", "The agent stayed within what it should have been answering.",
         "bounds"),

    # disclosure salience
    Item("C1", "I was aware throughout that I was talking to an AI, not to "
               "Kartik.",
         "disclosure"),
    Item("C2", "At some point I forgot I was talking to an AI.",
         "disclosure", reverse=True),
    Item("C3", "Where did you first notice the agent was an AI?",
         "disclosure", kind="choice",
         options=["The name label", "The chat message", "How it sounded",
                  "How it answered", "The pauses", "Someone told me",
                  "I did not notice"]),

    # outcome
    Item("D1", "The meeting achieved what it needed to.", "outcome"),
    Item("D2", "Kartik's absence was handled acceptably.", "outcome"),

    # free text, kept short
    Item("E1", "What did the agent do that a person would not have?",
         "open", kind="text"),
    Item("E2", "Was there a moment it should have spoken but did not, or spoke "
               "when it should not have?",
         "open", kind="text"),
    Item("E3", "Would you want a colleague to use this? Why or why not?",
         "open", kind="text"),
]

PAIRED_KEYS = [i.key for i in PRE if i.paired]
ITEMS = {"pre": PRE, "post": POST}


def score(item: Item, raw: int) -> int:
    """Likert 1-5, reverse-scored so higher always means more favourable."""
    return (6 - raw) if item.reverse else raw


# ------------------------------------------------------------------ storage
@dataclass
class Response:
    participant: str
    phase: str                  # pre | post
    session: str = ""
    t: float = field(default_factory=time.time)
    answers: dict = field(default_factory=dict)

    @staticmethod
    def new_participant() -> str:
        """Anonymous but stable, so pre and post can be paired."""
        return uuid.uuid4().hex[:6]


class SurveyStore:
    def __init__(self, directory: Path | None = None):
        self.dir = directory or SURVEY_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "responses.jsonl"

    def write(self, r: Response) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(r)) + "\n")

    def read(self) -> list[Response]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                out.append(Response(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
        return out


# ------------------------------------------------------------------ scoring
def _by_key(phase: str) -> dict[str, Item]:
    return {i.key: i for i in ITEMS[phase]}


def straight_lining(r: Response) -> bool:
    """Same raw value on every Likert item, including the reversed ones.

    Not proof of inattention, but a reason to look at that response before
    including it.
    """
    items = _by_key(r.phase)
    vals = [v for k, v in r.answers.items()
            if k in items and items[k].kind == "likert" and isinstance(v, int)]
    return len(vals) >= 4 and len(set(vals)) == 1


def construct_means(responses: list[Response], phase: str) -> dict[str, float]:
    items = _by_key(phase)
    buckets: dict[str, list[int]] = {}
    for r in responses:
        if r.phase != phase:
            continue
        for k, v in r.answers.items():
            it = items.get(k)
            if it and it.kind == "likert" and isinstance(v, int):
                buckets.setdefault(it.construct, []).append(score(it, v))
    return {c: round(statistics.mean(v), 2) for c, v in sorted(buckets.items())}


def paired_shift(responses: list[Response]) -> dict:
    """Per-item before/after change for participants who answered both."""
    pre = {r.participant: r for r in responses if r.phase == "pre"}
    post = {r.participant: r for r in responses if r.phase == "post"}
    both = sorted(set(pre) & set(post))
    if not both:
        return {"n": 0, "note": "No participant has completed both halves yet."}

    items = _by_key("pre")
    out: dict[str, dict] = {}
    for key in PAIRED_KEYS:
        it = items[key]
        deltas, befores, afters = [], [], []
        for p in both:
            a, b = pre[p].answers.get(key), post[p].answers.get(key)
            if isinstance(a, int) and isinstance(b, int):
                befores.append(score(it, a))
                afters.append(score(it, b))
                deltas.append(score(it, b) - score(it, a))
        if deltas:
            out[key] = {
                "text": it.text,
                "construct": it.construct,
                "before": round(statistics.mean(befores), 2),
                "after": round(statistics.mean(afters), 2),
                "shift": round(statistics.mean(deltas), 2),
                "moved_up": sum(d > 0 for d in deltas),
                "moved_down": sum(d < 0 for d in deltas),
                "unchanged": sum(d == 0 for d in deltas),
                "n": len(deltas),
            }
    return {"n": len(both), "items": out}


def report(responses: list[Response]) -> dict:
    post = [r for r in responses if r.phase == "post"]
    items = _by_key("post")

    def mean_of(keys):
        vals = [score(items[k], v) for r in post for k, v in r.answers.items()
                if k in keys and k in items and isinstance(v, int)]
        return round(statistics.mean(vals), 2) if vals else None

    noticed: dict[str, int] = {}
    for r in post:
        c = r.answers.get("C3")
        if isinstance(c, str):
            noticed[c] = noticed.get(c, 0) + 1

    return {
        "n_pre": sum(r.phase == "pre" for r in responses),
        "n_post": len(post),
        "flagged_straight_lining": [r.participant for r in responses
                                    if straight_lining(r)],
        "shift": paired_shift(responses),
        "post_constructs": construct_means(responses, "post"),
        "escalation_accepted": mean_of({"B1", "B2", "B3"}),
        "stayed_in_bounds": mean_of({"B4"}),
        "disclosure_salient": mean_of({"C1", "C2"}),
        "first_noticed": dict(sorted(noticed.items(), key=lambda x: -x[1])),
        "open_responses": [
            {"participant": r.participant,
             **{k: r.answers[k] for k in ("E1", "E2", "E3") if r.answers.get(k)}}
            for r in post
            if any(r.answers.get(k) for k in ("E1", "E2", "E3"))
        ],
    }


def format_report(rep: dict) -> str:
    L = [f"PARTICIPANTS  pre={rep['n_pre']}  post={rep['n_post']}"]
    if rep["flagged_straight_lining"]:
        L.append(f"  flagged for straight-lining: "
                 f"{', '.join(rep['flagged_straight_lining'])}")

    sh = rep["shift"]
    L.append("\nATTITUDE SHIFT (higher is more favourable, 1-5)")
    if not sh.get("items"):
        L.append(f"  {sh.get('note', 'nothing paired yet')}")
    else:
        L.append(f"  paired participants: {sh['n']}")
        L.append(f"  {'item':<6}{'before':>8}{'after':>8}{'shift':>8}   up/down/same")
        for k, v in sh["items"].items():
            L.append(f"  {k:<6}{v['before']:>8.2f}{v['after']:>8.2f}"
                     f"{v['shift']:>+8.2f}   {v['moved_up']}/{v['moved_down']}/{v['unchanged']}")

    L.append("\nBOUNDED AUTONOMY")
    for label, key in (("escalation accepted", "escalation_accepted"),
                       ("stayed in bounds", "stayed_in_bounds"),
                       ("disclosure salient", "disclosure_salient")):
        v = rep[key]
        L.append(f"  {label:<22}{'—' if v is None else f'{v:.2f}'}")

    if rep["first_noticed"]:
        L.append("\nFIRST NOTICED IT WAS AN AI")
        for k, v in rep["first_noticed"].items():
            L.append(f"  {k:<22}{v}")

    if rep["open_responses"]:
        L.append(f"\nOPEN RESPONSES ({len(rep['open_responses'])})")
        for o in rep["open_responses"][:5]:
            for k in ("E1", "E2", "E3"):
                if o.get(k):
                    L.append(f"  [{o['participant']}] {k}: {o[k][:110]}")
    return "\n".join(L)


def as_markdown(phase: str) -> str:
    """Printable version, for running the survey on paper if the room is offline."""
    lines = [f"# quorum.ai — {phase}-session survey", ""]
    if phase == "pre":
        lines += ["Participant code: ______    (keep this; you will need it after)", ""]
    else:
        lines += ["Participant code: ______    (the same one as before)", ""]
    for i, item in enumerate(ITEMS[phase], 1):
        lines.append(f"**{i}. {item.text}**")
        if item.kind == "likert":
            lines.append("   " + "   ".join(f"[{n}] {l}" for n, l in enumerate(LIKERT, 1)))
        elif item.kind == "choice":
            lines.append("   " + "   ".join(f"[ ] {o}" for o in item.options))
        else:
            lines.append("   " + "_" * 60)
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "report"
    if arg in ("pre", "post"):
        print(as_markdown(arg))
    else:
        print(format_report(report(SurveyStore().read())))
