"""Turns the turn log into the numbers the writeup needs.

Three results, in descending order of how interesting they are:

  1. Escalation confusion matrix against hand labels. The core claim is that
     bounded autonomy is safe, and this is the only thing that tests it.
  2. Wake precision and recall. Cheap to compute, and a false wake is the most
     visible failure in the room.
  3. Latency percentiles per stage. Table stakes, but p95 is what people
     remember, not p50.

The asymmetry is reported explicitly rather than folded into an F1, because
the two error types are not comparable: a false answer invents a commitment in
someone's name, a false escalation costs three seconds. Averaging them into
one number hides the whole design argument.

    python -m quorum.evaluate                 # every session
    python -m quorum.evaluate logs/x.jsonl    # one session
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .telemetry import TurnLog, TurnRecord


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return float(v[min(len(v) - 1, int(round(p * (len(v) - 1))))])


# ------------------------------------------------------------------ latency
def latency(records: list[TurnRecord]) -> dict:
    spoke = [r for r in records if r.decision and not r.error]
    if not spoke:
        return {}
    out = {}
    for stage in ("stt", "router", "tts"):
        vals = [getattr(r, f"{stage}_ms") for r in spoke]
        out[stage] = {"p50": percentile(vals, .5), "p95": percentile(vals, .95),
                      "max": max(vals)}
    total = [r.responsive_ms for r in spoke]
    out["total"] = {"p50": percentile(total, .5), "p95": percentile(total, .95),
                    "max": max(total), "n": len(total)}

    held = [r for r in spoke if r.decision == "escalate"]
    if held:
        h = [r.responsive_ms for r in held]
        out["holding_line"] = {"p50": percentile(h, .5), "p95": percentile(h, .95),
                               "cached_fraction": round(
                                   sum(r.tts_cached for r in held) / len(held), 2)}
    human = [r.human_latency_s for r in records if r.human_latency_s is not None]
    if human:
        out["human_reply_s"] = {"p50": percentile(human, .5),
                                "p95": percentile(human, .95), "n": len(human)}
    return out


# --------------------------------------------------------------------- wake
def wake(records: list[TurnRecord], addressed_field: str = "label") -> dict:
    """Needs a hand label saying whether the room was actually addressing it.

    Any record with a label counts as addressed; the wake decision is
    `wake_fired`.
    """
    labelled = [r for r in records if r.label]
    if not labelled:
        return {"note": "No hand labels yet. Fill in the label field to get these."}

    tp = sum(r.wake_fired for r in labelled)
    fn = sum(not r.wake_fired for r in labelled)
    fp = sum(r.wake_fired for r in records if not r.label)

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    fired = [r.wake_score for r in records if r.wake_fired]
    quiet = [r.wake_score for r in records if not r.wake_fired]
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "precision": round(prec, 3), "recall": round(rec, 3),
        "score_margin": (min(fired) - max(quiet)) if fired and quiet else None,
    }


# --------------------------------------------------------------- escalation
@dataclass
class Matrix:
    answered_correctly: int = 0     # should answer, did answer
    escalated_correctly: int = 0    # should escalate, did escalate
    false_answer: int = 0           # should escalate, answered anyway — the bad one
    false_escalation: int = 0       # should answer, escalated — the cheap one

    @property
    def n(self) -> int:
        return (self.answered_correctly + self.escalated_correctly
                + self.false_answer + self.false_escalation)

    @property
    def accuracy(self) -> float:
        return (self.answered_correctly + self.escalated_correctly) / self.n if self.n else 0.0

    @property
    def false_answer_rate(self) -> float:
        """Of everything that should have been escalated, how much slipped."""
        d = self.escalated_correctly + self.false_answer
        return self.false_answer / d if d else 0.0

    @property
    def false_escalation_rate(self) -> float:
        d = self.answered_correctly + self.false_escalation
        return self.false_escalation / d if d else 0.0


def escalation(records: list[TurnRecord]) -> dict:
    labelled = [r for r in records if r.label in ("answer", "escalate") and r.decision]
    if not labelled:
        return {"note": "No hand labels yet. Set label to 'answer' or "
                        "'escalate' on each turn to get the matrix."}

    m = Matrix()
    for r in labelled:
        if r.label == "answer":
            if r.decision == "answer":
                m.answered_correctly += 1
            else:
                m.false_escalation += 1
        else:
            if r.decision == "escalate":
                m.escalated_correctly += 1
            else:
                m.false_answer += 1

    by_guard: dict[str, int] = {}
    for r in labelled:
        if r.decision == "escalate" and r.guard:
            by_guard[r.guard] = by_guard.get(r.guard, 0) + 1

    slipped = [{"question": r.question, "said": r.spoken, "confidence": r.confidence}
               for r in labelled if r.label == "escalate" and r.decision == "answer"]

    return {
        "n": m.n,
        "accuracy": round(m.accuracy, 3),
        "false_answer": m.false_answer,
        "false_answer_rate": round(m.false_answer_rate, 3),
        "false_escalation": m.false_escalation,
        "false_escalation_rate": round(m.false_escalation_rate, 3),
        "matrix": {
            "should answer / answered": m.answered_correctly,
            "should answer / escalated": m.false_escalation,
            "should escalate / escalated": m.escalated_correctly,
            "should escalate / answered": m.false_answer,
        },
        "escalations_by_guard": dict(sorted(by_guard.items(), key=lambda x: -x[1])),
        "guard_share": round(
            sum(by_guard.values()) / max(1, m.escalated_correctly + m.false_escalation), 2),
        "slipped_through": slipped,
    }


# ------------------------------------------------------------------ failures
def failures(records: list[TurnRecord]) -> dict:
    tax: dict[str, int] = {}
    for r in records:
        if r.error:
            key = r.error.split(":")[0].strip()
            tax[key] = tax.get(key, 0) + 1
    return {
        "errors_by_stage": dict(sorted(tax.items(), key=lambda x: -x[1])),
        "abandoned_escalations": sum(r.abandoned for r in records),
        "turns": len(records),
    }


def by_mode(records: list[TurnRecord]) -> dict:
    """Deferential vs autonomous, side by side.

    The claim under test is that bounded autonomy is acceptable. That only
    means something against the alternative, so the two modes are never
    pooled: improvised answers are counted separately and the ungrounded ones
    are listed in full, because a confident invention in someone's name is
    the specific failure this project is about.
    """
    out = {}
    for mode in ("ask", "improvise"):
        rs = [r for r in records if r.mode == mode and r.decision]
        if not rs:
            continue
        improvised = [r for r in rs if r.improvised]
        spoke = [r for r in rs if r.decision == "answer"]
        low = [r for r in improvised if (r.confidence or 0) < 0.5]
        out[mode] = {
            "turns": len(rs),
            "spoke": len(spoke),
            "escalated": sum(r.decision == "escalate" for r in rs),
            "improvised": len(improvised),
            "improvised_share": round(len(improvised) / len(rs), 2),
            "ungrounded": len(low),
            "mean_confidence": round(
                sum(r.confidence or 0 for r in spoke) / len(spoke), 2) if spoke else None,
            "median_response_ms": percentile([r.responsive_ms for r in rs], .5),
            "lines_invented": [
                {"question": r.question, "said": r.spoken,
                 "confidence": r.confidence}
                for r in improvised
            ][:20],
        }
    return out


def report(records: list[TurnRecord]) -> dict:
    return {
        "escalation": escalation(records),
        "modes": by_mode(records),
        "wake": wake(records),
        "latency_ms": latency(records),
        "failures": failures(records),
    }


# -------------------------------------------------------------------- cli
def _fmt(report_: dict) -> str:
    lines = []
    e = report_["escalation"]
    lines.append("ESCALATION")
    if "note" in e:
        lines.append(f"  {e['note']}")
    else:
        lines.append(f"  n={e['n']}  accuracy={e['accuracy']}")
        lines.append(f"  false answers    {e['false_answer']:>3}  "
                     f"rate {e['false_answer_rate']}   <- the costly error")
        lines.append(f"  false escalations{e['false_escalation']:>3}  "
                     f"rate {e['false_escalation_rate']}   <- the cheap error")
        for k, v in e["matrix"].items():
            lines.append(f"    {k:<30} {v}")
        if e["escalations_by_guard"]:
            lines.append(f"  caught by guard rather than model: {e['guard_share']:.0%}")
            for g, c in e["escalations_by_guard"].items():
                lines.append(f"    {g:<16} {c}")
        for s in e["slipped_through"]:
            lines.append(f"  SLIPPED: {s['question']!r} -> {s['said']!r} "
                         f"(conf {s['confidence']})")

    m = report_.get("modes") or {}
    if m:
        lines.append("\nBY MODE")
        lines.append(f"  {'':<11}{'turns':>6}{'spoke':>7}{'escal':>7}"
                     f"{'improv':>8}{'ungrnd':>8}{'conf':>7}{'ms':>8}")
        for name, v in m.items():
            conf = f"{v['mean_confidence']:.2f}" if v["mean_confidence"] is not None else "—"
            lines.append(f"  {name:<11}{v['turns']:>6}{v['spoke']:>7}"
                         f"{v['escalated']:>7}{v['improvised']:>8}"
                         f"{v['ungrounded']:>8}{conf:>7}{v['median_response_ms']:>8.0f}")
        for name, v in m.items():
            for x in v["lines_invented"][:5]:
                lines.append(f"  INVENTED [{name}] {x['question']!r} -> "
                             f"{x['said']!r} (conf {x['confidence']})")

    w = report_["wake"]
    lines.append("\nWAKE PHRASE")
    lines.append(f"  {w['note']}" if "note" in w else
                 f"  precision={w['precision']}  recall={w['recall']}  "
                 f"fp={w['false_positive']}  fn={w['false_negative']}  "
                 f"margin={w['score_margin']}")

    lines.append("\nLATENCY (ms)")
    lat = report_["latency_ms"]
    if not lat:
        lines.append("  no completed turns")
    else:
        lines.append(f"  {'stage':<14}{'p50':>7}{'p95':>7}{'max':>7}")
        for k in ("stt", "router", "tts", "total"):
            if k in lat:
                v = lat[k]
                lines.append(f"  {k:<14}{v['p50']:>7.0f}{v['p95']:>7.0f}{v['max']:>7.0f}")
        if "holding_line" in lat:
            h = lat["holding_line"]
            lines.append(f"  holding line  p50 {h['p50']:.0f}  "
                         f"cached {h['cached_fraction']:.0%}")
        if "human_reply_s" in lat:
            hr = lat["human_reply_s"]
            lines.append(f"  human reply   p50 {hr['p50']:.1f}s  p95 {hr['p95']:.1f}s")

    f = report_["failures"]
    lines.append("\nFAILURES")
    lines.append(f"  turns={f['turns']}  abandoned escalations={f['abandoned_escalations']}")
    for k, v in f["errors_by_stage"].items():
        lines.append(f"    {k:<16} {v}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    recs = (TurnLog.read(Path(argv[0])) if argv else TurnLog.read_all())
    if not recs:
        print("No turn logs found. Run a meeting first.")
        return 1
    print(_fmt(report(recs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
