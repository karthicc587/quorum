import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quorum.evaluate import _fmt as format_report
from quorum.evaluate import escalation, latency, report, wake
from quorum.telemetry import TurnLog, TurnRecord


# ------------------------------------------------------------------ telemetry
def test_turn_log_round_trip(tmp_path):
    log = TurnLog("s1", tmp_path)
    log.write(TurnRecord(heard="AI Kartik, status?", decision="answer",
                         confidence=0.9, stt_ms=200, router_ms=500, tts_ms=1100))
    log.write(TurnRecord(heard="can you do Monday", decision="escalate",
                         guard="commitment"))

    recs = TurnLog.read(log.path)
    assert len(recs) == 2
    assert recs[0].responsive_ms == 1800
    assert recs[1].guard == "commitment"


def test_truncated_final_line_is_tolerated(tmp_path):
    log = TurnLog("s2", tmp_path)
    log.write(TurnRecord(heard="fine"))
    with log.path.open("a") as f:
        f.write('{"heard": "cut off mid-wr')
    assert len(TurnLog.read(log.path)) == 1, "a crash mid-write must not lose the file"


def test_read_all_spans_sessions(tmp_path):
    TurnLog("a", tmp_path).write(TurnRecord(heard="one"))
    TurnLog("b", tmp_path).write(TurnRecord(heard="two"))
    assert len(TurnLog.read_all(tmp_path)) == 2


# ------------------------------------------------------------------- evaluate
def _labelled():
    return [
        # correct answers
        TurnRecord(question="who is on the team", decision="answer", label="answer",
                   wake_fired=True, wake_score=95, stt_ms=200, router_ms=600, tts_ms=1200),
        TurnRecord(question="is the draft in", decision="answer", label="answer",
                   wake_fired=True, wake_score=91, stt_ms=220, router_ms=580, tts_ms=1150),
        # correct escalations
        TurnRecord(question="can you do Monday", decision="escalate", label="escalate",
                   guard="commitment", wake_fired=True, wake_score=88,
                   stt_ms=210, router_ms=3, tts_ms=400, tts_cached=True,
                   escalated_to_human=True, human_latency_s=7.2),
        TurnRecord(question="should we pivot", decision="escalate", label="escalate",
                   guard="opinion", wake_fired=True, wake_score=93,
                   stt_ms=205, router_ms=2, tts_ms=390, tts_cached=True,
                   escalated_to_human=True, human_latency_s=12.0),
        # the costly error
        TurnRecord(question="what did the analysis conclude", decision="answer",
                   label="escalate", confidence=0.81, spoken="It found two viable rivals.",
                   wake_fired=True, wake_score=90, stt_ms=230, router_ms=700, tts_ms=1300),
        # the cheap error
        TurnRecord(question="how often do you meet", decision="escalate",
                   label="answer", guard="scheduling", wake_fired=True, wake_score=87,
                   stt_ms=215, router_ms=4, tts_ms=395),
        # unaddressed room chatter, correctly ignored
        TurnRecord(heard="anyway the weather is awful", wake_fired=False, wake_score=41),
        # a false wake
        TurnRecord(heard="the artifacts are checked in", wake_fired=True, wake_score=74),
    ]


def test_escalation_matrix_separates_the_two_error_types():
    e = escalation(_labelled())
    assert e["n"] == 6
    assert e["matrix"]["should escalate / answered"] == 1
    assert e["matrix"]["should answer / escalated"] == 1
    assert e["false_answer_rate"] == pytest.approx(1 / 3, abs=0.01)
    assert e["false_escalation_rate"] == pytest.approx(1 / 3, abs=0.01)
    assert e["accuracy"] == pytest.approx(4 / 6, abs=0.01)


def test_slipped_answers_are_listed_verbatim():
    e = escalation(_labelled())
    assert len(e["slipped_through"]) == 1
    s = e["slipped_through"][0]
    assert s["question"] == "what did the analysis conclude"
    assert s["said"] == "It found two viable rivals."
    assert s["confidence"] == 0.81


def test_guard_share_is_reported():
    e = escalation(_labelled())
    assert e["escalations_by_guard"] == {"commitment": 1, "opinion": 1, "scheduling": 1}
    assert 0 < e["guard_share"] <= 1


def test_unlabelled_log_says_so_instead_of_inventing_numbers():
    recs = [TurnRecord(question="q", decision="answer")]
    assert "note" in escalation(recs)
    assert "note" in wake(recs)


def test_wake_metrics():
    w = wake(_labelled())
    assert w["false_positive"] == 1, "the artifacts line"
    assert w["false_negative"] == 0
    assert w["precision"] == pytest.approx(6 / 7, abs=0.01)
    assert w["recall"] == 1.0


def test_latency_percentiles_and_holding_line():
    lat = latency(_labelled())
    assert lat["total"]["n"] == 6
    assert lat["stt"]["p50"] > 0
    assert lat["holding_line"]["cached_fraction"] == pytest.approx(0.67, abs=0.01)
    assert lat["human_reply_s"]["n"] == 2


def test_failures_counts_abandoned_escalations():
    recs = _labelled() + [
        TurnRecord(question="q", decision="escalate", escalated_to_human=True,
                   abandoned=True),
        TurnRecord(question="q", error="router: groq down"),
    ]
    f = report(recs)["failures"]
    assert f["abandoned_escalations"] == 1
    assert f["errors_by_stage"] == {"router": 1}


def test_report_is_complete():
    r = report(_labelled())
    assert set(r) == {"escalation", "modes", "wake", "latency_ms", "failures"}


def test_empty_log_does_not_crash():
    r = report([])
    assert r["latency_ms"] == {}
    assert "note" in r["escalation"]


# --------------------------------------------------------------- autonomy
def _modes():
    return [
        TurnRecord(question="who is on the team", decision="answer", mode="ask",
                   confidence=0.95, stt_ms=200, router_ms=600, tts_ms=1100),
        TurnRecord(question="can you do Monday", decision="escalate", mode="ask",
                   guard="commitment", stt_ms=210, router_ms=3, tts_ms=400),
        TurnRecord(question="what did it conclude", decision="answer",
                   mode="improvise", improvised=True, confidence=0.35,
                   spoken="My sense is it favoured the incumbent, but I'd check.",
                   stt_ms=205, router_ms=900, tts_ms=1200),
        TurnRecord(question="how many pages", decision="answer",
                   mode="improvise", improvised=True, confidence=0.8,
                   spoken="It's a short one, a few pages.",
                   stt_ms=200, router_ms=850, tts_ms=1150),
        TurnRecord(question="can you do Monday", decision="escalate",
                   mode="improvise", guard="commitment",
                   stt_ms=210, router_ms=3, tts_ms=400),
    ]


def test_modes_are_never_pooled():
    m = escalation(_modes())          # unlabelled, so no matrix
    by = report(_modes())["modes"]
    assert set(by) == {"ask", "improvise"}
    assert by["ask"]["improvised"] == 0
    assert by["improvise"]["improvised"] == 2


def test_guarded_questions_still_escalate_in_improvise_mode():
    by = report(_modes())["modes"]
    assert by["improvise"]["escalated"] == 1, "the commitment guard must survive"


def test_ungrounded_improvisations_are_counted_and_listed():
    by = report(_modes())["modes"]
    assert by["improvise"]["ungrounded"] == 1, "confidence 0.35 is ungrounded"
    said = [x["said"] for x in by["improvise"]["lines_invented"]]
    assert any("favoured the incumbent" in x for x in said)


def test_mode_table_renders():
    out = format_report(report(_modes()))
    assert "BY MODE" in out and "improvise" in out
    assert "INVENTED" in out
