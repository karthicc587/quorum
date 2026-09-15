"""Pipeline tests with every hardware component faked.

Covers the turn flow, the self-reply loop, escalation blocking, and error
recovery. What it cannot cover: whether Whisper hears the room correctly and
whether the clone sounds like anyone. Those need the machine.
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from quorum import kb as kbmod
from quorum.config import Settings, TIERS
from quorum.gate import State
from quorum.pipeline import Hooks, Pipeline
from quorum.router import Router
from quorum.stt import Utterance
from quorum.telemetry import TurnLog
from quorum.tts import Speech


# ------------------------------------------------------------------- fakes
class FakeCapture:
    def __init__(self, n_blocks=3):
        self.n = n_blocks
        self.started = False

    def start(self): self.started = True
    def stop(self): self.started = False
    def blocks(self):
        for _ in range(self.n):
            yield np.zeros(512, dtype=np.float32)


class FakeSegmenter:
    """Emits one scripted utterance per block fed."""
    def __init__(self, script): self.script = list(script)
    def feed(self, block):
        if self.script:
            yield self.script.pop(0), time.time()


class FakeSTT:
    def __init__(self, texts): self.texts = list(texts); self.loaded = False
    def load(self): self.loaded = True
    def transcribe(self, audio, started_at):
        return Utterance(text=str(audio), started_at=started_at,
                         ended_at=time.time(), audio_ms=900, stt_ms=240)


class FakeVoice:
    sample_rate = 24000
    enrolled = True
    def __init__(self): self.said = []; self.prerendered = []; self.loaded = False
    def load(self): self.loaded = True
    def prerender(self, t): self.prerendered.append(t)
    def say(self, text):
        self.said.append(text)
        return Speech(np.zeros(120, dtype=np.float32), 24000, 1100)


class FakePlayback:
    def __init__(self): self.plays = []
    def play(self, samples, on_start=None, on_end=None):
        if on_start: on_start()
        self.plays.append(len(samples))
        if on_end: on_end()
    def interrupt(self): pass


class StubLLM:
    def __init__(self, payload): self.payload = payload
    def complete(self, s, u): return self.payload


class Recorder(Hooks):
    def __init__(self):
        self.stages, self.heard, self.answers, self.escalations, self.errors = [], [], [], [], []
        super().__init__(
            on_stage=self.stages.append,
            on_heard=lambda t, m: self.heard.append(t),
            on_answer=lambda t, d, s: self.answers.append((t, d, s)),
            on_escalate=lambda q, d: self.escalations.append((q, d)),
            on_error=lambda w, e: self.errors.append((w, e)),
        )


_LOGDIR = None


@pytest.fixture(autouse=True)
def isolate_logs(tmp_path):
    """Tests must never append to the real logs/ directory."""
    global _LOGDIR
    _LOGDIR = tmp_path / "logs"
    yield
    _LOGDIR = None


def build(utterances, llm_payload, kb_text=None, settings=None):
    kb = kbmod.parse(kb_text or kbmod.TEMPLATE)
    rec = Recorder()
    p = Pipeline(
        tier=TIERS["cpu"], settings=settings or Settings(), kb=kb,
        router=Router(StubLLM(llm_payload), kb),
        voice=FakeVoice(), hooks=rec,
        log=TurnLog(directory=_LOGDIR),
    )
    p.capture = FakeCapture(n_blocks=len(utterances))
    p.segmenter = FakeSegmenter(utterances)
    p.stt = FakeSTT(utterances)
    p.playback = FakePlayback()
    return p, rec


ANSWER = ('{"tier":"answer","confidence":0.94,"reason":"in kb",'
          '"draft":"Yes, that is underway and almost complete."}')


# ------------------------------------------------------------------- tests
def test_answer_turn_speaks_and_records_timing():
    p, rec = build(["AI Kartik, are you working on the competitor analysis"], ANSWER)
    p.gate.start(); p._run()

    assert p.voice.said == ["Yes, that is underway and almost complete."]
    assert len(p.playback.plays) == 1
    assert rec.answers and not rec.escalations and not rec.errors
    st = rec.answers[0][2]
    assert st.stt == 240 and st.tts == 1100 and st.responsive_ms > 1300
    assert p.gate.state is State.LISTENING


def test_non_addressed_speech_is_ignored():
    p, rec = build(["so anyway the deadline moved to Friday"], ANSWER)
    p.gate.start(); p._run()
    assert not p.voice.said
    assert rec.heard, "still transcribed for context"
    assert not rec.answers


def test_guarded_question_escalates_without_calling_the_model():
    p, rec = build(["AI Kartik, can you have it done by Monday"], ANSWER)
    p.gate.start()
    threading.Thread(target=p._run, daemon=True).start()
    time.sleep(0.3)

    assert p.voice.said == [Settings().holding_line]
    assert rec.escalations
    assert rec.escalations[0][1].guard == "commitment"
    assert p.gate.state is State.WAITING

    p.submit_human_reply("Wednesday is the earliest, realistically.")
    time.sleep(0.3)
    assert p.voice.said[-1] == "Wednesday is the earliest, realistically."
    assert p.gate.state is State.LISTENING


def test_dismiss_puts_no_words_in_the_meeting():
    p, rec = build(["AI Kartik, do you think we should pivot"], ANSWER)
    p.gate.start()
    threading.Thread(target=p._run, daemon=True).start()
    time.sleep(0.3)
    spoken_before = len(p.voice.said)

    p.dismiss()
    time.sleep(0.3)
    assert len(p.voice.said) == spoken_before, "dismiss must stay silent"
    assert p.gate.state is State.LISTENING


def test_agent_ignores_its_own_answer_echoed_back():
    """The loop the whole gate exists to prevent, end to end.

    The tail alone cannot catch this: Meet's jitter buffer can return our own
    audio well after playback finished, and the echoed segment carries the
    room's wake phrase along with it. The echo guard is what stops it, so the
    tail is set to zero here to make sure the guard is what is being tested.
    """
    answer = "Yes, that is underway and almost complete."
    echoed = f"AI Kartik, are you working on the competitor analysis {answer}"

    settings = Settings(tts_tail_ms=0)
    p, rec = build(["AI Kartik, are you working on the competitor analysis", echoed],
                   ANSWER, settings=settings)
    p.gate.start(); p._run()

    assert p.gate.accepts_transcript(echoed) is None, "tail must not be the thing catching it"
    assert p.voice.said == [answer], "must speak exactly once"


def test_without_the_echo_guard_the_agent_answers_itself():
    """Control for the test above: disable the guard, watch the loop happen."""
    answer = "Yes, that is underway and almost complete."
    echoed = f"AI Kartik, are you working on the competitor analysis {answer}"

    p, rec = build(["AI Kartik, are you working on the competitor analysis", echoed],
                   ANSWER, settings=Settings(tts_tail_ms=0))
    p.echo.is_echo = lambda text: False
    p.gate.start(); p._run()

    assert p.voice.said == [answer, answer], "the loop this module prevents"


def test_low_confidence_answer_becomes_an_escalation():
    p, rec = build(["AI Kartik, who is on the team"],
                   '{"tier":"answer","confidence":0.4,"reason":"unsure","draft":"Maybe three of us."}')
    p.gate.start()
    threading.Thread(target=p._run, daemon=True).start()
    time.sleep(0.3)
    assert p.voice.said == [Settings().holding_line]
    assert "Maybe three of us." not in p.voice.said
    p.dismiss(); time.sleep(0.2)


def test_router_failure_recovers_instead_of_stranding():
    class Boom:
        def decide(self, *a, **k): raise RuntimeError("groq down")
    p, rec = build(["AI Kartik, who is on the team"], ANSWER)
    p.router = Boom()
    p.gate.start(); p._run()
    assert rec.errors and rec.errors[0][0] == "router"
    assert p.gate.state is State.LISTENING, "must accept the next question"


def test_tts_failure_recovers():
    class BadVoice(FakeVoice):
        def say(self, text): raise RuntimeError("no latent")
    p, rec = build(["AI Kartik, who is on the team"], ANSWER)
    p.voice = BadVoice()
    p.gate.start(); p._run()
    assert rec.errors and rec.errors[0][0] == "tts"
    assert p.gate.state is State.LISTENING


def test_empty_transcription_is_skipped():
    p, rec = build([""], ANSWER)
    p.gate.start(); p._run()
    assert not rec.heard and not p.voice.said


def test_transcript_window_is_bounded():
    utts = [f"filler line number {i}" for i in range(20)]
    p, rec = build(utts, ANSWER)
    p.gate.start(); p._run()
    assert len(p.transcript) == Settings().transcript_window


def test_warm_loads_models_and_prerenders_holding_line():
    """The voice model must load during warm-up, not on the first question.

    Regression: warm() called prerender(), which returns early when the
    holding line is already cached — so the multi-second weight load fell on
    the first real question of the meeting.
    """
    p, rec = build([], ANSWER)
    p.stt = FakeSTT([])
    p.warm()
    assert p.stt.loaded
    assert p.voice.loaded, "weights must be resident before the meeting starts"
    assert Settings().holding_line in p.voice.prerendered


def test_warm_loads_the_voice_even_when_the_holding_line_is_cached():
    class CachedVoice(FakeVoice):
        def prerender(self, t): pass          # already on disk, does nothing

    p, rec = build([], ANSWER)
    p.stt = FakeSTT([])
    p.voice = CachedVoice()
    p.warm()
    assert p.voice.loaded, "a warm cache must not skip loading the model"


def test_latency_report_shape():
    p, rec = build(["AI Kartik, who is on the team"] , ANSWER)
    p.gate.start(); p._run()
    r = p.latency_report()
    assert r["total"]["n"] == 1
    assert set(r) == {"stt", "router", "tts", "total"}
    assert r["stt"]["p50"] == 240


def test_unanswered_escalation_releases_the_agent():
    """An abandoned escalation must not deafen the agent for the whole meeting."""
    p, rec = build(["AI Kartik, can you have it done by Monday"], ANSWER,
                   settings=Settings(escalation_timeout_s=1))
    p.gate.start()
    t = threading.Thread(target=p._run, daemon=True)
    t.start(); t.join(timeout=6)

    assert not t.is_alive(), "pipeline thread must not hang"
    assert any(w == "escalation" for w, _ in rec.errors)
    assert p.gate.state is State.LISTENING
    assert p.voice.said == [Settings().holding_line], "must not invent an answer"
