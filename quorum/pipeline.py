"""The worker that ties everything together.

Runs on its own thread so model inference never blocks the dashboard's event
loop. Talks to the Hub through a small callback surface rather than importing
it, which keeps the whole pipeline testable with fakes.

Timing is recorded per stage on every turn, because the eval needs p50/p95
per stage and reconstructing that from logs afterwards is miserable.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

from .audio import Capture, Playback
from .config import Settings, Tier
from .gate import Gate, State
from .kb import KB
from .router import Router, Tier as Decision
from .stt import Segmenter, Transcriber, Utterance
from .telemetry import TurnLog, TurnRecord
from .tts import CachedVoice
from .wake import EchoGuard, detect


@dataclass
class Stages:
    """Milliseconds per stage for one turn."""
    audio: int = 0
    stt: int = 0
    router: int = 0
    tts: int = 0

    @property
    def responsive_ms(self) -> int:
        """What the room actually waits: everything after speech ends."""
        return self.stt + self.router + self.tts

    def as_dict(self) -> dict:
        return {"stt": self.stt, "router": self.router, "tts": self.tts}


@dataclass
class Hooks:
    """Everything the pipeline tells the outside world about."""
    on_stage: callable = lambda s: None
    on_heard: callable = lambda text, meta: None
    on_answer: callable = lambda text, decision, stages: None
    on_escalate: callable = lambda question, decision: None
    on_error: callable = lambda where, exc: None


class Pipeline:
    def __init__(self, tier: Tier, settings: Settings, kb: KB, router: Router,
                 voice: CachedVoice, hooks: Hooks,
                 capture_device=None, playback_device=None,
                 log: TurnLog | None = None):
        self.tier, self.settings, self.kb = tier, settings, kb
        self.router, self.voice, self.hooks = router, voice, hooks

        self.gate = Gate(tail_ms=settings.tts_tail_ms, now=time.monotonic)
        self.echo = EchoGuard(threshold=settings.echo_suppress_threshold)
        self.segmenter = Segmenter(close_ms=settings.vad_close_ms)
        self.stt = Transcriber(tier.stt_model, tier.stt_device, tier.stt_compute)
        self.capture = Capture(capture_device)
        self.playback = Playback(playback_device, voice.sample_rate)

        self.log = log or TurnLog()
        self.transcript: list[str] = []
        self.turns: list[Stages] = []
        self._thread: threading.Thread | None = None
        self._pending_human: str | None = None
        self._human_ready = threading.Event()

    # -- lifecycle ---------------------------------------------------------
    def warm(self):
        """Load models and pre-render the holding line before anyone joins.

        Doing this lazily means the first question of the meeting takes eight
        seconds instead of two. Always warm before the call.
        """
        self.hooks.on_stage("warming")
        self.stt.load()
        if self.voice.enrolled:
            self.voice.prerender(self.settings.holding_line)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.gate.start()
        self.capture.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.hooks.on_stage(State.LISTENING.value)

    def stop(self):
        self.gate.stop()
        self.capture.stop()
        self.playback.interrupt()
        self._human_ready.set()
        self.hooks.on_stage(State.IDLE.value)

    # -- the human's typed reply ------------------------------------------
    def submit_human_reply(self, text: str):
        self._pending_human = text
        self._human_ready.set()

    def dismiss(self):
        self._pending_human = None
        self._human_ready.set()

    # -- main loop ---------------------------------------------------------
    def _run(self):
        try:
            for block in self.capture.blocks():
                if self.gate.state is State.IDLE:
                    break
                # Cheap check first: skip VAD entirely while we are talking.
                if self.gate.state is State.SPEAKING:
                    continue
                for audio, started in self.segmenter.feed(block):
                    self._handle_utterance(audio, started)
        except Exception as e:
            self.hooks.on_error("capture", e)
            self.gate.abort()

    def _handle_utterance(self, audio, started_at: float):
        try:
            utt = self.stt.transcribe(audio, started_at)
        except Exception as e:
            return self.hooks.on_error("stt", e)
        if not utt.text:
            return

        reject = self.gate.accepts_transcript(utt.text)
        if reject:
            return
        if self.echo.is_echo(utt.text):
            return

        self.transcript.append(utt.text)
        del self.transcript[:-self.settings.transcript_window]

        hit = detect(utt.text, self.settings.wake_phrase, self.settings.wake_threshold)
        self.hooks.on_heard(utt.text, {"wake_score": hit.score, "stt_ms": utt.stt_ms})

        rec = TurnRecord(heard=utt.text, wake_score=hit.score,
                         wake_fired=hit.matched, stt_ms=utt.stt_ms,
                         audio_ms=utt.audio_ms)
        if not hit.matched or not self.gate.on_wake():
            self._record(rec)
            return

        stages = Stages(audio=utt.audio_ms, stt=utt.stt_ms)
        question = hit.utterance or utt.text
        rec.question = question

        self.hooks.on_stage(State.DECIDING.value)
        self.gate.on_route()
        try:
            decision = self.router.decide(question, self.transcript[:-1])
        except Exception as e:
            self.gate.abort()
            rec.error = f"router: {e}"
            self._record(rec)
            return self.hooks.on_error("router", e)
        stages.router = decision.latency_ms
        rec.router_ms = decision.latency_ms
        rec.decision = decision.tier.value
        rec.guard = decision.guard
        rec.confidence = decision.confidence
        rec.reason = decision.reason

        if decision.tier is Decision.ANSWER:
            self._speak(decision.draft, stages, rec)
            rec.spoken = decision.draft
            self.hooks.on_answer(decision.draft, decision, stages)
            self.turns.append(stages)
            self._record(rec)
            return

        # Escalation: holding line first so the meeting does not stall.
        self.gate.on_escalate()
        self._speak(self.settings.holding_line, stages, rec)
        rec.spoken = self.settings.holding_line
        rec.escalated_to_human = True
        self.hooks.on_escalate(question, decision)
        self.turns.append(stages)
        self._await_human(rec)
        self._record(rec)

    def _record(self, rec: TurnRecord):
        try:
            self.log.write(rec)
        except Exception as e:
            self.hooks.on_error("log", e)

    def _await_human(self, rec: TurnRecord | None = None):
        """Block until the human answers, dismisses, or the window closes.

        The wait is bounded on purpose. If nobody is watching the dashboard,
        an unanswered escalation would otherwise leave the agent deaf for the
        rest of the meeting — a worse failure than saying nothing.
        """
        self._human_ready.clear()
        self._pending_human = None
        t0 = time.time()
        answered = self._human_ready.wait(timeout=self.settings.escalation_timeout_s)
        if rec is not None:
            rec.human_latency_s = round(time.time() - t0, 1)
            rec.abandoned = not answered
        if not answered:
            self.hooks.on_error("escalation", TimeoutError(
                f"no reply within {self.settings.escalation_timeout_s}s"))
        text = self._pending_human
        if rec is not None:
            rec.human_reply = text
        if text and self.gate.on_human_reply():
            self._speak(text, Stages())
        else:
            self.gate.on_dismiss()
        self.hooks.on_stage(self.gate.state.value)

    def _speak(self, text: str, stages: Stages, rec: TurnRecord | None = None):
        if not text.strip() or not self.gate.can_speak():
            return
        try:
            speech = self.voice.say(text)
        except Exception as e:
            self.gate.abort()
            if rec is not None:
                rec.error = f"tts: {e}"
                self._record(rec)
            return self.hooks.on_error("tts", e)
        stages.tts = speech.tts_ms
        if rec is not None:
            rec.tts_ms = speech.tts_ms
            rec.tts_cached = speech.cached
        self.echo.remember(text)
        self.hooks.on_stage(State.SPEAKING.value)
        try:
            self.playback.play(
                speech.samples,
                on_start=self.gate.on_speech_start,
                on_end=self.gate.on_speech_end,
            )
        except Exception as e:
            self.gate.abort()
            return self.hooks.on_error("playback", e)
        self.hooks.on_stage(self.gate.state.value)

    # -- eval --------------------------------------------------------------
    def latency_report(self) -> dict:
        if not self.turns:
            return {}
        def pct(vals, p):
            v = sorted(vals)
            return v[min(len(v) - 1, int(len(v) * p))]
        r = {}
        for stage in ("stt", "router", "tts"):
            vals = [getattr(t, stage) for t in self.turns]
            r[stage] = {"p50": pct(vals, .5), "p95": pct(vals, .95)}
        total = [t.responsive_ms for t in self.turns]
        r["total"] = {"p50": pct(total, .5), "p95": pct(total, .95), "n": len(total)}
        return r
