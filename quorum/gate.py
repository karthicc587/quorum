"""The gate decides, moment to moment, whether audio reaching the machine is
worth transcribing and whether the agent is allowed to speak.

This is the piece that keeps the agent from talking to itself. On one machine
the TTS output goes into a virtual cable, into Meet, and back out of Meet's
mixed downstream — which is the same stream we are transcribing. Without a
gate the agent hears its own answer, treats it as a new utterance, and
replies to it.

Three defences, in order of reliability:

  1. Mute STT for as long as we are speaking, plus a tail. Cheap and catches
     almost everything.
  2. Refuse to start a second reply while one is in flight. Meet's jitter
     buffer can delay our audio past the tail, so the tail alone is not
     enough.
  3. Fuzzy-match incoming transcripts against what we recently said
     (EchoGuard in wake.py). Last line of defence.

No real clock and no audio here, so the whole thing is testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class State(str, Enum):
    IDLE = "idle"              # not running
    LISTENING = "listening"    # transcribing, waiting for the wake phrase
    HEARING = "hearing"        # wake phrase hit, utterance still being spoken
    DECIDING = "deciding"      # router is running
    SPEAKING = "speaking"      # TTS is playing into the meeting
    WAITING = "waiting"        # escalated, holding for a human reply


class Reject(str, Enum):
    NOT_RUNNING = "not running"
    SELF_SPEAKING = "we are speaking"
    IN_TAIL = "playback tail"
    ECHO = "matches our own speech"
    TOO_SHORT = "below minimum length"


@dataclass
class Gate:
    tail_ms: int = 500
    min_chars: int = 4
    now: Callable[[], float] = field(default=lambda: 0.0)

    state: State = State.IDLE
    _speech_ended_at: float | None = None
    _pending_reply: bool = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self.state = State.LISTENING

    def stop(self) -> None:
        self.state = State.IDLE
        self._speech_ended_at = None
        self._pending_reply = False

    # -- inbound audio -----------------------------------------------------
    def accepts_transcript(self, text: str) -> Reject | None:
        """None means the transcript may proceed to wake matching."""
        if self.state is State.IDLE:
            return Reject.NOT_RUNNING
        if self.state is State.SPEAKING:
            return Reject.SELF_SPEAKING
        if self._in_tail():
            return Reject.IN_TAIL
        if len(text.strip()) < self.min_chars:
            return Reject.TOO_SHORT
        return None

    def _in_tail(self) -> bool:
        if self._speech_ended_at is None:
            return False
        return (self.now() - self._speech_ended_at) * 1000 < self.tail_ms

    # -- turn progression --------------------------------------------------
    def on_wake(self) -> bool:
        """Wake phrase matched. False if we are busy and must ignore it."""
        if self.state is not State.LISTENING:
            return False
        self.state = State.HEARING
        return True

    def on_route(self) -> None:
        self.state = State.DECIDING

    def can_speak(self) -> bool:
        """Barge-in is deliberately not supported: one utterance at a time."""
        return self.state is not State.SPEAKING

    def on_speech_start(self) -> None:
        self.state = State.SPEAKING

    def on_speech_end(self) -> None:
        self._speech_ended_at = self.now()
        self.state = State.WAITING if self._pending_reply else State.LISTENING

    # -- escalation --------------------------------------------------------
    def on_escalate(self) -> None:
        """Set before the holding line plays, so we land in WAITING after."""
        self._pending_reply = True

    def on_human_reply(self) -> bool:
        """False if nothing was actually waiting."""
        if not self._pending_reply:
            return False
        self._pending_reply = False
        return True

    def on_dismiss(self) -> None:
        self._pending_reply = False
        if self.state is State.WAITING:
            self.state = State.LISTENING

    # -- recovery ----------------------------------------------------------
    def abort(self) -> None:
        """Something threw mid-turn. Never strand the gate in a busy state."""
        self._pending_reply = False
        self._speech_ended_at = self.now()
        if self.state is not State.IDLE:
            self.state = State.LISTENING
