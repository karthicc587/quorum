import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quorum.gate import Gate, Reject, State
from quorum.wake import EchoGuard


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, ms):
        self.t += ms / 1000


@pytest.fixture
def gate():
    c = Clock()
    g = Gate(tail_ms=500, now=c)
    g.clock = c
    return g


def test_idle_rejects_everything(gate):
    assert gate.accepts_transcript("AI Kartik, status?") is Reject.NOT_RUNNING


def test_running_accepts_normal_speech(gate):
    gate.start()
    assert gate.accepts_transcript("AI Kartik, status?") is None


def test_transcripts_dropped_while_speaking(gate):
    gate.start()
    gate.on_speech_start()
    assert gate.accepts_transcript("Yes, that's underway.") is Reject.SELF_SPEAKING


def test_tail_blocks_then_expires(gate):
    gate.start()
    gate.on_speech_start()
    gate.on_speech_end()
    assert gate.accepts_transcript("something") is Reject.IN_TAIL
    gate.clock.advance(499)
    assert gate.accepts_transcript("something") is Reject.IN_TAIL
    gate.clock.advance(2)
    assert gate.accepts_transcript("something") is None


def test_short_fragments_dropped(gate):
    gate.start()
    assert gate.accepts_transcript("um") is Reject.TOO_SHORT
    assert gate.accepts_transcript("  ") is Reject.TOO_SHORT


def test_wake_ignored_while_busy(gate):
    gate.start()
    assert gate.on_wake() is True
    assert gate.on_wake() is False, "already mid-turn"


def test_no_barge_in(gate):
    gate.start()
    gate.on_speech_start()
    assert gate.can_speak() is False


def test_full_answer_turn_returns_to_listening(gate):
    gate.start()
    gate.on_wake(); gate.on_route(); gate.on_speech_start(); gate.on_speech_end()
    assert gate.state is State.LISTENING
    gate.clock.advance(600)
    assert gate.on_wake() is True, "gate must be reusable for the next question"


def test_escalation_lands_in_waiting(gate):
    gate.start()
    gate.on_wake(); gate.on_route()
    gate.on_escalate()
    gate.on_speech_start(); gate.on_speech_end()   # holding line
    assert gate.state is State.WAITING


def test_human_reply_clears_waiting(gate):
    gate.start()
    gate.on_wake(); gate.on_route(); gate.on_escalate()
    gate.on_speech_start(); gate.on_speech_end()
    assert gate.on_human_reply() is True
    gate.on_speech_start(); gate.on_speech_end()
    assert gate.state is State.LISTENING


def test_human_reply_without_escalation_is_rejected(gate):
    gate.start()
    assert gate.on_human_reply() is False


def test_dismiss_returns_to_listening(gate):
    gate.start()
    gate.on_wake(); gate.on_route(); gate.on_escalate()
    gate.on_speech_start(); gate.on_speech_end()
    gate.on_dismiss()
    assert gate.state is State.LISTENING


def test_abort_never_strands_the_gate(gate):
    for setup in (
        lambda: (gate.on_wake(),),
        lambda: (gate.on_wake(), gate.on_route()),
        lambda: (gate.on_wake(), gate.on_route(), gate.on_speech_start()),
        lambda: (gate.on_wake(), gate.on_route(), gate.on_escalate()),
    ):
        gate.stop(); gate.start(); setup()
        gate.abort()
        assert gate.state is State.LISTENING
        gate.clock.advance(600)
        assert gate.on_wake() is True


def test_abort_while_idle_stays_idle(gate):
    gate.abort()
    assert gate.state is State.IDLE


# ------------------------------------------------------- the actual failure
def test_agent_does_not_reply_to_its_own_voice(gate):
    """The loop this whole module exists to prevent.

    Agent answers; Meet echoes that audio back with jitter-buffer delay long
    enough to clear the tail; STT transcribes it; it contains the wake phrase
    because the room said it. Without the echo guard the agent answers itself.
    """
    echo = EchoGuard()
    spoken = "Yes, that's underway. The draft is due Wednesday."

    gate.start()
    gate.on_wake(); gate.on_route()
    gate.on_speech_start()
    echo.remember(spoken)
    gate.on_speech_end()

    gate.clock.advance(900)                       # tail long gone
    late = "yes thats underway the draft is due wednesday"

    assert gate.accepts_transcript(late) is None, "tail cannot catch this"
    assert echo.is_echo(late), "echo guard must"


def test_echo_guard_lets_a_real_follow_up_through(gate):
    echo = EchoGuard()
    echo.remember("Yes, that's underway. The draft is due Wednesday.")
    gate.start()
    follow_up = "AI Kartik, who else is on the team?"
    assert gate.accepts_transcript(follow_up) is None
    assert not echo.is_echo(follow_up)
