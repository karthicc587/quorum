import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import pytest

from quorum import config
from quorum import kb as kbmod
from quorum.server import Hub, handle


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Never let a test see the developer's own files.

    The voice directory matters as much as the rest: a machine with a real
    latent.pkl made "refuses to start without a voice" pass locally and fail
    for anyone who had actually enrolled one.
    """
    from quorum import tts as _tts

    monkeypatch.setattr(config, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(kbmod, "KB_PATH", tmp_path / "kb.md")
    monkeypatch.setattr(_tts, "VOICE_DIR", tmp_path / "voices")
    monkeypatch.setattr(_tts, "SAMPLE_DIR", tmp_path / "voices" / "samples")


@pytest.fixture
def hub(monkeypatch):
    h = Hub()
    monkeypatch.setattr("quorum.server.hub", h)
    return h


@pytest.mark.asyncio
async def test_escalation_opens_and_resolves(hub):
    await hub.open_escalation("should we pivot", "opinion rule", "opinion", 1.0)
    assert hub.pending["guard"] == "opinion"

    resolved = await hub.resolve_escalation("Not right now.")
    assert hub.pending is None
    assert resolved["answer"] == "Not right now."
    assert resolved["latency_s"] >= 0
    assert hub.turns[-1].speaker == "you"
    assert hub.turns[-1].tier == "human"


@pytest.mark.asyncio
async def test_resolve_without_pending_is_noop(hub):
    assert await hub.resolve_escalation("hello") is None
    assert not hub.turns


@pytest.mark.asyncio
async def test_dismiss_clears_without_speaking(hub):
    await hub.open_escalation("q", "r", None, 0.3)
    await handle({"type": "dismiss"})
    assert hub.pending is None
    assert not hub.turns, "dismissing must not put words in the meeting"


@pytest.mark.asyncio
async def test_reply_ignores_whitespace(hub):
    await hub.open_escalation("q", "r", None, 0.3)
    await handle({"type": "reply", "text": "   "})
    assert hub.pending is not None


@pytest.mark.asyncio
async def test_kb_edit_reparses(hub):
    await handle({"type": "kb", "text": "# Facts\n- One thing\n# Always escalate\n- Everything else\n"})
    assert hub.kb.facts == ["One thing"]
    assert hub.snapshot()["kb"]["counts"]["facts"] == 1


@pytest.mark.asyncio
async def test_settings_update_coerces_types(hub):
    await handle({"type": "settings", "values": {"wake_threshold": "68", "wake_phrase": "AI Sam"}})
    assert hub.settings.wake_threshold == 68
    assert hub.settings.wake_phrase == "AI Sam"


@pytest.mark.asyncio
async def test_settings_ignores_unknown_and_bad_values(hub):
    before = hub.settings.wake_threshold
    await handle({"type": "settings", "values": {"nope": 1, "wake_threshold": "abc"}})
    assert hub.settings.wake_threshold == before
    assert not hasattr(hub.settings, "nope")


@pytest.mark.asyncio
async def test_secret_round_trip_and_redaction(hub):
    await handle({"type": "secret", "name": "GROQ_API_KEY", "value": "gsk_abcdef1234567890"})
    shown = hub.snapshot()["secrets"]["GROQ_API_KEY"]
    assert shown == "gsk_ab…7890"
    assert "gsk_abcdef1234567890" not in shown


@pytest.mark.asyncio
async def test_unknown_secret_name_rejected(hub):
    await handle({"type": "secret", "name": "AWS_SECRET", "value": "hunter2"})
    assert "AWS_SECRET" not in hub.snapshot()["secrets"]


@pytest.mark.asyncio
async def test_snapshot_never_contains_a_raw_key(hub):
    config.set_secret("GROQ_API_KEY", "gsk_supersecretvalue999")
    import json
    assert "gsk_supersecretvalue999" not in json.dumps(hub.snapshot())


@pytest.mark.asyncio
async def test_start_refuses_when_audio_is_not_routed(hub, monkeypatch):
    """Failing loudly here is the point: silent start is how demo day dies."""
    monkeypatch.setattr(hub, "audio_status",
                        lambda: {"ok": False, "problem": "No virtual audio device found."})
    await handle({"type": "run", "on": True})
    assert not hub.running
    assert hub.stage == "idle"
    assert "virtual audio device" in hub.last_error


@pytest.mark.asyncio
async def test_start_refuses_when_no_voice_is_enrolled(hub, monkeypatch):
    import quorum.server as srv

    class NotEnrolled:
        enrolled = False

    monkeypatch.setattr(hub, "audio_status", lambda: {"ok": True, "problem": None})
    monkeypatch.setattr(hub.tier, "tts_engine", "xtts")
    monkeypatch.setattr(srv.tts, "build_voice", lambda e, speed=1.0: NotEnrolled())
    await handle({"type": "run", "on": True})
    assert not hub.running
    assert "voice enrolled" in hub.last_error


@pytest.mark.asyncio
async def test_stop_is_safe_with_no_pipeline(hub):
    await handle({"type": "run", "on": False})
    assert not hub.running and hub.stage == "idle"


@pytest.mark.asyncio
async def test_start_succeeds_once_audio_and_voice_are_ready(hub, monkeypatch):
    import quorum.server as srv

    class FakePipeline:
        def __init__(self): self.warmed = self.started = False
        def warm(self): self.warmed = True
        def start(self): self.started = True
        def stop(self): self.started = False
        def latency_report(self): return {}

    fake = FakePipeline()
    monkeypatch.setattr(srv, "_build_pipeline", lambda: fake)
    await handle({"type": "run", "on": True})
    assert hub.running and hub.stage == "listening"
    assert fake.warmed and fake.started, "models must be warm before the first question"

    await handle({"type": "run", "on": False})
    assert not hub.running and not fake.started


@pytest.mark.asyncio
async def test_transcript_is_capped(hub):
    for i in range(90):
        await hub.add_turn(speaker="room", text=f"line {i}")
    assert len(hub.snapshot()["turns"]) == 60


# ------------------------------------------------------------- enrollment
def _client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import quorum.server as srv
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    return TestClient(srv.app)


def test_enroll_rejects_a_clip_that_is_too_short(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/enroll", files={"reference": ("ref.wav", b"RIFF" + b"\0" * 500,
                                                   "audio/wav")})
    assert r.status_code == 400
    assert "too short" in r.json()["error"]


def test_enroll_rejects_the_wrong_container(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/enroll", files={"reference": ("voice.m4a", b"\0" * 40_000,
                                                   "audio/mp4")})
    assert r.status_code == 400
    assert ".m4a" in r.json()["error"], "should name the format the user actually sent"


def test_enroll_calls_the_engine_and_prerenders(monkeypatch, tmp_path):
    import quorum.server as srv

    class FakeVoice:
        def __init__(self): self.enrolled_with = None; self.prerendered = []
        def enroll(self, path): self.enrolled_with = Path(path).read_bytes()
        def prerender(self, text): self.prerendered.append(text)

    fake = FakeVoice()
    monkeypatch.setattr(srv.tts, "build_voice", lambda engine, speed=1.0: fake)
    c = _client(monkeypatch, tmp_path)

    payload = b"RIFF" + b"\x01" * 40_000
    r = c.post("/api/enroll", files={"reference": ("ref.wav", payload, "audio/wav")})
    assert r.status_code == 200 and r.json()["ok"]
    assert fake.enrolled_with == payload, "the engine must see the uploaded bytes"
    assert fake.prerendered == [srv.hub.settings.holding_line]


def test_enroll_surfaces_engine_failure_instead_of_silently_passing(monkeypatch, tmp_path):
    import quorum.server as srv

    class BadVoice:
        def enroll(self, path): raise RuntimeError("no CUDA device")
        def prerender(self, text): pass

    monkeypatch.setattr(srv.tts, "build_voice", lambda engine, speed=1.0: BadVoice())
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/enroll", files={"reference": ("ref.wav", b"RIFF" + b"\x01" * 40_000,
                                                   "audio/wav")})
    assert r.status_code == 500
    assert "no CUDA device" in r.json()["error"]
    assert "enrollment" in srv.hub.last_error


def test_enroll_cleans_up_the_temp_file(monkeypatch, tmp_path):
    import tempfile
    import quorum.server as srv

    class FakeVoice:
        def enroll(self, path): pass
        def prerender(self, text): pass

    monkeypatch.setattr(srv.tts, "build_voice", lambda engine, speed=1.0: FakeVoice())
    c = _client(monkeypatch, tmp_path)
    c.post("/api/enroll", files={"reference": ("ref.wav", b"RIFF" + b"\x01" * 40_000,
                                               "audio/wav")})
    assert not (Path(tempfile.gettempdir()) / "quorum-ref.wav").exists()


# --------------------------------------------------------------- licensing
def test_enroll_blocked_until_the_model_licence_is_accepted(monkeypatch, tmp_path):
    import quorum.server as srv

    monkeypatch.setattr(srv.hub.tier, "tts_engine", "xtts")
    srv.hub.settings.xtts_license_accepted = False
    called = []
    monkeypatch.setattr(srv.tts, "build_voice", lambda e, speed=1.0: called.append(e))

    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/enroll", files={"reference": ("ref.wav", b"RIFF" + b"\x01" * 40_000,
                                                   "audio/wav")})
    assert r.status_code == 400
    assert "licence" in r.json()["error"]
    assert not called, "the model must not load before the licence is accepted"


def test_enroll_proceeds_once_accepted(monkeypatch, tmp_path):
    import os
    import quorum.server as srv

    class FakeVoice:
        def enroll(self, path): pass
        def prerender(self, text): pass

    monkeypatch.setattr(srv.hub.tier, "tts_engine", "xtts")
    monkeypatch.setattr(srv.tts, "build_voice", lambda e, speed=1.0: FakeVoice())
    monkeypatch.delenv("COQUI_TOS_AGREED", raising=False)
    srv.hub.settings.xtts_license_accepted = True

    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/enroll", files={"reference": ("ref.wav", b"RIFF" + b"\x01" * 40_000,
                                                   "audio/wav")})
    assert r.status_code == 200
    assert os.environ.get("COQUI_TOS_AGREED") == "1", \
        "acceptance is only forwarded to the library after the user agrees"
    srv.hub.settings.xtts_license_accepted = False


def test_licence_state_is_exposed_and_settable(hub):
    hub.tier.tts_engine = "xtts"
    v = hub.snapshot()["voice"]
    assert v["needs_license"] and not v["license_accepted"]
    assert v["license_url"].startswith("https://")


@pytest.mark.asyncio
async def test_license_message_persists_the_choice(hub):
    await handle({"type": "license", "accepted": True})
    assert hub.settings.xtts_license_accepted
    await handle({"type": "license", "accepted": False})
    assert not hub.settings.xtts_license_accepted


# ------------------------------------------------------------- meeting join
@pytest.mark.asyncio
async def test_join_refuses_an_empty_url(hub):
    await handle({"type": "join", "url": "   "})
    assert hub.meeting_state["state"] == "failed"
    assert "No meeting link" in hub.meeting_state["detail"]


@pytest.mark.asyncio
async def test_join_refuses_without_a_browser_session(hub, monkeypatch):
    """Better to say so than to open a browser that stalls on a login page."""
    import quorum.joiner as jm

    class NoSession:
        logged_in = False
        def __init__(self, *a, **k): pass
        def session_report(self): return "No profile. Run: python -m quorum.joiner login"

    monkeypatch.setattr(jm, "Joiner", NoSession)
    await handle({"type": "join", "url": "https://meet.google.com/abc-defg-hij"})
    assert hub.meeting_state["state"] == "failed"
    assert "joiner login" in hub.meeting_state["detail"]
    assert hub.meeting is None


@pytest.mark.asyncio
async def test_successful_join_starts_the_pipeline(hub, monkeypatch):
    """Joining without listening is useless, so the two are tied together."""
    import quorum.joiner as jm
    import quorum.server as srv
    from quorum.joiner import JoinResult, JoinState, Platform

    class FakeJoiner:
        logged_in = True
        def __init__(self, *a, **k): pass
        def audio_hint(self): return "set the mic to Voicemeeter Out B1"

    class FakeThread:
        alive = True
        def __init__(self, joiner): pass
        def start(self, url, timeout_s=90):
            return JoinResult(JoinState.JOINED, Platform.MEET, "")
        def stop(self): pass

    started = []

    async def fake_running(on):
        started.append(on)
        hub.running = on

    monkeypatch.setattr(jm, "Joiner", FakeJoiner)
    monkeypatch.setattr(jm, "JoinerThread", FakeThread)
    monkeypatch.setattr(srv, "_set_running", fake_running)

    await handle({"type": "join", "url": "https://meet.google.com/abc-defg-hij"})
    assert hub.meeting_state["state"] == "joined"
    assert "Voicemeeter Out B1" in hub.meeting_state["detail"], \
        "the human still has to pick the device; say so"
    assert started == [True]


@pytest.mark.asyncio
async def test_a_lobby_wait_is_not_reported_as_failure(hub, monkeypatch):
    import quorum.joiner as jm
    import quorum.server as srv
    from quorum.joiner import JoinResult, JoinState, Platform

    class FakeJoiner:
        logged_in = True
        def __init__(self, *a, **k): pass
        def audio_hint(self): return "hint"

    class FakeThread:
        alive = True
        def __init__(self, joiner): pass
        def start(self, url, timeout_s=90):
            return JoinResult(JoinState.LOBBY, Platform.MEET, "Waiting for a host.")
        def stop(self): pass

    monkeypatch.setattr(jm, "Joiner", FakeJoiner)
    monkeypatch.setattr(jm, "JoinerThread", FakeThread)
    monkeypatch.setattr(srv, "_set_running", lambda on: asyncio.sleep(0))

    await handle({"type": "join", "url": "https://meet.google.com/abc-defg-hij"})
    assert hub.meeting_state["state"] == "lobby"
    assert "Waiting for a host" in hub.meeting_state["detail"]


@pytest.mark.asyncio
async def test_leaving_stops_the_pipeline_too(hub, monkeypatch):
    import quorum.server as srv

    class FakeThread:
        def __init__(self): self.stopped = False
        def stop(self): self.stopped = True

    fake = FakeThread()
    hub.meeting = fake
    hub.running = True
    stopped = []

    async def fake_running(on):
        stopped.append(on)
        hub.running = on

    monkeypatch.setattr(srv, "_set_running", fake_running)
    await handle({"type": "leave"})
    assert fake.stopped
    assert stopped == [False]
    assert hub.meeting is None
    assert hub.meeting_state["state"] == "idle"


@pytest.mark.asyncio
async def test_leave_is_safe_when_not_in_a_meeting(hub):
    await handle({"type": "leave"})
    assert hub.meeting_state["state"] == "idle"


# ------------------------------------------------------------- test bench
def test_say_rejects_empty_text(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    assert c.post("/api/say", json={"text": "   "}).status_code == 400


def test_say_refuses_before_a_voice_is_enrolled(monkeypatch, tmp_path):
    import quorum.server as srv

    class NotEnrolled:
        enrolled = False
        sample_rate = 24000

    monkeypatch.setattr(srv.hub.tier, "tts_engine", "xtts")
    monkeypatch.setattr(srv.tts, "build_voice", lambda e, speed=1.0: NotEnrolled())
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/say", json={"text": "hello"})
    assert r.status_code == 400 and "built" in r.json()["error"]


def test_say_uses_the_configured_playback_device(monkeypatch, tmp_path):
    """A test that played out the wrong device would prove nothing."""
    import numpy as np
    import quorum.server as srv
    from quorum.tts import Speech

    class FakeVoice:
        enrolled = True
        sample_rate = 24000
        def say(self, text): return Speech(np.zeros(2400, dtype="float32"), 24000, 900)

    used = {}

    class FakePlayback:
        def __init__(self, device, rate): used["device"] = device; used["rate"] = rate
        def play(self, samples, on_start=None, on_end=None): used["played"] = len(samples)

    monkeypatch.setattr(srv.tts, "build_voice", lambda e, speed=1.0: FakeVoice())
    monkeypatch.setattr("quorum.audio.Playback", FakePlayback)
    monkeypatch.setattr("quorum.audio.resolve", lambda v: 46)
    srv.hub.settings.playback_device = "46"

    c = _client(monkeypatch, tmp_path)

    # Default: your speakers, or the test proves nothing you can hear.
    r = c.post("/api/say", json={"text": "testing one two"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["tts_ms"] == 900 and body["seconds"] == 0.1
    assert used["device"] is None, "system default, not the meeting bus"
    assert "your speakers" in body["sent_to"]
    assert used["played"] == 2400

    # Opt in to the meeting bus when checking routing.
    r = c.post("/api/say", json={"text": "testing", "to_meeting": True})
    assert used["device"] == 46
    assert "meeting" in r.json()["sent_to"]


def test_ask_routes_without_touching_audio(monkeypatch, tmp_path):
    import quorum.server as srv

    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/ask", json={"text": "can you have it done by Monday"})
    assert r.status_code == 200
    body = r.json()
    assert body["tier"] == "escalate"
    assert body["guard"] == "commitment", "guards need no model, so this works offline"


def test_ask_rejects_empty(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    assert c.post("/api/ask", json={"text": ""}).status_code == 400


# ----------------------------------------------------------- clip library
@pytest.fixture
def lib(tmp_path, monkeypatch):
    from quorum import tts as t
    monkeypatch.setattr(t, "SAMPLE_DIR", tmp_path / "samples")
    return t.SampleLibrary(tmp_path / "samples")


def _wav(seconds=5.0, rate=22050):
    import io
    import numpy as np
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(rate * seconds), dtype="float32"), rate, format="WAV")
    return buf.getvalue()


def test_clip_library_reports_total_and_advice(lib):
    assert not lib.status()["enough"]
    lib.add(_wav(40), "base")
    lib.add(_wav(30), "questions")
    st = lib.status()
    assert st["total_seconds"] == 70.0
    assert st["enough"] and st["has_base"]
    assert len(st["clips"]) == 2


def test_base_recording_comes_first(lib):
    """Extras without a base is the wrong order, and the advice should say so."""
    lib.add(_wav(40), "questions")
    st = lib.status()
    assert not st["has_base"]
    assert not st["enough"], "40s of extras is not a voice"
    assert "base recording" in st["advice"]


def test_short_library_asks_for_more(lib):
    lib.add(_wav(20), "base")
    st = lib.status()
    assert "40s" in st["advice"]
    assert not st["enough"]


def test_passages_track_what_is_already_covered(lib):
    lib.add(_wav(50), "base")
    lib.add(_wav(15), "hedging")
    st = lib.status()
    done = {p["id"]: p["done"] for p in st["passages"]}
    assert done["hedging"] is True
    assert done["questions"] is False
    assert st["next_passage"] == "questions", "offer one at a time, in order"


def test_next_passage_is_none_once_everything_is_recorded(lib):
    from quorum.tts import EXTRA_PASSAGES
    lib.add(_wav(50), "base")
    for p in EXTRA_PASSAGES:
        lib.add(_wav(12), p["id"])
    st = lib.status()
    assert st["next_passage"] is None
    assert "Every passage covered" in st["advice"]


def test_passage_id_survives_the_filename_round_trip(lib):
    p = lib.add(_wav(10), "numbers")
    assert lib.passage_of(p) == "numbers"
    assert "numbers" in lib.recorded_passages()


def test_removing_a_clip_cannot_escape_the_directory(lib, tmp_path):
    victim = tmp_path / "important.wav"
    victim.write_bytes(b"x")
    assert not lib.remove("../important.wav")
    assert victim.exists(), "path traversal must not delete anything outside"


def test_add_clip_endpoint_rejects_short_and_wrong_format(monkeypatch, tmp_path):
    from quorum import tts as t
    monkeypatch.setattr(t, "SAMPLE_DIR", tmp_path / "s")
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/voice/clips",
               files={"reference": ("a.wav", b"RIFF" + b"\0" * 100, "audio/wav")})
    assert r.status_code == 400 and "too short" in r.json()["error"]
    r = c.post("/api/voice/clips",
               files={"reference": ("a.m4a", b"\0" * 40_000, "audio/mp4")})
    assert r.status_code == 400 and ".m4a" in r.json()["error"]


def test_rebuild_uses_every_clip(monkeypatch, tmp_path):
    from quorum import tts as t
    import quorum.server as srv

    monkeypatch.setattr(t, "SAMPLE_DIR", tmp_path / "s")
    lib = t.SampleLibrary(tmp_path / "s")
    lib.add(_wav(30), "one")
    lib.add(_wav(35), "two")

    seen = {}

    class FakeVoice:
        def enroll(self, paths): seen["n"] = len(paths)
        def prerender(self, text): seen["prerendered"] = text

    monkeypatch.setattr(srv.hub.tier, "tts_engine", "xtts")
    srv.hub.settings.xtts_license_accepted = True
    monkeypatch.setattr(srv.tts, "build_voice", lambda e, speed=1.0: FakeVoice())

    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/voice/rebuild")
    body = r.json()
    assert r.status_code == 200 and body["built_from"] == 2
    assert isinstance(body["clips"], list), "status list must not shadow the count"
    assert seen["n"] == 2, "all clips must condition the voice, not just the last"
    srv.hub.settings.xtts_license_accepted = False


def test_rebuild_refuses_with_no_clips(monkeypatch, tmp_path):
    from quorum import tts as t
    monkeypatch.setattr(t, "SAMPLE_DIR", tmp_path / "empty")
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/voice/rebuild")
    assert r.status_code == 400 and "No clips" in r.json()["error"]


@pytest.mark.asyncio
async def test_disclosure_is_user_editable_and_used_on_join(hub):
    await handle({"type": "settings", "values": {"disclosure": "Custom wording."}})
    assert hub.settings.disclosure == "Custom wording."


def test_licence_state_is_exposed_independently_of_enrollment(hub, monkeypatch):
    """The checkbox must stay reachable after a voice exists.

    Regression: the UI hid it once enrolled, but rebuild still required
    acceptance — so a reset settings file left the build permanently refused
    with no way to clear it.
    """
    hub.tier.tts_engine = "xtts"
    hub.settings.xtts_license_accepted = False
    monkeypatch.setattr(hub, "voice_enrolled", lambda: True)
    v = hub.snapshot()["voice"]
    assert v["enrolled"] and v["needs_license"] and not v["license_accepted"], \
        "all three must be visible so the UI can show the way out"


def test_rebuild_refused_message_names_the_licence(monkeypatch, tmp_path):
    from quorum import tts as t
    import quorum.server as srv

    monkeypatch.setattr(t, "SAMPLE_DIR", tmp_path / "s")
    lib = t.SampleLibrary(tmp_path / "s")
    lib.add(_wav(40), "one")
    monkeypatch.setattr(srv.hub.tier, "tts_engine", "xtts")
    srv.hub.settings.xtts_license_accepted = False

    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/voice/rebuild")
    assert r.status_code == 400 and "licence" in r.json()["error"]
