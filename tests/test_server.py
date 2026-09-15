import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quorum import config
from quorum import kb as kbmod
from quorum.server import Hub, handle


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Never let a test touch the real kb.md, .env, or settings.json."""
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(kbmod, "KB_PATH", tmp_path / "kb.md")


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
    monkeypatch.setattr(hub, "audio_status", lambda: {"ok": True, "problem": None})
    monkeypatch.setattr(hub.tier, "tts_engine", "xtts")
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
    monkeypatch.setattr(srv.tts, "build_voice", lambda engine: fake)
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

    monkeypatch.setattr(srv.tts, "build_voice", lambda engine: BadVoice())
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

    monkeypatch.setattr(srv.tts, "build_voice", lambda engine: FakeVoice())
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
    monkeypatch.setattr(srv.tts, "build_voice", lambda e: called.append(e))

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
    monkeypatch.setattr(srv.tts, "build_voice", lambda e: FakeVoice())
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
