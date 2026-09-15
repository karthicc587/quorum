"""Dashboard server. Owns the run state; the audio pipeline pushes into it."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import kb as kbmod
from . import platforms, tts
from .config import ROOT, Settings, detect_tier, get_secret, redact, set_secret

STATIC = ROOT / "static"
SECRET_KEYS = ["GROQ_API_KEY"]


@dataclass
class Turn:
    id: str
    t: float
    speaker: str            # "room" | "agent" | "you"
    text: str
    tier: str | None = None
    improvised: bool = False
    confidence: float | None = None
    guard: str | None = None
    reason: str | None = None
    stages: dict = field(default_factory=dict)   # stage -> ms


class Hub:
    """Single source of truth. Every client sees the same state."""

    def __init__(self):
        self.settings = Settings.load()
        self.tier = detect_tier()
        self.kb = kbmod.load()
        self.turns: list[Turn] = []
        self.stage = "idle"          # idle|listening|hearing|deciding|speaking
        self.pending: dict | None = None   # the open escalation
        self.running = False
        self.pipeline = None
        self.last_error = None
        self.platform_id = self.settings.platform
        self.preflight = platforms.Preflight()
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    # -- fanout ------------------------------------------------------------
    async def join(self, ws: WebSocket):
        await ws.accept()
        self._clients.add(ws)
        await ws.send_text(json.dumps(self.snapshot()))

    def leave(self, ws: WebSocket):
        self._clients.discard(ws)

    async def broadcast(self):
        if not self._clients:
            return
        msg = json.dumps(self.snapshot())
        dead = []
        for ws in self._clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def snapshot(self) -> dict:
        return {
            "stage": self.stage,
            "running": self.running,
            "pending": self.pending,
            "turns": [asdict(t) for t in self.turns[-60:]],
            "tier": asdict(self.tier),
            "settings": asdict(self.settings),
            "kb": {
                "text": self.kb.raw,
                "tokens": self.kb.token_estimate(),
                "warnings": self.kb.warnings(),
                "counts": {
                    "facts": len(self.kb.facts),
                    "positions": len(self.kb.positions),
                    "boundaries": len(self.kb.boundaries),
                },
            },
            "secrets": {k: redact(get_secret(k)) for k in SECRET_KEYS},
            "audio": self.audio_status(),
            "voice": {"enrolled": self.voice_enrolled(),
                      "engine": tts.engine_label(self.tier.tts_engine),
                      "script": tts.REFERENCE_SCRIPT, "note": tts.ENROLLMENT_NOTE,
                      "needs_license": self.tier.tts_engine == "xtts",
                      "license_accepted": self.settings.xtts_license_accepted,
                      "license_url": tts.XTTS_LICENSE_URL,
                      "license_summary": tts.XTTS_LICENSE_SUMMARY},
            "latency": self.pipeline.latency_report() if self.pipeline else {},
            "error": self.last_error,
            "preflight": self.preflight.status(platforms.get(self.platform_id)),
            "platforms": [{"id": p.id, "name": p.name} for p in platforms.PLATFORMS.values()],
        }

    def audio_status(self) -> dict:
        try:
            from .audio import describe_index, diagnose
            d = diagnose()
            d["capture_label"] = describe_index(self.settings.capture_device)
            d["playback_label"] = describe_index(self.settings.playback_device)
            return d
        except Exception as e:
            return {"ok": False, "problem": str(e), "devices": [],
                    "capture_candidates": [], "playback_candidates": []}

    def voice_enrolled(self) -> bool:
        try:
            return tts.build_voice(self.tier.tts_engine).enrolled
        except Exception:
            return False

    # -- pipeline hooks ----------------------------------------------------
    async def set_stage(self, stage: str):
        self.stage = stage
        await self.broadcast()

    async def add_turn(self, **kw) -> Turn:
        t = Turn(id=uuid.uuid4().hex[:8], t=time.time(), **kw)
        self.turns.append(t)
        await self.broadcast()
        return t

    async def open_escalation(self, question: str, reason: str, guard: str | None,
                              confidence: float | None):
        self.pending = {
            "id": uuid.uuid4().hex[:8],
            "question": question,
            "reason": reason,
            "guard": guard,
            "confidence": confidence,
            "opened_at": time.time(),
        }
        await self.broadcast()

    async def resolve_escalation(self, text: str) -> dict | None:
        async with self._lock:
            p = self.pending
            if not p:
                return None
            p["answer"] = text
            p["latency_s"] = round(time.time() - p["opened_at"], 1)
            self.pending = None
        await self.add_turn(speaker="you", text=text, tier="human")
        return p


hub = Hub()
app = FastAPI(title="quorum.ai")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/enroll")
async def enroll(reference: UploadFile = File(...)):
    """Accept a reference recording and compute the speaker latent once.

    Runs off the event loop because conditioning takes a few seconds, and
    writes the upload to a real file first — the TTS engines want a path.
    """
    import tempfile

    data = await reference.read()
    if len(data) < 32_000:
        return JSONResponse(
            {"ok": False, "error": "That clip is too short. Aim for 45 to 60 "
                                   "seconds — reference length is most of what "
                                   "determines how the clone sounds."},
            status_code=400)

    suffix = Path(reference.filename or "ref.wav").suffix.lower() or ".wav"
    if suffix not in (".wav", ".flac"):
        return JSONResponse(
            {"ok": False, "error": f"Need a .wav or .flac file, got {suffix}. "
                                   "Windows Voice Recorder saves .m4a — convert "
                                   "it first."},
            status_code=400)

    tmp = Path(tempfile.gettempdir()) / f"quorum-ref{suffix}"
    tmp.write_bytes(data)

    if hub.tier.tts_engine == "xtts" and not hub.settings.xtts_license_accepted:
        tmp.unlink(missing_ok=True)
        return JSONResponse(
            {"ok": False, "error": "Accept the XTTS model licence above before "
                                   "enrolling a voice."},
            status_code=400)

    try:
        tts.accept_xtts_license()
        voice = tts.build_voice(hub.tier.tts_engine)
        await asyncio.to_thread(voice.enroll, tmp)
        await asyncio.to_thread(voice.prerender, hub.settings.holding_line)
        hub.last_error = None
    except Exception as e:
        hub.last_error = f"enrollment: {e}"
        await hub.broadcast()
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        tmp.unlink(missing_ok=True)

    await hub.broadcast()
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await hub.join(ws)
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            await handle(msg)
    except WebSocketDisconnect:
        hub.leave(ws)
    except Exception:
        hub.leave(ws)


async def handle(msg: dict):
    kind = msg.get("type")

    if kind == "reply":
        text = (msg.get("text") or "").strip()
        if text:
            await hub.resolve_escalation(text)
            if hub.pipeline:
                hub.pipeline.submit_human_reply(text)
            await hub.set_stage("speaking")

    elif kind == "dismiss":
        hub.pending = None
        if hub.pipeline:
            hub.pipeline.dismiss()
        await hub.broadcast()

    elif kind == "kb":
        hub.kb = kbmod.save(msg.get("text", ""))
        await hub.broadcast()

    elif kind == "secret":
        name, value = msg.get("name"), (msg.get("value") or "").strip()
        if name in SECRET_KEYS and value:
            set_secret(name, value)
        await hub.broadcast()

    elif kind == "settings":
        for k, v in (msg.get("values") or {}).items():
            if hasattr(hub.settings, k):
                cur = getattr(hub.settings, k)
                try:
                    setattr(hub.settings, k, type(cur)(v))
                except (TypeError, ValueError):
                    pass
        hub.settings.save()
        await hub.broadcast()

    elif kind == "license":
        hub.settings.xtts_license_accepted = bool(msg.get("accepted"))
        hub.settings.save()
        await hub.broadcast()

    elif kind == "autonomy":
        mode = msg.get("mode")
        if mode in ("ask", "improvise"):
            hub.settings.autonomy = mode
            hub.settings.save()
            if hub.pipeline:
                hub.pipeline.settings.autonomy = mode   # applies mid-meeting
        await hub.broadcast()

    elif kind == "platform":
        pid = msg.get("id")
        if pid in platforms.PLATFORMS:
            hub.platform_id = pid
            hub.settings.platform = pid
            hub.settings.save()
        await hub.broadcast()

    elif kind == "preflight":
        hub.preflight.confirm(msg.get("key", ""), bool(msg.get("on", True)))
        await hub.broadcast()

    elif kind == "run":
        await _set_running(bool(msg.get("on")))

    elif kind == "demo":
        asyncio.create_task(_demo())


async def _set_running(on: bool):
    if not on:
        if hub.pipeline:
            hub.pipeline.stop()
        hub.running, hub.stage = False, "idle"
        return await hub.broadcast()

    try:
        hub.pipeline = _build_pipeline()
        hub.last_error = None
    except Exception as e:
        hub.last_error = str(e)
        hub.running, hub.stage = False, "idle"
        return await hub.broadcast()

    hub.running, hub.stage = True, "warming"
    await hub.broadcast()
    await asyncio.to_thread(hub.pipeline.warm)
    hub.pipeline.start()
    hub.stage = "listening"
    await hub.broadcast()


def _build_pipeline():
    """Constructed fresh on each start so edits to the KB and settings apply."""
    from .backends import build_llm
    from .pipeline import Hooks, Pipeline
    from .router import Router

    diag = hub.audio_status()
    if not diag["ok"]:
        raise RuntimeError(diag["problem"])

    loop = asyncio.get_running_loop()

    def spawn(coro):
        asyncio.run_coroutine_threadsafe(coro, loop)

    def on_error(where, exc):
        hub.last_error = f"{where}: {exc}"
        spawn(hub.broadcast())

    hooks = Hooks(
        on_stage=lambda st: spawn(hub.set_stage(st)),
        on_heard=lambda text, meta: spawn(hub.add_turn(speaker="room", text=text)),
        on_answer=lambda text, d, st: spawn(hub.add_turn(
            speaker="agent", text=text, tier="answer", confidence=d.confidence,
            improvised=getattr(d, "improvised", False),
            reason=d.reason, stages=st.as_dict())),
        on_escalate=lambda q, d: spawn(_escalated(q, d)),
        on_error=on_error,
    )

    voice = tts.build_voice(hub.tier.tts_engine)
    if hub.tier.tts_engine == "xtts" and not voice.enrolled:
        raise RuntimeError("No voice enrolled yet. Record a reference clip "
                           "under Settings, or switch to the Piper fallback.")

    return Pipeline(
        tier=hub.tier, settings=hub.settings, kb=hub.kb,
        router=Router(build_llm(hub.settings), hub.kb,
                      hub.settings.escalate_below_confidence,
                      autonomy=hub.settings.autonomy),
        voice=voice, hooks=hooks,
        capture_device=hub.settings.capture_device or None,
        playback_device=hub.settings.playback_device or None,
    )


async def _escalated(question, d):
    await hub.add_turn(speaker="agent", text=hub.settings.holding_line,
                       tier="escalate", confidence=d.confidence,
                       guard=d.guard, reason=d.reason)
    await hub.open_escalation(question, d.reason, d.guard, d.confidence)


async def _demo():
    """Scripted walkthrough so the UI can be driven without a soundcard."""
    async def beat(stage, s=0.5):
        await hub.set_stage(stage)
        await asyncio.sleep(s)

    hub.running = True
    await hub.add_turn(speaker="room", text="Alright, quick round of updates.")
    await asyncio.sleep(0.8)

    await hub.add_turn(speaker="room", text="AI Kartik, are you working on the competitor analysis?")
    await beat("hearing", 0.4); await beat("deciding", 0.7); await beat("speaking", 0.3)
    await hub.add_turn(
        speaker="agent",
        text="Yes, that's underway. The draft is due Wednesday.",
        tier="answer", confidence=0.93, reason="in knowledge base",
        stages={"vad": 300, "stt": 241, "router": 612, "tts": 1180},
    )
    await beat("listening", 1.4)

    await hub.add_turn(speaker="room", text="AI Kartik, do you think we should pivot the business?")
    await beat("hearing", 0.4); await beat("deciding", 0.2)
    await hub.add_turn(
        speaker="agent", text="Let me check on that and come back to you.",
        tier="escalate", confidence=1.0, guard="opinion", reason="opinion rule",
        stages={"vad": 300, "stt": 236, "router": 3, "tts": 0},
    )
    await hub.open_escalation(
        "do you think we should pivot the business",
        "opinion rule", "opinion", 1.0,
    )
    await hub.set_stage("waiting")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
