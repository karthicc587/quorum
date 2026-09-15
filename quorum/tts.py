"""Voice synthesis.

Two things here matter more than the engine choice:

  Cache the speaker latent. XTTS recomputes it from the reference clip on
  every call, which costs roughly 800ms per utterance for a value that never
  changes. Computed once at enrollment and pickled, it is free thereafter.

  Pre-render the holding line. The escalation path says the same sentence
  every time, so synthesising it live wastes the one moment where latency is
  most visible — the room has just asked a question and nobody has answered.
  Rendered at enrollment, it plays in about 400ms instead of 2.5s.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ROOT

VOICE_DIR = ROOT / "voices"
VOICE_DIR.mkdir(exist_ok=True)

# The base recording. Deliberately mixed: a flat statement, a question, a
# hesitation, numbers, an aside. A clone built only on even declarative
# sentences has nothing to go on when the agent has to ask something or
# trail off.
MASTER_SCRIPT = [
    "The quarterly review is scheduled for the first week of next month.",
    "I've gone through the draft and left a few comments in the margin.",
    "Could you send that over when you get a chance? No rush at all.",
    "Seventeen, forty-two, ninety-eight, three hundred and six.",
    "Honestly — and I might be wrong here — I think the second option makes "
    "more sense.",
    "That's a good question. Let me look into it and come back to you.",
    "Wait, sorry, can you say that again? I lost the thread for a second.",
    "Right, yeah, that works for me.",
]

# Optional passages, each covering something the base recording is thin on.
# Offered one at a time rather than as a wall of text: the realistic failure
# is someone recording sixty seconds once and never coming back, so every
# addition has to be a small, obviously worthwhile ask.
EXTRA_PASSAGES = [
    {
        "id": "questions",
        "title": "Asking things",
        "why": "The agent asks for clarification more than it states facts, "
               "and rising intonation is what the base script covers least.",
        "lines": [
            "Sorry, could you repeat the last part? I want to make sure I have it right.",
            "Is that the same deadline we agreed on, or has it moved?",
            "Who's picking that up — is it still with the same team?",
            "Do you want me to send the notes round afterwards?",
        ],
    },
    {
        "id": "hedging",
        "title": "Not being sure",
        "why": "Most of what the agent says out loud is a hedge. If this is "
               "missing it delivers 'I'd have to check' with more confidence "
               "than the words carry.",
        "lines": [
            "I'd have to check on that, off the top of my head I'm not certain.",
            "My sense is it's fine, but don't hold me to the exact number.",
            "That's a good question. Let me look into it and come back to you.",
            "Honestly, I'm not the right person to answer that one.",
        ],
    },
    {
        "id": "numbers",
        "title": "Numbers and dates",
        "why": "Dates and figures come out mechanically unless the model has "
               "heard you say some.",
        "lines": [
            "We're looking at the fourteenth, maybe the fifteenth at a push.",
            "It came in around twelve hundred, a bit under budget.",
            "Q3 was up about eight percent on the same quarter last year.",
            "Two thousand and twenty-six, first week of March.",
        ],
    },
    {
        "id": "quick",
        "title": "Talking quickly",
        "why": "The base script is read at reading pace. This catches how you "
               "sound when you are actually in a meeting and moving.",
        "lines": [
            "Yeah no that's fine, go ahead, I'll catch up on the notes after.",
            "Right — so if we do that, we push everything else back a week, "
            "which I don't love but it works.",
            "Sure, sure. Makes sense. Let's do that.",
        ],
    },
    {
        "id": "flat",
        "title": "Low energy",
        "why": "Nobody sounds bright in a Tuesday status meeting, and a clone "
               "built only on clear reading sounds oddly enthusiastic.",
        "lines": [
            "Nothing much to report from my side this week.",
            "Same as last time, still waiting on the other team.",
            "No blockers. It's moving, just slowly.",
        ],
    },
]


def passage(pid: str) -> dict | None:
    return next((p for p in EXTRA_PASSAGES if p["id"] == pid), None)


ENROLLMENT_NOTE = (
    "Read these in a quiet room at the pace you actually speak in a meeting. "
    "Around 60 seconds. No echo, no fan noise, no compression — reference "
    "quality cannot be fixed afterwards."
)


SAMPLE_DIR = VOICE_DIR / "samples"

# Below this the clone is recognisably you but not convincingly; above it,
# similarity stops improving and enrollment just gets slow.
GOOD_SECONDS = 60
ENOUGH_SECONDS = 90


class SampleLibrary:
    """The reference clips the voice is built from.

    Kept as separate files rather than one recording so they can be added a
    few at a time. Variety matters more than length past a point — six flat
    declarative sentences is the usual reason a clone sounds close but wrong —
    so the flow is designed around adding another take, not re-doing it all.
    """

    def __init__(self, directory: Path | None = None):
        self.dir = directory or SAMPLE_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def paths(self) -> list[Path]:
        return sorted(p for p in self.dir.glob("*.wav") if p.stat().st_size > 0)

    @staticmethod
    def duration(path: Path) -> float:
        try:
            import soundfile as sf
            info = sf.info(str(path))
            return round(info.frames / info.samplerate, 1)
        except Exception:
            return 0.0

    def add(self, data: bytes, passage_id: str = "extra") -> Path:
        """Filename carries the passage id, so the UI knows what is covered."""
        import re as _re
        pid = _re.sub(r"[^a-z0-9]+", "-", (passage_id or "extra").lower()).strip("-")[:24]
        n = len(self.paths()) + 1
        p = self.dir / f"{n:02d}__{pid or 'extra'}.wav"
        p.write_bytes(data)
        return p

    @staticmethod
    def passage_of(path: Path) -> str:
        return path.stem.split("__", 1)[-1] if "__" in path.stem else "extra"

    def recorded_passages(self) -> set[str]:
        return {self.passage_of(p) for p in self.paths()}

    @property
    def has_base(self) -> bool:
        return "base" in self.recorded_passages()

    def remove(self, name: str) -> bool:
        p = self.dir / Path(name).name        # never escape the directory
        if p.exists() and p.parent == self.dir:
            p.unlink()
            return True
        return False

    def clear(self) -> None:
        for p in self.paths():
            p.unlink()

    def total_seconds(self) -> float:
        return round(sum(self.duration(p) for p in self.paths()), 1)

    def status(self) -> dict:
        done = self.recorded_passages()
        clips = [{"name": p.name, "seconds": self.duration(p),
                  "passage": self.passage_of(p)} for p in self.paths()]
        total = round(sum(c["seconds"] for c in clips), 1)

        if not self.has_base:
            advice = ("Start with the base recording below — everything else "
                      "builds on it.")
        elif total < GOOD_SECONDS:
            advice = (f"{total:.0f}s. Under a minute the clone tends to sound "
                      f"close but not quite right; another {GOOD_SECONDS - total:.0f}s "
                      "makes a real difference.")
        else:
            remaining = [p for p in EXTRA_PASSAGES if p["id"] not in done]
            advice = (f"{total:.0f}s across {len(clips)} recording"
                      f"{'' if len(clips) == 1 else 's'}. "
                      + (f"{len(remaining)} optional passage"
                         f"{'' if len(remaining) == 1 else 's'} left if you want "
                         "to sharpen it further." if remaining else
                         "Every passage covered."))

        return {
            "clips": clips,
            "total_seconds": total,
            "advice": advice,
            "enough": total >= GOOD_SECONDS and self.has_base,
            "has_base": self.has_base,
            "passages": [{**p, "done": p["id"] in done} for p in EXTRA_PASSAGES],
            "next_passage": next((p["id"] for p in EXTRA_PASSAGES
                                  if p["id"] not in done), None),
        }


@dataclass
class Speech:
    samples: np.ndarray
    sample_rate: int
    tts_ms: int
    cached: bool = False


XTTS_LICENSE_URL = "https://coqui.ai/cpml"
XTTS_LICENSE_SUMMARY = (
    "XTTS-v2 is released under the Coqui Public Model License (CPML). "
    "Non-commercial use only. Coursework and research are fine; shipping it "
    "in anything commercial is not. You must accept this before the model "
    "will load."
)


def accept_xtts_license() -> None:
    """Coqui prompts for CPML agreement on stdin, which deadlocks a server.

    The acceptance itself is real and is collected in the dashboard — this
    only stops the library asking a second time on a terminal nobody is
    watching. Never call it without the user having agreed.
    """
    import os
    os.environ["COQUI_TOS_AGREED"] = "1"


def _load_audio_soundfile(path, *_, **__):
    """torchaudio.load replacement backed by soundfile.

    From torch 2.9, torchaudio.load delegates to torchcodec, which needs
    FFmpeg shared libraries installed system-wide — on Windows that means
    hunting down a "full-shared" build and editing PATH. We only ever load
    the reference clip, which the upload endpoint has already restricted to
    WAV or FLAC, and soundfile handles both through a bundled libsndfile
    with no system dependencies at all.
    """
    import soundfile as sf
    import torch

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


def _ensure_audio_backend() -> bool:
    """Patch torchaudio.load only if the real one cannot decode a WAV.

    Probing beats assuming: if torchcodec is properly installed we leave it
    alone. Returns True if we patched.
    """
    import tempfile

    import numpy as np
    import soundfile as sf
    import torchaudio

    if getattr(torchaudio.load, "_quorum_patched", False):
        return True

    probe = Path(tempfile.gettempdir()) / "quorum-backend-probe.wav"
    try:
        sf.write(probe, np.zeros(2048, dtype="float32"), 22050)
        torchaudio.load(str(probe))
        return False
    except Exception:
        _load_audio_soundfile._quorum_patched = True
        torchaudio.load = _load_audio_soundfile
        return True
    finally:
        probe.unlink(missing_ok=True)


_HINTS = {
    "transformers": "pip install 'transformers>=4.57,<5'",
    "torchcodec": "pip install torchcodec",
    "torchaudio": "pip install torchcodec",
}


def _import_hint(msg: str) -> str:
    low = msg.lower()
    for needle, fix in _HINTS.items():
        if needle in low:
            return f"  Try: {fix}"
    return ""


class Voice:
    """Base class. Engines differ; the pipeline should not have to care."""

    sample_rate = 24_000
    clones = True                       # does this engine reproduce a person?

    def enroll(self, reference_wav: Path) -> None: ...
    def say(self, text: str) -> Speech: ...

    def load(self) -> None:
        """Pull weights into memory now rather than on the first question."""

    @property
    def enrolled(self) -> bool:
        return False


# How much reference audio conditions the GPT. XTTS defaults to 6 seconds,
# which is enough to be recognisably you and not enough to be convincingly
# you. 30 is the practical ceiling — past that similarity stops improving and
# enrollment gets slow.
GPT_COND_LEN = 30
MAX_REF_LEN = 30


class XTTSVoice(Voice):
    """XTTS-v2 via the maintained `coqui-tts` fork.

    The weights are CPML licensed — non-commercial only, which is fine for
    coursework but must be stated if the repo is public.
    """

    def __init__(self, latent_path: Path | None = None, speed: float = 1.0):
        self.latent_path = latent_path or VOICE_DIR / "latent.pkl"
        self.speed = speed
        self._model = None
        self._latent = None

    def _load(self):
        if self._model is None:
            _ensure_audio_backend()
            try:
                from TTS.api import TTS
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "coqui-tts is not installed. Run: pip install -e '.[voice]'"
                ) from e
            except ImportError as e:
                # The package is present but something inside it failed to
                # import — usually a dependency that has moved on. Surface the
                # real error, and add a fix only when we recognise the cause.
                raise RuntimeError(
                    f"coqui-tts is installed but failed to load: {e}"
                    f"{_import_hint(str(e))}"
                ) from e
            self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        return self._model

    @property
    def enrolled(self) -> bool:
        return self.latent_path.exists()

    def enroll(self, reference_wav: Path | list[Path]) -> None:
        """Compute the speaker latent once, from one clip or several.

        Several is better: XTTS averages the conditioning, so a handful of
        clips covering different pitch and pace generalises where one flat
        reading does not.
        """
        paths = [reference_wav] if isinstance(reference_wav, (str, Path)) else list(reference_wav)
        m = self._load().synthesizer.tts_model
        gpt_latent, speaker_emb = m.get_conditioning_latents(
            audio_path=[str(p) for p in paths],
            gpt_cond_len=GPT_COND_LEN,
            max_ref_length=MAX_REF_LEN,
        )
        self.latent_path.write_bytes(pickle.dumps((gpt_latent, speaker_emb)))
        self._latent = (gpt_latent, speaker_emb)

    def load(self) -> None:
        self._load()
        self._get_latent()

    def _get_latent(self):
        if self._latent is None:
            if not self.enrolled:
                raise RuntimeError("No voice enrolled yet. Record a reference "
                                   "clip in the dashboard first.")
            self._latent = pickle.loads(self.latent_path.read_bytes())
        return self._latent

    def say(self, text: str) -> Speech:
        t0 = time.perf_counter()
        gpt_latent, speaker_emb = self._get_latent()
        m = self._load().synthesizer.tts_model
        out = m.inference(text, "en", gpt_latent, speaker_emb,
                          temperature=0.6, speed=self.speed)
        wav = np.asarray(out["wav"], dtype=np.float32)
        return Speech(wav, self.sample_rate, int((time.perf_counter() - t0) * 1000))


class PiperVoice(Voice):
    """Fallback. Not a clone — a fixed neural voice, but fast and reliable.

    `enrolled` stays False by design. Nothing needs recording, but the
    dashboard should still say plainly that this is not the user's voice
    rather than showing a green tick that means something else.
    """

    sample_rate = 22_050
    clones = False

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path
        self._v = None

    @property
    def enrolled(self) -> bool:
        return False                    # never a clone of anyone

    def _load(self):
        if self._v is None:
            try:
                from piper import PiperVoice as _P
            except ImportError as e:
                raise RuntimeError("piper-tts is not installed.") from e
            self._v = _P.load(str(self.model_path))
        return self._v

    def load(self) -> None:
        self._load()

    def say(self, text: str) -> Speech:
        t0 = time.perf_counter()
        chunks = [np.frombuffer(c, dtype=np.int16) for c in self._load().synthesize_stream_raw(text)]
        wav = (np.concatenate(chunks).astype(np.float32) / 32768.0) if chunks else np.zeros(0, np.float32)
        return Speech(wav, self.sample_rate, int((time.perf_counter() - t0) * 1000))


class CachedVoice(Voice):
    """Wraps any engine and memoises fixed phrases to disk.

    Only the holding line and a couple of other constants ever hit this, so
    the cache stays tiny — it is not a general TTS cache.
    """

    def __init__(self, inner: Voice, cache_dir: Path | None = None):
        self.inner = inner
        self.dir = cache_dir or VOICE_DIR / "cache"
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def sample_rate(self):
        return self.inner.sample_rate

    @property
    def clones(self) -> bool:
        return self.inner.clones

    @property
    def enrolled(self) -> bool:
        return self.inner.enrolled

    def enroll(self, reference_wav: Path) -> None:
        self.inner.enroll(reference_wav)
        for f in self.dir.glob("*.npy"):
            f.unlink()                  # a new voice invalidates every cache entry

    def _path(self, text: str) -> Path:
        # Key on speed too: a cached holding line rendered at the old rate
        # would keep playing after the setting changed.
        import hashlib
        speed = getattr(self.inner, "speed", 1.0)
        key = f"{text}|{speed:.2f}".encode()
        return self.dir / f"{hashlib.sha1(key).hexdigest()[:16]}.npy"

    def load(self) -> None:
        self.inner.load()

    def prerender(self, text: str) -> None:
        p = self._path(text)
        if not p.exists():
            np.save(p, self.inner.say(text).samples)

    def say(self, text: str) -> Speech:
        p = self._path(text)
        if p.exists():
            t0 = time.perf_counter()
            wav = np.load(p)
            return Speech(wav, self.sample_rate,
                          int((time.perf_counter() - t0) * 1000), cached=True)
        return self.inner.say(text)


def build_voice(tier_engine: str, speed: float = 1.0) -> CachedVoice:
    inner: Voice = XTTSVoice(speed=speed) if tier_engine == "xtts" else PiperVoice()
    return CachedVoice(inner)


def engine_label(tier_engine: str) -> str:
    return {
        "xtts": "XTTS-v2 — clones your voice from a recorded sample",
        "openvoice": "OpenVoice — tone transfer, faster but less like you",
    }.get(tier_engine, "Piper — a fixed neural voice, not a clone")
