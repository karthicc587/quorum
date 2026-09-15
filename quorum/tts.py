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

REFERENCE_SCRIPT = [
    "The quarterly review is scheduled for the first week of next month.",
    "I've gone through the draft and left a few comments in the margin.",
    "Could you send that over when you get a chance? No rush at all.",
    "Seventeen, forty-two, ninety-eight, three hundred and six.",
    "Honestly, I think the second option makes more sense than the first.",
    "That's a good question — let me look into it and get back to you.",
]

ENROLLMENT_NOTE = (
    "Read the six lines at a normal speaking pace in a quiet room. Reference "
    "quality is the single biggest factor in how the clone sounds and it "
    "cannot be fixed afterwards: no echo, no fan noise, no compression."
)


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


class XTTSVoice(Voice):
    """XTTS-v2 via the maintained `coqui-tts` fork.

    The weights are CPML licensed — non-commercial only, which is fine for
    coursework but must be stated if the repo is public.
    """

    def __init__(self, latent_path: Path | None = None):
        self.latent_path = latent_path or VOICE_DIR / "latent.pkl"
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

    def enroll(self, reference_wav: Path) -> None:
        m = self._load().synthesizer.tts_model
        gpt_latent, speaker_emb = m.get_conditioning_latents(
            audio_path=[str(reference_wav)]
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
        out = m.inference(text, "en", gpt_latent, speaker_emb, temperature=0.6)
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
        import hashlib
        return self.dir / f"{hashlib.sha1(text.encode()).hexdigest()[:16]}.npy"

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


def build_voice(tier_engine: str) -> CachedVoice:
    inner: Voice = XTTSVoice() if tier_engine == "xtts" else PiperVoice()
    return CachedVoice(inner)


def engine_label(tier_engine: str) -> str:
    return {
        "xtts": "XTTS-v2 — clones your voice from a recorded sample",
        "openvoice": "OpenVoice — tone transfer, faster but less like you",
    }.get(tier_engine, "Piper — a fixed neural voice, not a clone")
