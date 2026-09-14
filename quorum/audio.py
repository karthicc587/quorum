"""Audio I/O and, more importantly, telling you when your routing is wrong.

Single-machine routing is where this class of project dies, so the interesting
function here is `diagnose()`, not the streams.

What the routing has to achieve:

    Meet's output  ──> loopback capture ──> our STT
    our TTS        ──> virtual mic      ──> Meet's input
    Meet's output  ──> your headphones  (or you go deaf for the whole call)

macOS   BlackHole 2ch, plus a Multi-Output Device combining BlackHole and
        your real output so you can still hear the meeting.
Windows VoiceMeeter Banana. A plain VB-Cable is not enough on one machine:
        you need the virtual mic and a loopback capture at the same time,
        which is what the mixer gives you.
Linux   PulseAudio/PipeWire null sink plus its monitor source.
"""
from __future__ import annotations

import platform
import queue
import threading
from dataclasses import dataclass

SAMPLE_RATE = 16_000        # what Whisper wants
BLOCK_MS = 32
BLOCK = SAMPLE_RATE * BLOCK_MS // 1000


def _sd():
    try:
        import sounddevice as sd
        return sd
    except ImportError as e:
        raise RuntimeError(
            "sounddevice is not installed. Run: pip install -e '.[audio]'"
        ) from e
    except OSError as e:
        # sounddevice imports fine but the PortAudio shared library is absent
        raise RuntimeError(
            "PortAudio is missing. macOS: brew install portaudio. "
            "Debian/Ubuntu: sudo apt install libportaudio2. "
            f"({e})"
        ) from e


# --------------------------------------------------------------- discovery
VIRTUAL_HINTS = (
    "blackhole", "voicemeeter", "vb-audio", "cable", "soundflower",
    "null sink", "monitor of", "pulse", "loopback",
)


@dataclass
class Device:
    index: int
    name: str
    inputs: int
    outputs: int

    @property
    def virtual(self) -> bool:
        return any(h in self.name.lower() for h in VIRTUAL_HINTS)


def devices() -> list[Device]:
    sd = _sd()
    return [
        Device(i, d["name"], d["max_input_channels"], d["max_output_channels"])
        for i, d in enumerate(sd.query_devices())
    ]


def find(substring: str, kind: str = "input") -> Device | None:
    want = substring.lower()
    for d in devices():
        if want in d.name.lower():
            if kind == "input" and d.inputs > 0:
                return d
            if kind == "output" and d.outputs > 0:
                return d
    return None


INSTALL = {
    "Darwin": ("BlackHole 2ch", "https://existential.audio/blackhole/",
               "Then open Audio MIDI Setup and build a Multi-Output Device "
               "combining BlackHole with your speakers, or you will not hear "
               "the meeting."),
    "Windows": ("VoiceMeeter Banana", "https://vb-audio.com/Voicemeeter/banana.htm",
                "A plain VB-Cable will not do on one machine — you need the "
                "virtual mic and a loopback capture at once."),
    "Linux": ("a PulseAudio null sink", "",
              "pactl load-module module-null-sink sink_name=quorum "
              "sink_properties=device.description=quorum"),
}


def diagnose() -> dict:
    """Called by setup and by the dashboard. Never raises."""
    system = platform.system()
    name, url, note = INSTALL.get(system, ("a virtual audio device", "", ""))
    try:
        devs = devices()
    except Exception as e:
        return {"ok": False, "problem": str(e), "devices": [],
                "needs": name, "url": url, "note": note}

    v_in = [d for d in devs if d.virtual and d.inputs > 0]
    v_out = [d for d in devs if d.virtual and d.outputs > 0]

    problem = None
    if not v_in and not v_out:
        problem = (f"No virtual audio device found. Install {name} and reboot "
                   f"— it needs admin rights, so do this before demo day.")
    elif not v_in:
        problem = ("Found a virtual output but no matching input. On macOS the "
                   "Multi-Output Device is probably missing BlackHole.")
    elif not v_out:
        problem = "Found a virtual input but no virtual output to send speech to."

    return {
        "ok": problem is None,
        "problem": problem,
        "needs": name, "url": url, "note": note,
        "capture_candidates": [d.name for d in v_in],
        "playback_candidates": [d.name for d in v_out],
        "devices": [{"name": d.name, "in": d.inputs, "out": d.outputs,
                     "virtual": d.virtual} for d in devs],
    }


# ----------------------------------------------------------------- capture
class Capture:
    """Mono 16 kHz float32 blocks off the loopback device."""

    def __init__(self, device: int | str | None = None):
        self.device = device
        self.q: queue.Queue = queue.Queue(maxsize=64)
        self._stream = None
        self.dropped = 0

    def _cb(self, indata, frames, time_info, status):
        try:
            self.q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.dropped += 1          # surfaced in the dashboard, not silent

    def start(self):
        sd = _sd()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK, dtype="float32",
            channels=1, device=self.device, callback=self._cb,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop(); self._stream.close(); self._stream = None

    def blocks(self):
        while self._stream is not None:
            try:
                yield self.q.get(timeout=0.25)
            except queue.Empty:
                continue


# ---------------------------------------------------------------- playback
class Playback:
    """Writes generated speech into the virtual mic Meet is reading."""

    def __init__(self, device: int | str | None = None, sample_rate: int = 24_000):
        self.device = device
        self.sample_rate = sample_rate
        self._stop = threading.Event()

    def play(self, samples, on_start=None, on_end=None):
        """Blocking. Call from the pipeline worker, never the audio callback."""
        sd = _sd()
        self._stop.clear()
        if on_start:
            on_start()
        try:
            sd.play(samples, self.sample_rate, device=self.device, blocking=False)
            while sd.get_stream().active:
                if self._stop.wait(0.02):
                    sd.stop()
                    break
        finally:
            if on_end:
                on_end()

    def interrupt(self):
        self._stop.set()
