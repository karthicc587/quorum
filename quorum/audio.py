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


# Host APIs in preference order, from measurement rather than reputation.
# Probing a VoiceMeeter B-bus at the 16 kHz mono Whisper wants: MME opened and
# carried audio, DirectSound opened, WASAPI refused outright — it will not
# resample and only accepts the device's native rate. WDM-KS is exclusive-mode
# and locks the device away from everything else.
_API_RANK = {"MME": 0, "Windows DirectSound": 1, "Windows WASAPI": 2, "Windows WDM-KS": 3}


def _api_of(d: Device) -> str:
    sd = _sd()
    return sd.query_hostapis(sd.query_devices()[d.index]["hostapi"])["name"]


def _describe(devs: list[Device], kind: str, collapse: bool = True) -> list[dict]:
    """One entry per device, at its best host API.

    Windows exposes the same device once per host API, so eight virtual buses
    become forty dropdown entries that all look alike. Only the ranking makes
    them different, and the ranking is ours — so pick the best one per device
    and hide the rest. The index is what gets stored either way, since names
    repeat and selecting by name is ambiguous.
    """
    entries = []
    for d in devs:
        api = _api_of(d)
        entries.append({
            "index": d.index,
            "name": d.name,
            "api": api,
            "rank": _API_RANK.get(api, 9),
            "channels": d.inputs if kind == "input" else d.outputs,
            "label": d.name if collapse else f"{d.name}  ·  {api}",
            "recommended": api == "MME",
        })

    if collapse:
        best: dict[str, dict] = {}
        for e in entries:
            cur = best.get(e["name"])
            if cur is None or e["rank"] < cur["rank"]:
                best[e["name"]] = e
        entries = list(best.values())

    entries.sort(key=lambda e: (e["rank"], e["name"]))
    for e in entries:
        e.pop("rank", None)
    return entries


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
        "capture_candidates": _describe(v_in, "input"),
        "playback_candidates": _describe(v_out, "output"),
        "devices": [{"name": d.name, "in": d.inputs, "out": d.outputs,
                     "virtual": d.virtual} for d in devs],
    }


def resolve(value) -> int | None:
    """Turn a stored setting into a device index sounddevice can use.

    Always an index, never a name. sounddevice matches names by substring, so
    "Voicemeeter Input" also matches "Voicemeeter AUX Input" and it raises
    rather than choosing. Indices are unambiguous.
    """
    if value in (None, "", "default"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    # A name left over from an older config: take the highest-ranked match.
    want = str(value).lower()
    matches = [d for d in devices() if want in d.name.lower()]
    if not matches:
        return None
    matches.sort(key=lambda d: _API_RANK.get(_api_of(d), 9))
    return matches[0].index


def describe_index(index) -> str:
    """Human label for whatever is currently selected."""
    i = resolve(index)
    if i is None:
        return "the system default"
    # Look up by the device's own index, not by list position. They coincide
    # today, but relying on that turns any reordering into a silent mislabel.
    d = next((x for x in devices() if x.index == i), None)
    if d is None:
        return f"device {index} (not found)"
    return f"{d.name} ({_api_of(d)})"


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
