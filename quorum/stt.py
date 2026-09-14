"""Speech to text, chunked by voice activity rather than by a sliding window.

Whisper's own streaming examples slide a fixed window over the audio and
re-transcribe the overlap, which produces duplicated and half-formed phrases.
We instead let Silero VAD mark where speech starts and stops, and hand Whisper
one complete utterance at a time. Slightly higher latency, far cleaner text,
and it gives the turn boundaries the router needs anyway.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE

VAD_WINDOW = 512            # samples; Silero wants exactly this at 16 kHz


@dataclass
class Utterance:
    text: str
    started_at: float
    ended_at: float
    audio_ms: int
    stt_ms: int


class Segmenter:
    """Turns a stream of audio blocks into complete utterances.

    Pre-roll matters: speech onset is detected slightly after it begins, so we
    keep a short ring buffer and prepend it. Without this the first syllable
    is clipped, and "AI Kartik" arrives as "I Kartik" — which the fuzzy wake
    matcher tolerates, but only just.
    """

    def __init__(self, close_ms: int = 300, preroll_ms: int = 240,
                 min_speech_ms: int = 250, max_utterance_ms: int = 20_000,
                 threshold: float = 0.5):
        self.close_ms = close_ms
        self.min_speech_ms = min_speech_ms
        self.max_utterance_ms = max_utterance_ms
        self.threshold = threshold
        self._vad = None
        self._preroll = deque(maxlen=max(1, preroll_ms * SAMPLE_RATE // 1000 // VAD_WINDOW))
        self._buf: list[np.ndarray] = []
        self._speaking = False
        self._silence = 0
        self._started_at = 0.0

    def _model(self):
        if self._vad is None:
            try:
                from silero_vad import load_silero_vad
            except ImportError as e:
                raise RuntimeError(
                    "silero-vad is not installed. Run: pip install -e '.[audio]'"
                ) from e
            self._vad = load_silero_vad()
        return self._vad

    def _is_speech(self, window: np.ndarray) -> bool:
        import torch
        with torch.no_grad():
            prob = self._model()(torch.from_numpy(window), SAMPLE_RATE).item()
        return prob >= self.threshold

    def feed(self, block: np.ndarray):
        """Yields a complete utterance's audio when one closes."""
        for i in range(0, len(block) - VAD_WINDOW + 1, VAD_WINDOW):
            w = block[i:i + VAD_WINDOW]
            speech = self._is_speech(w)

            if not self._speaking:
                self._preroll.append(w)
                if speech:
                    self._speaking = True
                    self._silence = 0
                    self._started_at = time.time()
                    self._buf = list(self._preroll)
                    self._preroll.clear()
                continue

            self._buf.append(w)
            self._silence = 0 if speech else self._silence + 1

            closed = self._silence * VAD_WINDOW * 1000 / SAMPLE_RATE >= self.close_ms
            too_long = len(self._buf) * VAD_WINDOW * 1000 / SAMPLE_RATE >= self.max_utterance_ms

            if closed or too_long:
                audio = np.concatenate(self._buf)
                started = self._started_at
                self._speaking = False
                self._buf = []
                self._silence = 0
                if len(audio) * 1000 / SAMPLE_RATE >= self.min_speech_ms:
                    yield audio, started


class Transcriber:
    def __init__(self, model: str = "medium.en", device: str = "cpu",
                 compute_type: str = "int8"):
        self.name = model
        self.device = device
        self.compute_type = compute_type
        self._m = None

    def load(self):
        if self._m is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise RuntimeError(
                    "faster-whisper is not installed. Run: pip install -e '.[audio]'"
                ) from e
            self._m = WhisperModel(self.name, device=self.device,
                                   compute_type=self.compute_type)
        return self._m

    def transcribe(self, audio: np.ndarray, started_at: float) -> Utterance:
        t0 = time.perf_counter()
        segments, _ = self.load().transcribe(
            audio, language="en", beam_size=1, vad_filter=False,
            condition_on_previous_text=False,   # stops runaway repetition
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return Utterance(
            text=text,
            started_at=started_at,
            ended_at=time.time(),
            audio_ms=int(len(audio) * 1000 / SAMPLE_RATE),
            stt_ms=int((time.perf_counter() - t0) * 1000),
        )
