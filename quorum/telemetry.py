"""Per-turn logging.

One JSON object per line, one line per turn, appended as the meeting runs. The
eval reads these back; nothing else does. Written synchronously because the
volume is a few dozen lines an hour and losing the last turn to a buffer on a
crash would be worse than the cost of the write.

The `label` field is left empty on purpose. You fill it in afterwards by hand
— that is the ground truth the confusion matrix is built against, and it has
to come from a person, not from the system grading its own homework.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import ROOT

LOG_DIR = ROOT / "logs"


@dataclass
class TurnRecord:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    t: float = field(default_factory=time.time)

    heard: str = ""             # raw transcript
    question: str = ""          # wake phrase stripped
    wake_score: int = 0
    wake_fired: bool = False

    decision: str = ""          # answer | escalate
    guard: str | None = None
    confidence: float | None = None
    reason: str = ""
    spoken: str = ""

    escalated_to_human: bool = False
    human_reply: str | None = None
    human_latency_s: float | None = None
    abandoned: bool = False

    stt_ms: int = 0
    router_ms: int = 0
    tts_ms: int = 0
    audio_ms: int = 0
    tts_cached: bool = False

    error: str | None = None

    # filled in by hand afterwards
    label: str = ""             # answer | escalate — what should have happened
    label_note: str = ""

    @property
    def responsive_ms(self) -> int:
        return self.stt_ms + self.router_ms + self.tts_ms


class TurnLog:
    def __init__(self, session: str | None = None, directory: Path | None = None):
        self.dir = directory or LOG_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session = session or time.strftime("%Y%m%d-%H%M%S")
        self.path = self.dir / f"{self.session}.jsonl"
        self.count = 0

    def write(self, rec: TurnRecord) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        self.count += 1

    @staticmethod
    def read(path: Path) -> list[TurnRecord]:
        out = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(TurnRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue        # a truncated last line after a crash is normal
        return out

    @staticmethod
    def read_all(directory: Path | None = None) -> list[TurnRecord]:
        d = directory or LOG_DIR
        if not d.exists():
            return []
        recs = []
        for p in sorted(d.glob("*.jsonl")):
            recs.extend(TurnLog.read(p))
        return recs
