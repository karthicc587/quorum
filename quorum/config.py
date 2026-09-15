"""Runtime configuration. Secrets come from .env — never from source."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
STATE_DIR = ROOT / ".quorum"
STATE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# secrets
# --------------------------------------------------------------------------
def _load_env() -> None:
    """Minimal .env loader. No dependency, no surprises."""
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_env()


def get_secret(name: str) -> str | None:
    """Single accessor for every credential in the app."""
    val = os.environ.get(name)
    return val.strip() if val and val.strip() else None


def set_secret(name: str, value: str) -> None:
    """Persist a key to .env from the Settings screen. Chmod 600 on POSIX."""
    lines: list[str] = []
    if ENV_PATH.exists():
        lines = [
            ln for ln in ENV_PATH.read_text().splitlines()
            if not ln.strip().startswith(f"{name}=")
        ]
    lines.append(f"{name}={value.strip()}")
    ENV_PATH.write_text("\n".join(lines) + "\n")
    if os.name == "posix":
        ENV_PATH.chmod(0o600)
    os.environ[name] = value.strip()


def redact(value: str | None) -> str:
    if not value:
        return ""
    return f"{value[:6]}…{value[-4:]}" if len(value) > 12 else "…"


# --------------------------------------------------------------------------
# hardware tiers
# --------------------------------------------------------------------------
@dataclass
class Tier:
    name: str
    stt_model: str
    stt_compute: str
    stt_device: str
    tts_engine: str
    router_backend: str
    est_latency_ms: int
    note: str


TIERS = {
    "cuda": Tier(
        name="cuda",
        stt_model="medium.en",
        stt_compute="int8_float16",
        stt_device="cuda",
        tts_engine="xtts",
        router_backend="groq",
        est_latency_ms=2300,
        note="Router runs off-GPU so STT + XTTS + Chrome fit in 8GB.",
    ),
    "apple": Tier(
        name="apple",
        stt_model="medium.en",
        stt_compute="int8",
        stt_device="cpu",
        tts_engine="openvoice",
        router_backend="groq",
        est_latency_ms=4200,
        note="XTTS on Apple Silicon CPU is too slow; OpenVoice trades likeness for speed.",
    ),
    "cpu": Tier(
        name="cpu",
        stt_model="small.en",
        stt_compute="int8",
        stt_device="cpu",
        tts_engine="openvoice",
        router_backend="groq",
        est_latency_ms=3800,
        note="No local LLM on this tier — a 7B on CPU runs about 5 tok/s.",
    ),
}


def _has_cuda() -> tuple[bool, int]:
    """Return (available, vram_mb). Uses nvidia-smi so torch isn't needed."""
    if not shutil.which("nvidia-smi"):
        return False, 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return (True, int(out.stdout.strip().splitlines()[0])) if out.returncode == 0 else (False, 0)
    except Exception:
        return False, 0


def _torch_sees_cuda() -> tuple[bool, str]:
    """A GPU in the machine is not the same as a build that can use it.

    pip installs a CPU-only PyTorch by default on Windows, so nvidia-smi
    reporting a card tells you nothing about whether anything will run on it.
    Checking only nvidia-smi picks the cuda tier and then fails at load time
    with a missing-DLL error that looks unrelated.
    """
    try:
        import torch
    except ImportError:
        return False, "PyTorch is not installed yet"
    if torch.cuda.is_available():
        return True, ""
    build = "CPU-only build" if "+cpu" in torch.__version__ else "no usable CUDA device"
    return False, (
        f"PyTorch {torch.__version__} cannot use the GPU ({build}). Reinstall "
        "with CUDA support from https://pytorch.org/get-started/locally/ to "
        "get GPU speed; running on CPU otherwise."
    )


def detect_tier(override: str | None = None) -> Tier:
    if override:
        if override not in TIERS:
            raise ValueError(f"unknown tier {override!r}; pick from {list(TIERS)}")
        return TIERS[override]

    cuda, vram = _has_cuda()
    if cuda and vram >= 7000:
        usable, why = _torch_sees_cuda()
        if usable:
            return TIERS["cuda"]
        t = replace(TIERS["cpu"], note=f"A {vram}MB GPU was found but is unused. {why}")
        return t
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return TIERS["apple"]
    return TIERS["cpu"]


# --------------------------------------------------------------------------
# tunables — every magic number in the pipeline lives here
# --------------------------------------------------------------------------
def _state_dir() -> Path:
    """Read at call time so tests can redirect it."""
    import sys
    return getattr(sys.modules[__name__], "STATE_DIR")


@dataclass
class Settings:
    wake_phrase: str = "AI Kartik"
    wake_threshold: int = 72          # combined window/token score 0-100
    escalate_below_confidence: float = 0.70
    vad_close_ms: int = 300           # silence before an utterance is final
    tts_tail_ms: int = 500            # keep STT muted this long after playback
    echo_suppress_threshold: int = 85 # drop transcript matching our own speech
    transcript_window: int = 8        # turns of context handed to the router
    escalation_timeout_s: int = 180   # give up waiting for a typed reply
    # Groq retires model names on a schedule — llama-3.3-70b-versatile was
    # decommissioned in August 2026 and is now enterprise-only, which shows up
    # as a 403 rather than a 404. GROQ_FALLBACKS below covers the next rotation.
    groq_model: str = "openai/gpt-oss-120b"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    holding_line: str = "Let me check on that and come back to you."
    display_name: str = "AI Agent — Kartik"
    platform: str = "meet"            # meet | zoom
    # ask       — refuse, play the holding line, wait for a typed reply
    # improvise — answer anyway, from context, without waiting
    autonomy: str = "ask"
    xtts_license_accepted: bool = False
    capture_device: str = ""          # substring match; blank = system default
    playback_device: str = ""

    @classmethod
    def load(cls) -> "Settings":
        p = _state_dir() / "settings.json"
        if p.exists():
            try:
                return cls(**{**asdict(cls()), **json.loads(p.read_text())})
            except Exception:
                pass
        return cls()

    def save(self) -> None:
        (_state_dir() / "settings.json").write_text(json.dumps(asdict(self), indent=2))
