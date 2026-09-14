#!/usr/bin/env bash
# quorum.ai setup. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")"

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
warn() { printf "\033[33m%s\033[0m\n" "$1"; }
fail() { printf "\033[31m%s\033[0m\n" "$1"; }
ok()   { printf "\033[32m%s\033[0m\n" "$1"; }

bold "quorum.ai setup"
echo

# ---------------------------------------------------------------- python ---
if ! command -v python3 >/dev/null; then
  fail "python3 not found."; exit 1
fi
PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
  fail "Python 3.10+ required, found $PYV."; exit 1
fi
ok "Python $PYV"

# ------------------------------------------------------------------ tier ---
OS=$(uname -s)
ARCH=$(uname -m)
TIER="cpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  if [ "${VRAM:-0}" -ge 7000 ]; then
    TIER="cuda"; ok "NVIDIA GPU with ${VRAM}MB — tier: cuda"
  else
    warn "NVIDIA GPU with only ${VRAM}MB. Falling back to cpu tier."
  fi
elif [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
  TIER="apple"; ok "Apple Silicon — tier: apple"
fi
[ "$TIER" = "cpu" ] && warn "No usable GPU — tier: cpu. The router will use Groq's free API."

# --------------------------------------------------------- system deps -----
bold "\nSystem dependencies"
case "$OS" in
  Darwin)
    if ! brew list portaudio >/dev/null 2>&1; then
      warn "portaudio missing. Installing via Homebrew."
      brew install portaudio || fail "brew install portaudio failed — install it manually."
    else ok "portaudio"; fi
    if ! ls /Library/Audio/Plug-Ins/HAL 2>/dev/null | grep -qi blackhole; then
      warn "BlackHole not found. Download: https://existential.audio/blackhole/"
      warn "Then open Audio MIDI Setup and build a Multi-Output Device combining"
      warn "BlackHole with your speakers, or you will not hear the meeting."
      warn "It needs admin rights and a reboot — do this well before demo day."
    else ok "BlackHole"; fi
    ;;
  Linux)
    if ! ldconfig -p 2>/dev/null | grep -q libportaudio; then
      warn "libportaudio2 missing: sudo apt install libportaudio2"
    else ok "libportaudio2"; fi
    if command -v pactl >/dev/null; then
      pactl list short sinks 2>/dev/null | grep -q quorum \
        && ok "quorum null sink" \
        || warn "No null sink. Create one:
    pactl load-module module-null-sink sink_name=quorum \\
      sink_properties=device.description=quorum"
    fi
    ;;
  *)
    warn "Windows: install VoiceMeeter Banana — https://vb-audio.com/Voicemeeter/banana.htm"
    warn "A plain VB-Cable is not enough on one machine: you need the virtual mic"
    warn "and a loopback capture at the same time."
    ;;
esac

# ------------------------------------------------------------- packages ----
bold "\nPython packages"
EXTRAS="dev,audio"
[ "$TIER" != "cpu" ] && EXTRAS="$EXTRAS,voice"

if command -v uv >/dev/null 2>&1; then
  uv pip install -e ".[$EXTRAS]"
else
  python3 -m pip install -e ".[$EXTRAS]" || \
    python3 -m pip install --break-system-packages -e ".[$EXTRAS]"
fi
ok "installed extras: $EXTRAS"

# ------------------------------------------------------------------ env ----
if [ ! -f .env ]; then
  cp .env.example .env
  warn "Created .env — add your Groq key there, or paste it into Settings."
  echo "  Free tier, no card: https://console.groq.com/keys"
else
  grep -q '^GROQ_API_KEY=.\+' .env && ok "Groq key present" || warn "No Groq key in .env yet."
fi

# --------------------------------------------------------------- verify ----
bold "\nVerifying"
python3 - <<'PY'
from quorum.config import detect_tier
from quorum.audio import diagnose
t = detect_tier()
print(f"  tier            {t.name}")
print(f"  speech          {t.stt_model} on {t.stt_device}")
print(f"  voice           {t.tts_engine}")
print(f"  expected reply  ~{t.est_latency_ms/1000:.1f}s")
d = diagnose()
print(f"  audio routing   {'ready' if d['ok'] else 'NOT READY'}")
if not d["ok"]:
    print(f"    {d['problem']}")
PY

bold "\nNext"
cat <<'EOF'
  1. python -m uvicorn quorum.server:app
  2. Open http://127.0.0.1:8000
  3. Settings: add the Groq key, pick your audio devices, record the voice sample
  4. Pick your meeting platform and work through its checklist
  5. Press Run demo to see a full turn without any audio hardware

  Zoom needs Original Sound armed in every single meeting. It does not persist.
EOF
