# quorum.ai

A disclosed AI meeting delegate. It listens to a meeting, answers routine
questions from a bounded knowledge base in your voice, and hands anything
requiring judgment back to you through a dashboard.

Research question: **can a disclosed AI delegate represent someone in a
low-stakes meeting with bounded autonomy?**

Disclosure is part of the design, not a wrapper around it. The tile is renamed
`AI Agent — Kartik` and a message goes in chat at join.

## What runs today

Everything that doesn't need a soundcard:

| Module | Status |
|---|---|
| `quorum/wake.py` | wake matching + echo suppression |
| `quorum/router.py` | deterministic guards + model routing, fails closed |
| `quorum/kb.py` | knowledge base parsing and warnings |
| `quorum/backends.py` | Groq with Ollama fallback |
| `quorum/gate.py` | playback gate state machine |
| `quorum/pipeline.py` | orchestrator, per-stage timing, error recovery |
| `quorum/audio.py` | device discovery and routing diagnostics |
| `quorum/stt.py` | VAD segmentation + faster-whisper |
| `quorum/tts.py` | XTTS with cached latent, pre-rendered holding line |
| `quorum/platforms.py` | Meet and Zoom profiles, preflight checklist |
| `quorum/telemetry.py` | per-turn JSONL logging |
| `quorum/evaluate.py` | confusion matrix, wake metrics, latency percentiles |
| `quorum/survey.py` | participant instrument and scoring |
| `quorum/server.py` | dashboard, WebSocket, settings |
| `static/index.html` | operator console |

130 tests, all passing. Everything except the model weights and the soundcard
is exercised: the pipeline runs end to end against fakes.

`python -m uvicorn quorum.server:app` then open <http://127.0.0.1:8000> and
press **Run demo** — the whole escalation flow plays without any audio hardware.

## Running it

Double-click **start.bat** (Windows) or **start.command** (macOS). It installs
anything missing on first run, starts the server, and opens the dashboard.
Keep the window open — closing it stops the agent.

To pin it: right-click start.bat, Send to, Desktop (create shortcut). Change
the shortcut's icon and it behaves like any other app.

There is deliberately no packaged .exe. Bundling PyTorch and CUDA produces a
~6GB binary that fails in ways that cannot be debugged from the outside; a
batch file that calls a normal Python install is both smaller and fixable.

## Setup

```bash
git clone <your-repo> && cd quorum-ai
uv sync                # or: pip install -e ".[dev]"
python -m uvicorn quorum.server:app --reload
```

Add a Groq API key under **Settings**. It is written to `.env`, which is
gitignored, and never appears in source or in the WebSocket payload. Free
tier, no card: <https://console.groq.com/keys>.

## The two decisions everything else follows from

**Escalation is asymmetric.** A wrong answer invents a commitment in your
name. A wrong escalation costs three seconds of meeting time. So every
threshold is tuned hard toward escalating, and the router fails closed on
malformed output, low confidence, empty drafts, and model outages alike.

**A model cannot be trusted to refuse consistently.** Anything that would
create a commitment is caught by regex before the model is consulted, and the
guard cannot be argued out of. `test_guard_short_circuits_before_model`
asserts the model is never even called on those.

## The knowledge base

`kb.md` — one file, under 3k tokens, injected whole. No vector database: the
corpus is a page, and RAG would be a weekend of work for no recall gain.

Three sections, and the split is the safety property:

- **Facts** — may be stated verbatim
- **Positions** — may be restated, never extended
- **Always escalate** — no matter how confident the model is

## Measured so far

- Wake phrase: 19/19 fixtures correct, 22-point separation between the lowest
  true positive (83) and the highest false positive (61) at threshold 72.
  One documented miss is pinned in `KNOWN_MISSES` rather than tuned away —
  reaching it would also fire the agent on "artificial" and "karaoke".
- Guards: 100% precision on answerable fixtures, 100% recall on guardable ones.
- Self-reply loop: covered both ways. One test proves the echo guard stops it,
  a control test disables the guard and asserts the loop happens.
- Latency, transcription accuracy and clone quality: not yet measured. Those
  need the hardware. `Pipeline.latency_report()` produces p50/p95 per stage
  once you run it.

## Two ways to be in the meeting

**Borrowed microphone** (the default). You join the call yourself and point the
client's microphone at the virtual cable. The agent speaks through it. Simple,
nothing to authenticate, but your machine has to sit in the meeting.

**Its own participant** (`quorum/joiner.py`). A browser joins the call as the
agent, renames itself, and posts the disclosure in chat.

```bash
pip install -e ".[join]" && playwright install chromium
python -m quorum.joiner login                    # sign in by hand, once
python -m quorum.joiner join <meeting-url>
```

Sign-in is manual on purpose. Google blocks scripted logins, and every product
in this space uses the same approach: authenticate once into a persistent
browser profile and reuse the cookie. No password is ever passed to or stored
by this code.

The browser is not headless — headless Chrome exposes no audio devices, so it
could neither hear the meeting nor speak into it. Expect a real Chrome window
you leave alone.

If a platform blocks automated joining, that is the platform's decision and the
answer is to join by hand. Nothing here tries to defeat that.

## Meet and Zoom

Both work, and the audio path is identical — a virtual cable presents itself as
a microphone and neither client can tell what is behind it. What differs is
what each does to that audio, and how you rename a tile.

**Zoom is the fussier one.** Its noise suppression is aggressive enough to
classify synthetic speech as non-voice and gate it, so the room hears the first
syllable and then nothing. Original Sound has to be enabled in settings *and*
armed inside each meeting — the in-meeting toggle does not persist, so it is on
the checklist rather than left to memory. Zoom also lets you rename in-meeting;
Meet does not, so the Meet label has to be right before you join.

Pick the platform in Settings and work through its checklist. Steps marked
required block nothing technically — the app cannot see into Zoom's settings —
but the header pill tells you what is outstanding, and required ticks expire
after a day so a stale confirmation never reads as done.

## Evaluation

Every turn is appended to `logs/<session>.jsonl` as it happens, including the
ones where the agent stayed quiet. The `label` field is left blank on purpose:
you fill it in by hand afterwards with what *should* have happened. That is the
ground truth, and it has to come from a person rather than the system grading
itself.

```bash
python -m quorum.evaluate                 # every session
python -m quorum.evaluate logs/x.jsonl    # one session
```

Reports the escalation confusion matrix, wake precision and recall, and
per-stage latency percentiles. The two error types are reported separately
rather than folded into an F1, because they are not comparable — averaging
them hides the entire design argument.

## The survey

The technical metrics answer whether the pipeline works. They cannot answer
whether a labelled delegate with hard limits is acceptable to the people in
the room, which is the actual claim.

Six items are asked word-for-word before and after the session, and the shift
is the result — attitudes to AI in meetings vary too much for a post-only
measure to mean anything. Three items are reverse-worded so that agreeing is
the negative answer; anyone straight-lining the column gets flagged rather
than counted as enthusiastic. Escalation is measured on its own, because the
design argument is that handing questions back is a feature and it should not
be buried inside a general satisfaction score.

```bash
python -m quorum.survey pre      # printable, for running it on paper
python -m quorum.survey post
python -m quorum.survey          # scored report once responses exist
```

Run the pre half before anyone sees the agent. You cannot retrofit a
before-and-after design.

## Before demo day

1. **Install the virtual audio device.** BlackHole on macOS, VoiceMeeter Banana
   on Windows. Both need admin rights and a reboot, so do it days ahead, not
   on the morning. Settings tells you exactly what is missing.
2. **Record the reference clip.** Six lines, quiet room, no compression. This
   is the single biggest factor in how the clone sounds and it cannot be fixed
   afterwards.
3. **Turn off noise and echo suppression in Meet**, or it will treat synthetic
   speech as echo and clip it.
4. **Press Start before anyone joins.** Warming loads the models and
   pre-renders the holding line; skipping it makes the first question take
   eight seconds instead of two.

## Still to build

`setup.sh` with tier auto-detection, in-browser recording for enrollment
(upload works today), and the eval scripts that turn the per-turn logs into
the confusion matrix.

## Licensing

The code here is yours. The XTTS-v2 **weights** are under the Coqui Public
Model License — non-commercial use only. Coursework is fine; shipping it in
anything commercial is not. The dashboard asks you to accept this before it
will load the model, and that acceptance is what gets forwarded to the
library.

`coqui-tts` itself (the maintained Idiap fork) is MPL-2.0.

## Not in scope

No work aimed at defeating detection. The disclosure label stays on.
