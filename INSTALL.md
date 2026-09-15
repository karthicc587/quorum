# Setting up on a new machine

Ordered. Steps 1–4 need admin rights and a reboot, so do them a day ahead
rather than on the morning of a demo.

Every warning below cost real debugging time on the first install. None of
them are guessable from an error message.

---

## 1. Python

Install **3.11 or 3.12** from [python.org](https://www.python.org/downloads/windows/),
not the Microsoft Store — the Store build has permission quirks that break
editable installs.

**Tick "Add python.exe to PATH" on the first screen.** Miss it and every
command below fails with "not recognized".

3.13 works but is the least-tested path: the original Coqui `TTS` package caps
at <3.12, which is why this project uses the maintained `coqui-tts` fork.

Close and reopen your terminal after installing, then:

```
python --version
```

## 2. Git

[git-scm.com/download/win](https://git-scm.com/download/win), accept the
defaults. Reopen the terminal again.

## 3. VoiceMeeter Banana

[vb-audio.com/Voicemeeter/banana.htm](https://vb-audio.com/Voicemeeter/banana.htm)
— needs admin, **needs a reboot**.

Banana specifically. The basic VoiceMeeter has only one B bus, which forces
the agent's voice and the meeting audio down the same path — exactly the
feedback loop the gate module exists to prevent. Both install side by side, so
check the title bar says BANANA and not just VOICEMEETER.

## 4. Reboot

Not optional. The virtual audio devices do not appear until you do.

---

## 5. The code

```
git clone https://github.com/<you>/quorum.git
cd quorum
pip install -e ".[dev,audio,voice,join]"
playwright install chromium
python -m pytest tests/ -q
```

Expect **201 passed**. If you see import errors, the folder structure is
wrong — `quorum/`, `static/` and `tests/` must sit directly inside the repo
root.

If a fresh resolve pulls different versions and something breaks, install from
the lock file instead: `pip install -r requirements-lock.txt`.

### If the GPU is not being used

`nvidia-smi` reporting a card does not mean PyTorch can use it — pip installs
a CPU-only build by default on Windows, and CPU inference runs roughly ten
times slower.

```
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Settings → "This machine" reports which tier is actually live, and says why if
a GPU was found but is unusable.

---

## 6. VoiceMeeter routing

Open Banana. Two loops: the agent's voice out to the meeting, the meeting's
audio in to the transcriber.

| Strip | Lit buttons | Why |
|---|---|---|
| Stereo Input 1–3 (real mics) | nothing | otherwise your live voice is mixed into what the agent sends |
| Voicemeeter Input (strip 4) | **B1** | the agent's voice, heading for the meeting |
| Voicemeeter AUX Input (strip 5) | **B2**, **A1** | the meeting's audio, to the transcriber and to your ears |

Top right, set hardware out **A1** to your headphones or speakers.

Then Menu → **Save Settings** to a file. If the matrix ever gets scrambled,
Load Settings restores it in one click.

Windows default output → **VoiceMeeter Aux Input**.

**VoiceMeeter must be running.** Closed, the virtual devices carry no audio at
all, and nothing in the app can tell you why.

Speakers are fine — no echo risk here, because the meeting's microphone is a
software bus, not a real mic, so there is nothing acoustic to pick up.

### For solo testing only

Light **B2** on the mic strip so the app hears your live voice, and **A1** on
strip 4 so you can hear the agent. Turn both off before a real meeting, or
every question gets transcribed twice and the agent talks in your ear.

---

## 7. Keys and voice

Double-click `start.bat`, then in Settings:

1. **Groq API key** — free, no card, from
   [console.groq.com](https://console.groq.com). Paste, Save, then **restart
   the server**: `.env` is read at startup.
2. **Audio routing** — pick the entries marked **MME ✓**. Capture from
   `Voicemeeter Out B2`, speak to `Voicemeeter Input`. WASAPI entries refuse
   16 kHz and will fail.
3. **Voice** — accept the model licence, then upload a 45–60 second WAV of
   yourself reading the six lines shown. Quiet room, no compression. Reference
   quality is most of what determines how the clone sounds and it cannot be
   fixed later. Windows Voice Recorder saves `.m4a`; convert to WAV first.
4. **Knowledge base** — replace the placeholders with your real project facts.
   Until you do, every answer the agent gives is fiction.

First enrollment downloads about 2GB with no progress bar. It is not frozen.

---

## 8. Joining

```
python -m quorum.joiner login      # sign in by hand, once
python -m quorum.joiner status     # confirm the session saved
```

No password is ever typed by the code — Google blocks scripted sign-in, so the
browser profile holds the session instead.

**The profile is a live credential.** `.gitignore` must contain
`.quorum/browser-profile/` *before* your first commit. Committing it puts your
Google session in the repo, and deleting the file later does not remove it
from history.

Then paste a meeting link into the dashboard and press Join.

### The one thing that cannot be automated

Chrome has no switch for selecting a real audio device, so once the agent is
in the meeting, open the meeting client's own Settings → Audio:

- Microphone → **Voicemeeter Out B1**
- Speakers → **Voicemeeter Aux Input**
- Noise cancellation → **off**

The browser profile remembers all three per site, so this is once per machine.

Zoom additionally needs **Original Sound** enabled in settings *and* armed
inside each meeting — the in-meeting toggle does not persist, and without it
Zoom's noise suppression gates synthetic speech into fragments. The platform
checklist in Settings tracks this.

---

## Quick failure guide

| Symptom | Cause |
|---|---|
| "not recognized as a cmdlet" | PATH box unticked, or terminal not reopened |
| Nothing transcribed | VoiceMeeter closed, or no B2 on any strip |
| "cublas64_12.dll not found" | CPU-only PyTorch on the cuda tier |
| "error code: 1010" | Cloudflare blocking the HTTP client, not your API key |
| Router escalates everything | key saved after the server started — restart |
| First answer takes 30s | model loading; press Start and wait for "loading models" to clear |
| Agent audible to you, not the room | meeting client's mic not set to B1 |
| Room hears clipped fragments | noise suppression on, or Zoom without Original Sound |
