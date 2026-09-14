"""Per-platform setup, because the meeting client is not as neutral as the
virtual-cable architecture makes it look.

The audio path is identical for Meet and Zoom — both read whatever the OS
presents as a microphone, so neither can tell the difference. What differs is
what each client does to that audio on the way in, and how you rename a tile.

The one that actually breaks demos: **Zoom's noise suppression is far more
aggressive than Meet's.** Zoom classifies synthetic speech as non-voice and
gates it, so the room hears the first syllable and then silence. Enabling
Original Sound is not optional there, and it has to be re-armed in-meeting on
every call because the in-meeting toggle does not persist.

Nothing here can be verified from inside the app — these are all client-side
settings in someone else's UI. So they are presented as a checklist the user
confirms, and `Preflight` records what was confirmed and when.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import STATE_DIR


@dataclass
class Step:
    key: str
    title: str
    detail: str
    critical: bool = False      # demo fails outright without it


@dataclass
class Platform:
    id: str
    name: str
    disclosure_chat: str
    rename_how: str
    steps: list[Step]
    notes: list[str] = field(default_factory=list)


MEET = Platform(
    id="meet",
    name="Google Meet",
    rename_how=(
        "Meet takes the tile name from your Google account, and there is no "
        "in-meeting rename. Either switch to a Google profile whose name is "
        "already the disclosure label, or join from a browser profile signed "
        "into one. Do this before the call — changing it mid-meeting does not "
        "update the tile for people already in the room."
    ),
    disclosure_chat=(
        "Heads up: I'm attending through an AI delegate rather than in person. "
        "It answers from a fixed set of notes and hands anything else back to "
        "me. My tile is labelled accordingly."
    ),
    steps=[
        Step("meet_noise", "Turn off noise cancellation",
             "Settings, Audio, then untick Noise cancellation. Meet's "
             "suppression treats synthetic speech as background noise and "
             "clips the start of every sentence.", critical=True),
        Step("meet_mic", "Set the microphone to your virtual device",
             "Settings, Audio, Microphone — pick BlackHole or VoiceMeeter "
             "Output, not your real mic.", critical=True),
        Step("meet_speaker", "Set the speaker to your real output",
             "Same panel. If this points at the virtual device you will not "
             "hear the meeting, and neither will the transcriber."),
        Step("meet_name", "Confirm the tile reads as the disclosure label",
             "Look at your own tile in the grid, not the participants list."),
        Step("meet_chat", "Post the disclosure message in chat",
             "At join, before the agent speaks for the first time."),
    ],
    notes=[
        "Meet has no in-meeting rename, so the label has to be right before "
        "you join.",
        "Meet's grid hides your own tile from you by default in some layouts. "
        "Check what others see, ideally from a second device.",
    ],
)

ZOOM = Platform(
    id="zoom",
    name="Zoom",
    rename_how=(
        "Participants panel, hover your own name, More, Rename. It updates "
        "live for everyone. Hosts can disable renaming, so if you are not the "
        "host, confirm with them beforehand — and set your Zoom profile name "
        "to the label as a fallback."
    ),
    disclosure_chat=(
        "Heads up: I'm attending through an AI delegate rather than in person. "
        "It answers from a fixed set of notes and hands anything else back to "
        "me. My name in the participants list is labelled accordingly."
    ),
    steps=[
        Step("zoom_original_enable", "Enable Original Sound in settings",
             "Settings, Audio, Advanced, then tick 'Show in-meeting option to "
             "enable Original Sound'. This only reveals the toggle — it does "
             "not turn it on.", critical=True),
        Step("zoom_original_arm", "Turn Original Sound on in the meeting",
             "Top-left of the meeting window, every single call. It does not "
             "persist between meetings. Without it Zoom's noise suppression "
             "gates synthetic speech and the room hears a clipped fragment.",
             critical=True),
        Step("zoom_suppression", "Set background noise suppression to Low",
             "Settings, Audio, Suppress background noise. Auto is the default "
             "and is the aggressive one.", critical=True),
        Step("zoom_autogain", "Turn off Automatically adjust microphone volume",
             "Settings, Audio. Auto-gain pumps on synthetic speech because it "
             "has a flatter dynamic range than a real voice."),
        Step("zoom_mic", "Set the microphone to your virtual device",
             "Settings, Audio, Microphone.", critical=True),
        Step("zoom_speaker", "Set the speaker to your real output",
             "Same panel."),
        Step("zoom_name", "Rename yourself to the disclosure label",
             "Participants, hover your name, More, Rename."),
        Step("zoom_chat", "Post the disclosure message in chat",
             "At join. Zoom chat defaults to Everyone, but check — if it is "
             "set to Host only, nobody else sees the disclosure."),
    ],
    notes=[
        "Zoom's audio processing is the main difference from Meet, and it is "
        "the single most common cause of 'the agent spoke but nobody heard a "
        "full sentence'.",
        "Original Sound has to be re-armed every call. Put it on the checklist "
        "rather than trusting memory.",
        "If the host has locked renaming, the participants-list label is the "
        "one that persists, so set the Zoom profile name too.",
        "Zoom's recording captures the renamed label, which is what you want "
        "for anyone watching later.",
    ],
)

PLATFORMS = {p.id: p for p in (MEET, ZOOM)}


def get(platform_id: str) -> Platform:
    return PLATFORMS.get(platform_id, MEET)


# ---------------------------------------------------------------- preflight
class Preflight:
    """Remembers which client-side steps were confirmed, and when.

    Steps marked critical expire after a day: Zoom's Original Sound toggle
    genuinely does reset between meetings, so a tick from last week is worse
    than no tick at all — it tells you something is done when it is not.
    """

    TTL_S = 24 * 3600

    def __init__(self, path: Path | None = None):
        self.path = path or STATE_DIR / "preflight.json"
        self._data: dict = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception:
                self._data = {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))

    def confirm(self, key: str, on: bool = True):
        if on:
            self._data[key] = time.time()
        else:
            self._data.pop(key, None)
        self._save()

    def is_confirmed(self, step: Step) -> bool:
        ts = self._data.get(step.key)
        if ts is None:
            return False
        if step.critical and time.time() - ts > self.TTL_S:
            return False
        return True

    def status(self, platform: Platform) -> dict:
        steps = [{
            "key": s.key, "title": s.title, "detail": s.detail,
            "critical": s.critical, "done": self.is_confirmed(s),
        } for s in platform.steps]
        blocking = [s for s in steps if s["critical"] and not s["done"]]
        return {
            "platform": platform.id,
            "name": platform.name,
            "steps": steps,
            "notes": platform.notes,
            "rename_how": platform.rename_how,
            "disclosure_chat": platform.disclosure_chat,
            "ready": not blocking,
            "blocking": [s["title"] for s in blocking],
        }
