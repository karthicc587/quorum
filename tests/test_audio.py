"""Device resolution. The name-vs-index distinction caused a real failure:
sounddevice matches names by substring, so 'Voicemeeter Input' also matched
'Voicemeeter AUX Input' and it raised rather than picking one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quorum import audio
from quorum.audio import Device


FAKE = [
    Device(0, "Microphone (HD Web Camera)", 1, 0),
    Device(36, "Voicemeeter AUX Input (VB-Audio Voicemeeter VAIO)", 0, 8),
    Device(46, "Voicemeeter Input (VB-Audio Voicemeeter VAIO)", 0, 8),
    Device(58, "Voicemeeter Input (VB-Audio Voicemeeter VAIO)", 0, 2),
    Device(5, "Voicemeeter Out B2 (VB-Audio Voicemeeter VAIO)", 8, 0),
    Device(64, "Voicemeeter Out B2 (VB-Audio Voicemeeter VAIO)", 2, 0),
]
APIS = {0: "MME", 36: "Windows DirectSound", 46: "Windows DirectSound",
        58: "Windows WASAPI", 5: "MME", 64: "Windows WASAPI"}


@pytest.fixture(autouse=True)
def fake_devices(monkeypatch):
    monkeypatch.setattr(audio, "devices", lambda: FAKE)
    monkeypatch.setattr(audio, "_api_of", lambda d: APIS[d.index])


def test_index_passes_straight_through():
    assert audio.resolve(5) == 5
    assert audio.resolve("46") == 46


def test_blank_means_system_default():
    for v in (None, "", "default"):
        assert audio.resolve(v) is None


def test_ambiguous_legacy_name_picks_the_best_host_api():
    """The exact failure: this name matches three devices across two APIs."""
    assert audio.resolve("Voicemeeter Input") == 46, "DirectSound outranks WASAPI"


def test_unknown_name_falls_back_to_default_rather_than_raising():
    assert audio.resolve("No Such Device") is None


def test_describe_names_the_host_api():
    assert "MME" in audio.describe_index(5)
    assert audio.describe_index("") == "the system default"
    assert "not found" in audio.describe_index(999)


def test_candidates_carry_unique_indices():
    out = audio._describe([d for d in FAKE if d.outputs], "output")
    assert len({e["index"] for e in out}) == len(out)
    assert all("label" in e and "api" in e for e in out)


def test_best_host_api_wins_per_device():
    """Forty near-identical entries is not a menu. One per device, best API."""
    ins = audio._describe([d for d in FAKE if d.inputs], "input")
    assert len(ins) == len({e["name"] for e in ins}), "one entry per device"
    b2 = next(e for e in ins if "Out B2" in e["name"])
    assert b2["api"] == "MME" and b2["index"] == 5, "MME outranks WASAPI here"
    assert ins[0]["api"] == "MME"


def test_expanded_view_keeps_every_host_api():
    ins = audio._describe([d for d in FAKE if d.inputs], "input", collapse=False)
    assert len(ins) == len([d for d in FAKE if d.inputs])
    assert any(e["api"] == "Windows WASAPI" for e in ins)


def test_duplicate_names_collapse_to_one_entry():
    out = audio._describe([d for d in FAKE if d.outputs], "output")
    same = [e for e in out if e["name"].startswith("Voicemeeter Input")]
    assert len(same) == 1, "the same device must not appear twice"
    assert same[0]["index"] == 46, "DirectSound outranks WASAPI"


def test_expanded_labels_name_the_host_api():
    out = audio._describe([d for d in FAKE if d.outputs], "output", collapse=False)
    same = [e for e in out if e["name"].startswith("Voicemeeter Input")]
    assert len(same) == 2
    assert same[0]["label"] != same[1]["label"]
