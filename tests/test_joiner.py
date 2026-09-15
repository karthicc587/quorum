"""Joiner logic. The browser parts need a real machine; everything here is
URL handling, state, and the disclosure path — which is the part that must
not silently no-op.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quorum.joiner import (
    CHROME_ARGS, JoinState, Joiner, MEET_SELECTORS, Platform, ZOOM_SELECTORS,
    classify, normalize,
)


# ------------------------------------------------------------------- urls
@pytest.mark.parametrize("url,platform,mid", [
    ("https://meet.google.com/abc-defg-hij", Platform.MEET, "abc-defg-hij"),
    ("meet.google.com/abc-defg-hij", Platform.MEET, "abc-defg-hij"),
    ("https://meet.google.com/abc-defg-hij?authuser=1", Platform.MEET, "abc-defg-hij"),
    ("https://us02web.zoom.us/j/84512345678", Platform.ZOOM, "84512345678"),
    ("https://zoom.us/w/12345", Platform.ZOOM, "12345"),
    ("https://teams.microsoft.com/l/meetup", Platform.UNKNOWN, None),
    ("", Platform.UNKNOWN, None),
    ("not a url", Platform.UNKNOWN, None),
])
def test_classify(url, platform, mid):
    p, m = classify(normalize(url))
    assert p is platform and m == mid


def test_normalize_adds_scheme_and_strips_meet_params():
    assert normalize("meet.google.com/abc-defg-hij").startswith("https://")
    assert "?" not in normalize("https://meet.google.com/abc-defg-hij?hs=1")


def test_normalize_keeps_zoom_query_which_carries_the_passcode():
    url = "https://us02web.zoom.us/j/8451234?pwd=Secret"
    assert "pwd=Secret" in normalize(url), "stripping this locks the agent out"


# --------------------------------------------------------------- preflight
def test_join_refuses_an_unknown_platform(tmp_path):
    r = Joiner("AI Agent", profile_dir=tmp_path).join("https://teams.microsoft.com/x")
    assert r.state is JoinState.FAILED
    assert not r.ok
    assert "Meet or Zoom" in r.detail


def test_join_refuses_meet_without_a_saved_session(tmp_path):
    """Better to say so than to open a browser and stall on a login page."""
    j = Joiner("AI Agent", profile_dir=tmp_path)
    assert not j.logged_in
    r = j.join("https://meet.google.com/abc-defg-hij")
    assert r.state is JoinState.FAILED
    assert "quorum.joiner login" in r.detail


@pytest.mark.parametrize("rel", [
    "Default/Network/Cookies",      # current Chrome
    "Default/Cookies",              # older Chrome
    "Network/Cookies",
    "Cookies",
])
def test_cookie_jar_found_wherever_chrome_puts_it(tmp_path, rel):
    """Chrome has moved this file between releases; checking one path made a
    perfectly good session report as missing."""
    j = Joiner("AI Agent", profile_dir=tmp_path)
    assert not j.logged_in
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * 2048)
    assert j.logged_in
    assert "Session found" in j.session_report()


def test_empty_cookie_file_is_not_a_session(tmp_path):
    j = Joiner("AI Agent", profile_dir=tmp_path)
    p = tmp_path / "Default" / "Network" / "Cookies"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"")
    assert not j.logged_in


def test_session_report_explains_a_missing_profile(tmp_path):
    j = Joiner("AI Agent", profile_dir=tmp_path / "nope")
    assert "quorum.joiner login" in j.session_report()


def test_session_report_lists_contents_when_cookies_are_absent(tmp_path):
    (tmp_path / "Default").mkdir()
    j = Joiner("AI Agent", profile_dir=tmp_path)
    r = j.session_report()
    assert "no cookies" in r and "Default" in r


def test_no_credentials_are_accepted_anywhere():
    """The persistent profile is the only auth path.

    Checked on the signature rather than the source text, so the docstring
    explaining that passwords are not stored does not fail its own test.
    """
    import inspect
    params = set(inspect.signature(Joiner.__init__).parameters)
    assert not {"password", "passwd", "email", "username", "credentials"} & params
    j = Joiner("AI Agent")
    assert not any("pass" in a.lower() or "cred" in a.lower() for a in vars(j))


# ---------------------------------------------------------------- browser
def test_automation_banner_is_suppressed_not_fingerprint_spoofed():
    """Suppressing the infobar keeps selectors stable. Nothing here should be
    impersonating a different browser or defeating an integrity check."""
    joined = " ".join(CHROME_ARGS)
    assert "--disable-blink-features=AutomationControlled" in joined
    assert "user-agent" not in joined.lower()


def test_media_permission_is_pre_granted():
    assert "--use-fake-ui-for-media-stream" in CHROME_ARGS


def test_every_flow_has_selectors_for_each_step():
    for name, sel in (("meet", MEET_SELECTORS), ("zoom", ZOOM_SELECTORS)):
        for step in ("name_field", "join", "in_call", "lobby", "chat_open", "chat_box"):
            assert sel.get(step), f"{name} missing {step}"
            assert len(sel[step]) >= 1


# ------------------------------------------------------------- disclosure
class FakePage:
    """Minimal Playwright stand-in for the disclosure path."""

    def __init__(self, missing=()):
        self.missing = set(missing)
        self.typed = []
        self.pressed = []

    def locator(self, sel):
        page = self

        class L:
            first = None

            def wait_for(self, state=None, timeout=None):
                # Case-insensitive: selectors mix "Chat with everyone" and
                # "chat", and a case-sensitive fake silently passes.
                if any(m.lower() in sel.lower() for m in page.missing):
                    raise RuntimeError("not found")

            def click(self): pass
            def fill(self, text): page.typed.append(text)

        el = L()
        el.first = el
        return el

    def wait_for_timeout(self, ms): pass

    class keyboard:
        @staticmethod
        def press(k): pass


def test_disclosure_is_posted_when_chat_is_reachable():
    j = Joiner("AI Agent — Kartik", disclosure="Heads up: AI delegate here.")
    page, log = FakePage(), []
    j._post_disclosure(page, MEET_SELECTORS, log)
    assert page.typed == ["Heads up: AI delegate here."]
    assert any("disclosure posted" in x for x in log)


def test_missing_chat_box_is_also_loud():
    j = Joiner("AI Agent — Kartik", disclosure="Heads up.")
    page, log = FakePage(missing=["Send a message", "message"]), []
    j._post_disclosure(page, MEET_SELECTORS, log)
    assert page.typed == []
    assert any("NOT posted" in x for x in log)


def test_failure_to_post_the_disclosure_is_loud():
    """Silently joining undisclosed is the one outcome that must be visible."""
    j = Joiner("AI Agent — Kartik", disclosure="Heads up: AI delegate here.")
    log = []
    j._post_disclosure(FakePage(missing=["chat"]), MEET_SELECTORS, log)
    assert any("NOT posted" in x for x in log)


def test_leave_is_safe_before_any_join():
    j = Joiner("AI Agent")
    j.leave()
    assert j.state is JoinState.LEFT
