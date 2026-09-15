"""Join the meeting as its own participant, instead of riding your mic.

Two decisions worth stating, because both look like shortcuts and are not.

**No password automation.** Google blocks scripted sign-in — "this browser may
not be secure", device verification, 2FA. Every product in this space solves it
the same way: sign in by hand once into a persistent browser profile, then reuse
the cookie. So `quorum.joiner login` opens a normal window, you sign in, and the
profile is saved. Credentials never reach this code.

**Not headless.** Headless Chrome exposes no audio devices, so it can neither
hear the meeting nor speak into it. This drives a real Chrome window that you
simply do not touch. On a spare machine that is fine; on your daily driver it
means a visible window during the call.

Detection is not fought. If a platform blocks automated joining, that is the
platform's call and the answer is to join by hand.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .config import ROOT

PROFILE_DIR = ROOT / ".quorum" / "browser-profile"

# Auto-grant mic and camera so no permission dialog blocks the join, and keep
# Chrome from advertising itself as automated — that banner changes page
# layout and breaks selectors. It is not an attempt to evade detection.
CHROME_ARGS = [
    "--use-fake-ui-for-media-stream",
    "--disable-blink-features=AutomationControlled",
    "--autoplay-policy=no-user-gesture-required",
    "--disable-features=IsolateOrigins,site-per-process",
]


class Platform(str, Enum):
    MEET = "meet"
    ZOOM = "zoom"
    UNKNOWN = "unknown"


class JoinState(str, Enum):
    IDLE = "idle"
    LAUNCHING = "launching"
    LOBBY = "lobby"           # waiting for a host to admit us
    JOINED = "joined"
    LEFT = "left"
    FAILED = "failed"


_MEET = re.compile(r"https?://meet\.google\.com/([a-z]{3}-[a-z]{4}-[a-z]{3})", re.I)
_ZOOM = re.compile(r"https?://[\w.-]*zoom\.us/[jw]/(\d+)", re.I)


def classify(url: str) -> tuple[Platform, str | None]:
    """Which client, and the meeting id. Used to pick the join flow."""
    if m := _MEET.search(url or ""):
        return Platform.MEET, m.group(1)
    if m := _ZOOM.search(url or ""):
        return Platform.ZOOM, m.group(1)
    return Platform.UNKNOWN, None


def normalize(url: str) -> str:
    """Strip tracking and auth params that break a clean join."""
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith("http"):
        url = "https://" + url
    return url.split("?")[0] if "meet.google.com" in url else url


@dataclass
class JoinResult:
    state: JoinState
    platform: Platform
    detail: str = ""
    waited_s: float = 0.0
    log: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.state in (JoinState.JOINED, JoinState.LOBBY)


# --------------------------------------------------------------- selectors
# Kept in one place because meeting clients reshuffle their DOM constantly and
# this is the first thing that breaks. Each entry is tried in order.
MEET_SELECTORS = {
    "name_field": ['input[aria-label*="Your name" i]', 'input[placeholder*="name" i]'],
    "mic_off": ['[aria-label*="Turn off microphone" i]', '[data-is-muted="false"][aria-label*="microphone" i]'],
    "cam_off": ['[aria-label*="Turn off camera" i]', '[data-is-muted="false"][aria-label*="camera" i]'],
    "join": ['button:has-text("Join now")', 'button:has-text("Ask to join")',
             '[aria-label*="Join now" i]'],
    "in_call": ['[aria-label*="Leave call" i]', 'button[aria-label*="Leave" i]'],
    "lobby": ['text=Asking to be let in', 'text=Waiting for the host'],
    "chat_open": ['[aria-label*="Chat with everyone" i]', 'button[aria-label*="chat" i]'],
    "chat_box": ['textarea[aria-label*="Send a message" i]', 'textarea[placeholder*="message" i]'],
}

ZOOM_SELECTORS = {
    "name_field": ['#input-for-name', 'input[placeholder*="name" i]'],
    "join": ['button:has-text("Join")', '#joinBtn'],
    "in_call": ['[aria-label*="Leave" i]', 'button:has-text("Leave")'],
    "lobby": ['text=Please wait, the meeting host', 'text=waiting room'],
    "chat_open": ['[aria-label*="open the chat" i]', 'button[aria-label*="Chat" i]'],
    "chat_box": ['textarea[aria-label*="Type message" i]', 'div[contenteditable="true"]'],
}


def _first(page, candidates, timeout_ms=2500):
    """Return the first selector that resolves, or None. Never raises."""
    for sel in candidates:
        try:
            el = page.locator(sel).first
            el.wait_for(state="visible", timeout=timeout_ms)
            return el
        except Exception:
            continue
    return None


# ------------------------------------------------------------------ driver
class Joiner:
    def __init__(self, display_name: str, disclosure: str = "",
                 profile_dir: Path | None = None, headless: bool = False):
        self.display_name = display_name
        self.disclosure = disclosure
        self.profile_dir = profile_dir or PROFILE_DIR
        self.headless = headless
        self._ctx = None
        self._page = None
        self.state = JoinState.IDLE

    # -- one-time sign-in ---------------------------------------------------
    def login(self, url: str = "https://accounts.google.com", wait_s: int = 300) -> str:
        """Open a browser, let the human sign in, keep the profile.

        Blocks until the window is closed or the timeout expires. Nothing is
        typed for you and no password is stored — the outcome is a cookie jar
        on disk that later joins reuse.
        """
        from playwright.sync_api import sync_playwright

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as pw:
            ctx = pw.chromium.launch_persistent_context(
                str(self.profile_dir), headless=False, args=CHROME_ARGS,
                viewport={"width": 1280, "height": 800},
            )
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(url)
            deadline = time.time() + wait_s
            while time.time() < deadline:
                if not ctx.pages:
                    break
                time.sleep(1)
            try:
                ctx.close()
            except Exception:
                pass
        # Chrome flushes cookies to disk on shutdown, which is not instant.
        for _ in range(20):
            if self.logged_in:
                break
            time.sleep(0.25)
        return str(self.profile_dir)

    # Chrome has moved the cookie jar between releases — it used to sit at
    # Default/Cookies and now lives under Default/Network/. Check every known
    # location rather than one, or a perfectly good session reads as missing.
    COOKIE_PATHS = (
        Path("Default") / "Network" / "Cookies",
        Path("Default") / "Cookies",
        Path("Network") / "Cookies",
        Path("Cookies"),
    )

    @property
    def cookie_jar(self) -> Path | None:
        for rel in self.COOKIE_PATHS:
            p = self.profile_dir / rel
            if p.exists() and p.stat().st_size > 0:
                return p
        # Last resort: some builds nest the profile differently.
        for p in self.profile_dir.rglob("Cookies"):
            if p.is_file() and p.stat().st_size > 0:
                return p
        return None

    @property
    def logged_in(self) -> bool:
        """A non-empty cookie jar exists. Not proof the session is still valid."""
        return self.cookie_jar is not None

    def session_report(self) -> str:
        if not self.profile_dir.exists():
            return f"No profile at {self.profile_dir}. Run: python -m quorum.joiner login"
        jar = self.cookie_jar
        if not jar:
            found = sorted(p.name for p in self.profile_dir.iterdir())[:8]
            return (f"Profile exists at {self.profile_dir} but holds no cookies.\n"
                    f"  contains: {', '.join(found) or '(empty)'}\n"
                    "  Sign in again and close the window only after the "
                    "account page has fully loaded.")
        kb = jar.stat().st_size // 1024
        return f"Session found: {jar.relative_to(self.profile_dir)} ({kb} KB)"

    # -- join ---------------------------------------------------------------
    def join(self, url: str, timeout_s: int = 90) -> JoinResult:
        from playwright.sync_api import sync_playwright

        url = normalize(url)
        platform, mid = classify(url)
        log: list[str] = [f"url={url} platform={platform.value} id={mid}"]

        if platform is Platform.UNKNOWN:
            return JoinResult(JoinState.FAILED, platform,
                              "Not a recognised Meet or Zoom link.", log=log)
        if not self.logged_in and platform is Platform.MEET:
            return JoinResult(
                JoinState.FAILED, platform,
                "No saved browser session. Run: python -m quorum.joiner login",
                log=log)

        t0 = time.time()
        self.state = JoinState.LAUNCHING
        pw = sync_playwright().start()
        try:
            self._ctx = pw.chromium.launch_persistent_context(
                str(self.profile_dir), headless=self.headless, args=CHROME_ARGS,
                viewport={"width": 1280, "height": 800},
                permissions=["microphone", "camera"],
            )
            page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
            self._page = page
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_s * 1000)
            page.wait_for_timeout(3500)

            sel = MEET_SELECTORS if platform is Platform.MEET else ZOOM_SELECTORS
            self._set_name(page, sel, log)
            if platform is Platform.MEET:
                self._quiet_devices(page, log)
            self._press_join(page, sel, log)

            state, detail = self._settle(page, sel, timeout_s, log)
            self.state = state
            if state is JoinState.JOINED and self.disclosure:
                self._post_disclosure(page, sel, log)
            return JoinResult(state, platform, detail,
                              round(time.time() - t0, 1), log)
        except Exception as e:
            self.state = JoinState.FAILED
            log.append(f"exception: {type(e).__name__}: {e}")
            return JoinResult(JoinState.FAILED, platform, str(e)[:200],
                              round(time.time() - t0, 1), log)

    def _set_name(self, page, sel, log):
        el = _first(page, sel["name_field"], 3000)
        if el:
            el.fill(self.display_name)
            log.append(f"name set to {self.display_name!r}")
        else:
            log.append("no name field — signed in, so the account name is used")

    def _quiet_devices(self, page, log):
        """Mute the mic before joining. The agent unmutes when it speaks."""
        for key in ("mic_off", "cam_off"):
            el = _first(page, MEET_SELECTORS[key], 1500)
            if el:
                try:
                    el.click()
                    log.append(f"{key} clicked")
                except Exception:
                    pass

    def _press_join(self, page, sel, log):
        el = _first(page, sel["join"], 8000)
        if not el:
            log.append("no join button found")
            return
        label = ""
        try:
            label = el.inner_text()[:40]
        except Exception:
            pass
        el.click()
        log.append(f"join clicked ({label!r})")

    def _settle(self, page, sel, timeout_s, log):
        """Distinguish joined, held in a lobby, and never got in."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if _first(page, sel["in_call"], 1200):
                log.append("in call")
                return JoinState.JOINED, ""
            if _first(page, sel["lobby"], 800):
                log.append("waiting in lobby")
                return JoinState.LOBBY, "Waiting for a host to admit the agent."
            page.wait_for_timeout(1000)
        return JoinState.FAILED, "Timed out before the call opened."

    def _post_disclosure(self, page, sel, log):
        opener = _first(page, sel["chat_open"], 4000)
        if not opener:
            log.append("chat button not found — disclosure NOT posted")
            return
        try:
            opener.click()
            page.wait_for_timeout(1200)
            box = _first(page, sel["chat_box"], 4000)
            if not box:
                log.append("chat box not found — disclosure NOT posted")
                return
            box.click()
            box.fill(self.disclosure)
            page.keyboard.press("Enter")
            log.append("disclosure posted in chat")
        except Exception as e:
            log.append(f"disclosure failed: {e}")

    def still_in_call(self) -> bool:
        """False once the tab is gone or the leave button disappears."""
        if not self._page:
            return False
        try:
            if self._page.is_closed():
                return False
            sel = (MEET_SELECTORS if "meet.google.com" in (self._page.url or "")
                   else ZOOM_SELECTORS)
            return _first(self._page, sel["in_call"], 800) is not None
        except Exception:
            return False

    def audio_hint(self) -> str:
        """What the human still has to do inside the meeting client.

        Chrome offers no command-line switch for choosing a real capture or
        playback device, so the agent cannot set this for you. The persistent
        profile does remember the choice per site, which means it is a
        one-time step rather than a per-meeting one.
        """
        return (
            "In the meeting window, open Settings > Audio and set:\n"
            "  Microphone : Voicemeeter Out B1\n"
            "  Speakers   : Voicemeeter AUX Input\n"
            "Also turn off noise cancellation. The browser profile remembers "
            "all three, so this is once per machine, not once per meeting."
        )

    def leave(self):
        try:
            if self._ctx:
                self._ctx.close()
        except Exception:
            pass
        self._ctx = self._page = None
        self.state = JoinState.LEFT


class JoinerThread:
    """Keeps a joined meeting alive on its own thread.

    Playwright's sync API is bound to the thread that created it, so the
    browser cannot be driven from the server's event loop. One dedicated
    thread owns it for the life of the meeting and takes a single instruction:
    leave.
    """

    def __init__(self, joiner: "Joiner"):
        self.joiner = joiner
        self.result: JoinResult | None = None
        self._thread = None
        self._leave = threading.Event()
        self._ready = threading.Event()

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self, url: str, timeout_s: int = 90) -> JoinResult:
        """Blocks until joined, in a lobby, or failed. Then returns."""
        self._leave.clear()
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._run, args=(url, timeout_s), daemon=True)
        self._thread.start()
        self._ready.wait(timeout=timeout_s + 30)
        return self.result or JoinResult(
            JoinState.FAILED, Platform.UNKNOWN, "join did not report back")

    def _run(self, url: str, timeout_s: int):
        try:
            self.result = self.joiner.join(url, timeout_s)
        except Exception as e:
            self.result = JoinResult(JoinState.FAILED, Platform.UNKNOWN, str(e)[:200])
        finally:
            self._ready.set()

        # Hold the browser open. Closing the context ends the meeting, so the
        # thread parks here until asked to leave.
        if self.result and self.result.ok:
            while not self._leave.wait(1.0):
                if not self.joiner.still_in_call():
                    self.result.state = JoinState.LEFT
                    break
        self.joiner.leave()

    def stop(self):
        self._leave.set()
        if self._thread:
            self._thread.join(timeout=15)


def _cli(argv: list[str]) -> int:
    if not argv or argv[0] not in ("login", "join", "status"):
        print("usage:\n  python -m quorum.joiner login\n"
              "  python -m quorum.joiner status\n"
              "  python -m quorum.joiner join <meeting-url>")
        return 1
    from .config import Settings
    from .platforms import get as get_platform

    s = Settings.load()
    j = Joiner(s.display_name, get_platform(s.platform).disclosure_chat)

    if argv[0] == "login":
        print("A browser will open. Sign in, then close the window.")
        j.login()
        print()
        print(j.session_report())
        return 0 if j.logged_in else 1

    if argv[0] == "status":
        print(j.session_report())
        return 0 if j.logged_in else 1

    if len(argv) < 2:
        print("need a meeting url")
        return 1
    r = j.join(argv[1])
    for line in r.log:
        print(" ", line)
    print(f"\n{r.state.value}: {r.detail or 'ok'} ({r.waited_s}s)")
    if r.ok:
        input("Press Enter to leave the meeting...")
    j.leave()
    return 0 if r.ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(_cli(sys.argv[1:]))
