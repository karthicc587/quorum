"""Which HTTP client can actually reach Groq from this machine?

Cloudflare's 1010 block can key on either the User-Agent string or the TLS
fingerprint of the client library. A header fix cures the first, not the
second, so test both before changing the app.
"""
import json

from quorum.config import get_secret, redact

key = get_secret("GROQ_API_KEY")
print("key:", redact(key) if key else "NO KEY FOUND")
if not key:
    raise SystemExit("Nothing in GROQ_API_KEY.")

URL = "https://api.groq.com/openai/v1/chat/completions"
PAYLOAD = {"model": "openai/gpt-oss-20b",
           "messages": [{"role": "user", "content": "say ok"}],
           "max_tokens": 5}
UA = "quorum.ai/0.1 (meeting delegate)"


def show(label, status, body):
    verdict = "WORKS" if status == 200 else "fails"
    note = ""
    if body and "1010" in body:
        note = "   <- Cloudflare signature block, not an auth problem"
    print(f"  {verdict}  {label:34} status={status}  {str(body)[:70]}{note}")


# 1. urllib, default headers — what failed before
import urllib.error
import urllib.request


def try_urllib(label, extra):
    req = urllib.request.Request(
        URL, data=json.dumps(PAYLOAD).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json", **extra})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            show(label, r.status, r.read().decode()[:70])
            return True
    except urllib.error.HTTPError as e:
        show(label, e.code, e.read().decode())
    except Exception as e:
        show(label, "err", f"{type(e).__name__}: {e}")
    return False


print("\nprobing:")
ok_plain = try_urllib("urllib, default UA", {})
ok_ua = try_urllib("urllib + User-Agent", {"User-Agent": UA})

# 2. httpx — different TLS stack entirely
ok_httpx = False
try:
    import httpx
    try:
        r = httpx.post(URL, json=PAYLOAD, timeout=20,
                       headers={"Authorization": f"Bearer {key}",
                                "User-Agent": UA})
        show("httpx + User-Agent", r.status_code, r.text)
        ok_httpx = r.status_code == 200
    except Exception as e:
        show("httpx + User-Agent", "err", f"{type(e).__name__}: {e}")
except ImportError:
    print("  skipped httpx (pip install httpx)")

print()
if ok_ua:
    print("Fix confirmed: the User-Agent header is enough. The app already sends it.")
elif ok_httpx:
    print("urllib is blocked on its TLS fingerprint, not its headers.")
    print("The app needs to move to httpx. Tell Claude this line.")
elif not (ok_plain or ok_ua or ok_httpx):
    print("Every client is blocked. That points at the network rather than the")
    print("code — a campus, VPN or DNS filter sitting in front of Groq.")
    print("Try a phone hotspot: if it works there, it is the network.")
