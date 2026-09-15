import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quorum import kb as kbmod
from quorum.router import Router, Tier, guard_check, _parse
from quorum.wake import EchoGuard, detect, looks_like_question, strip_wake
from tests.fixtures import KNOWN_MISSES, ROUTER_CASES, WAKE_CASES, WAKE_PHRASE

THRESHOLD = 72


# --------------------------------------------------------------- wake phrase
@pytest.mark.parametrize("text,expected", WAKE_CASES)
def test_wake_detection(text, expected):
    assert detect(text, WAKE_PHRASE, THRESHOLD).matched is expected


@pytest.mark.parametrize("text", KNOWN_MISSES)
def test_documented_wake_misses(text):
    """Pinned so the eval reports them honestly instead of silently."""
    assert not detect(text, WAKE_PHRASE, THRESHOLD).matched


def test_matched_hit_carries_a_numeric_score():
    """Regression: the matched branch once returned the scoring function."""
    hit = detect("AI Kartik, are you working on the analysis?", WAKE_PHRASE)
    assert hit.matched
    assert isinstance(hit.score, int) and THRESHOLD <= hit.score <= 100


def test_wake_strips_phrase():
    hit = detect("AI Kartik, are you working on the analysis?", WAKE_PHRASE)
    assert "kartik" not in hit.utterance.lower()
    assert hit.utterance.startswith("are you working")


def test_wake_strips_midsentence():
    hit = detect("So before we move on, AI Kartik, is the draft ready?", WAKE_PHRASE)
    assert "draft ready" in hit.utterance


def test_question_shape_without_punctuation():
    assert looks_like_question("are you working on it")
    assert looks_like_question("what's the status")
    assert not looks_like_question("go ahead and start")


def test_strip_wake_noop_when_absent():
    assert strip_wake("nothing to see here", WAKE_PHRASE) == "nothing to see here"


# --------------------------------------------------------------- echo guard
def test_echo_guard_catches_own_speech():
    g = EchoGuard()
    g.remember("Yes, I'm working on the competitor analysis, draft due Wednesday.")
    assert g.is_echo("yes im working on the competitor analysis draft due wednesday")


def test_echo_guard_allows_new_speech():
    g = EchoGuard()
    g.remember("Yes, I'm working on the competitor analysis.")
    assert not g.is_echo("Can somebody share their screen please")


def test_echo_guard_ignores_short_fragments():
    g = EchoGuard()
    g.remember("Yes.")
    assert not g.is_echo("Yes.")


# --------------------------------------------------------------- guards
def test_guards_fire_on_expected_categories():
    assert guard_check("can you have it by Monday") == "commitment"
    assert guard_check("do you think we should pivot") == "opinion"
    assert guard_check("what's the budget") == "resource"
    assert guard_check("whose fault was the delay") == "interpersonal"


def test_guards_stay_quiet_on_plain_facts():
    for u in ("who is on the team",
              "what's the status of the competitor analysis",
              "how many competitors are in scope"):
        assert guard_check(u) is None, u


# --------------------------------------------------------------- json parsing
def test_parse_handles_fenced_json():
    d = _parse('```json\n{"tier":"answer","confidence":0.9,'
               '"reason":"in kb","draft":"Yes, it is underway."}\n```')
    assert d.tier is Tier.ANSWER and d.confidence == 0.9


def test_parse_handles_prose_wrapper():
    d = _parse('Sure! {"tier":"escalate","confidence":0.2,'
               '"reason":"opinion","draft":""} Hope that helps.')
    assert d.tier is Tier.ESCALATE


def test_parse_fails_closed():
    assert _parse("I'm not sure how to answer that.").tier is Tier.ESCALATE
    assert _parse("{broken json").tier is Tier.ESCALATE


def test_parse_clamps_confidence():
    assert _parse('{"tier":"answer","confidence":7,"draft":"x"}').confidence == 1.0
    assert _parse('{"tier":"answer","confidence":"nope","draft":"x"}').confidence == 0.0


# --------------------------------------------------------------- router gating
class StubLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return self.payload


class DeadLLM:
    def complete(self, system, user):
        raise ConnectionError("no route to host")


@pytest.fixture
def template_kb(tmp_path):
    p = tmp_path / "kb.md"
    p.write_text(kbmod.TEMPLATE)
    return kbmod.load(p)


def test_guard_short_circuits_before_model(template_kb):
    llm = StubLLM('{"tier":"answer","confidence":1.0,"draft":"Sure, Monday works."}')
    d = Router(llm, template_kb).decide("can you have it by Monday")
    assert d.tier is Tier.ESCALATE and d.guard == "commitment"
    assert llm.calls == 0, "guard must not consult the model"


def test_low_confidence_answer_is_downgraded(template_kb):
    llm = StubLLM('{"tier":"answer","confidence":0.55,"reason":"maybe","draft":"Probably."}')
    d = Router(llm, template_kb).decide("who is on the team")
    assert d.tier is Tier.ESCALATE and "gate:below-threshold" in d.trace


def test_confident_answer_passes(template_kb):
    llm = StubLLM('{"tier":"answer","confidence":0.95,"reason":"in kb",'
                  '"draft":"Kartik and two partners."}')
    d = Router(llm, template_kb).decide("who is on the team")
    assert d.tier is Tier.ANSWER and d.draft


def test_empty_draft_is_downgraded(template_kb):
    llm = StubLLM('{"tier":"answer","confidence":0.99,"draft":"   "}')
    assert Router(llm, template_kb).decide("who is on the team").tier is Tier.ESCALATE


def test_model_outage_fails_closed(template_kb):
    d = Router(DeadLLM(), template_kb).decide("who is on the team")
    assert d.tier is Tier.ESCALATE and d.confidence == 0.0


# --------------------------------------------------------------- kb parsing
def test_kb_parses_template_sections(template_kb):
    assert len(template_kb.facts) == 4
    assert len(template_kb.positions) == 2
    assert len(template_kb.boundaries) == 4
    assert "Kartik" in template_kb.identity


def test_kb_warns_when_empty(tmp_path):
    p = tmp_path / "kb.md"
    p.write_text("# Facts\n")
    w = kbmod.load(p).warnings()
    assert any("escalate everything" in x for x in w)


def test_kb_prompt_labels_the_restate_rule(template_kb):
    assert "never extend" in template_kb.as_prompt()


# --------------------------------------------------- guard coverage of fixtures
def test_guard_precision_on_answerable_fixtures():
    """A guard firing on an answerable question is a false escalation."""
    false_escalations = [
        (u, guard_check(u)) for u, tier, _ in ROUTER_CASES
        if tier == "answer" and guard_check(u)
    ]
    assert not false_escalations, false_escalations


def test_guard_recall_on_guardable_fixtures():
    expected = [(u, n.split(":")[1]) for u, t, n in ROUTER_CASES
                if n.startswith("guard:")]
    missed = [(u, g) for u, g in expected if guard_check(u) != g]
    assert not missed, missed


# ------------------------------------------------- tier / GPU availability
def test_cpu_only_torch_does_not_get_the_cuda_tier(monkeypatch):
    """A GPU being present is not the same as PyTorch being able to use it."""
    from quorum import config

    monkeypatch.setattr(config, "_has_cuda", lambda: (True, 8192))
    monkeypatch.setattr(config, "_torch_sees_cuda",
                        lambda: (False, "PyTorch 2.14.0+cpu cannot use the GPU"))
    t = config.detect_tier()
    assert t.name == "cpu"
    assert "8192MB GPU was found but is unused" in t.note
    assert "cannot use the GPU" in t.note


def test_working_cuda_gets_the_cuda_tier(monkeypatch):
    from quorum import config

    monkeypatch.setattr(config, "_has_cuda", lambda: (True, 8192))
    monkeypatch.setattr(config, "_torch_sees_cuda", lambda: (True, ""))
    assert config.detect_tier().name == "cuda"


def test_small_gpu_falls_back_without_consulting_torch(monkeypatch):
    from quorum import config

    monkeypatch.setattr(config, "_has_cuda", lambda: (True, 4096))
    monkeypatch.setattr(config, "_torch_sees_cuda",
                        lambda: (_ for _ in ()).throw(AssertionError("should not be called")))
    assert config.detect_tier().name in ("cpu", "apple")


def test_tier_override_is_respected(monkeypatch):
    from quorum import config
    assert config.detect_tier("cuda").name == "cuda"
    with pytest.raises(ValueError):
        config.detect_tier("nonsense")


def test_cuda_dll_helper_is_a_noop_off_windows(monkeypatch):
    import os
    from quorum import stt
    monkeypatch.setattr(os, "name", "posix")
    assert stt.add_cuda_dll_dirs() == []


# ------------------------------------------------------- router error detail
def test_router_error_keeps_the_real_message():
    """'model unavailable' alone cannot distinguish a bad key from no network."""
    from quorum import kb as kbmod
    from quorum.router import Router

    class Boom:
        def complete(self, s, u):
            raise RuntimeError("HTTP 401 — the API key was rejected")

    d = Router(Boom(), kbmod.parse(kbmod.TEMPLATE)).decide("who is on the team")
    assert d.tier is Tier.ESCALATE
    assert "401" in d.reason
    assert any("llm-error" in t and "401" in t for t in d.trace)


def test_http_errors_are_translated_to_advice():
    import io
    import urllib.error
    from quorum import backends

    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b"{}"))

    import urllib.request
    orig = urllib.request.urlopen
    urllib.request.urlopen = boom
    try:
        with pytest.raises(RuntimeError) as ei:
            backends._post("https://api.groq.com/x", {}, {})
        assert "401" in str(ei.value) and "key was rejected" in str(ei.value)
    finally:
        urllib.request.urlopen = orig


# --------------------------------------------------- groq model rotation
def test_groq_falls_through_retired_models(monkeypatch):
    """Groq retires model names on a schedule; a stale default must not be fatal."""
    from quorum import backends
    from quorum.config import Settings

    monkeypatch.setattr(backends, "get_secret", lambda n: "gsk_test")
    llm = backends.GroqLLM(Settings(groq_model="llama-3.3-70b-versatile"))

    calls = []

    def fake(model, key, system, user, strict_json=True):
        calls.append(model)
        if model == "llama-3.3-70b-versatile":
            raise RuntimeError("HTTP 403 — the API key is not permitted to use this model")
        return '{"tier":"answer","confidence":1.0,"draft":"hi"}'

    monkeypatch.setattr(llm, "_call", fake)
    assert "hi" in llm.complete("s", "u")
    assert calls[0] == "llama-3.3-70b-versatile"
    assert llm.model == "openai/gpt-oss-120b", "should stick with what worked"


def test_a_real_error_is_not_retried(monkeypatch):
    """A bad key must surface immediately, not after trying four models."""
    from quorum import backends
    from quorum.config import Settings

    monkeypatch.setattr(backends, "get_secret", lambda n: "gsk_test")
    llm = backends.GroqLLM(Settings())
    calls = []

    def fake(model, key, system, user, strict_json=True):
        calls.append(model)
        raise RuntimeError("HTTP 429 — rate limited by the free tier")

    monkeypatch.setattr(llm, "_call", fake)
    with pytest.raises(RuntimeError, match="429"):
        llm.complete("s", "u")
    assert len(calls) == 1


def test_all_models_gone_gives_actionable_advice(monkeypatch):
    from quorum import backends
    from quorum.config import Settings

    monkeypatch.setattr(backends, "get_secret", lambda n: "gsk_test")
    llm = backends.GroqLLM(Settings())
    monkeypatch.setattr(llm, "_call", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("HTTP 404 — model_not_found")))

    with pytest.raises(RuntimeError) as ei:
        llm.complete("s", "u")
    assert "console.groq.com/docs/models" in str(ei.value)


def test_missing_key_is_reported_before_any_call(monkeypatch):
    from quorum import backends
    from quorum.config import Settings

    monkeypatch.setattr(backends, "get_secret", lambda n: None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY missing"):
        backends.GroqLLM(Settings()).complete("s", "u")


def test_requests_identify_themselves(monkeypatch):
    """Default urllib UA is blocked by Cloudflare before reaching the API."""
    import io
    import urllib.request
    from quorum import backends

    seen = {}

    class FakeResp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        seen.update(req.headers)
        return FakeResp(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backends._post("https://api.groq.com/x", {}, {"Authorization": "Bearer k"})

    ua = seen.get("User-agent", "")
    assert ua and "quorum" in ua.lower()
    assert "python-urllib" not in ua.lower()


def test_cloudflare_block_is_not_reported_as_an_auth_problem(monkeypatch):
    import io
    import urllib.error
    import urllib.request
    from quorum import backends

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            "u", 403, "Forbidden", {}, io.BytesIO(b"error code: 1010"))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError) as ei:
        backends._post("https://api.groq.com/x", {}, {})
    msg = str(ei.value)
    assert "Cloudflare" in msg
    assert "key" not in msg.lower(), "must not send the user chasing the API key"


def test_non_json_error_body_is_preserved(monkeypatch):
    import io
    import urllib.error
    import urllib.request
    from quorum import backends

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            "u", 502, "Bad Gateway", {}, io.BytesIO(b"<html>upstream down</html>"))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="upstream down"):
        backends._post("https://api.groq.com/x", {}, {})


def test_json_mode_rejection_retries_without_it(monkeypatch):
    """Reasoning models emit thinking tokens, which Groq's json_object
    validator rejects. The router's parser recovers JSON from prose anyway."""
    from quorum import backends
    from quorum.config import Settings

    monkeypatch.setattr(backends, "get_secret", lambda n: "gsk_test")
    llm = backends.GroqLLM(Settings())
    calls = []

    def fake(model, key, system, user, strict_json=True):
        calls.append(strict_json)
        if strict_json:
            raise RuntimeError("HTTP 400: Failed to validate JSON. Please adjust your prompt.")
        return 'Sure! {"tier":"answer","confidence":0.9,"draft":"hi"}'

    monkeypatch.setattr(llm, "_call", fake)
    assert "hi" in llm.complete("s", "u")
    assert calls == [True, False], "strict first, then relaxed"
    assert llm.strict_json is False, "should stop retrying strict mode"


def test_a_400_unrelated_to_json_is_not_retried(monkeypatch):
    from quorum import backends
    from quorum.config import Settings

    monkeypatch.setattr(backends, "get_secret", lambda n: "gsk_test")
    llm = backends.GroqLLM(Settings())
    calls = []

    def fake(model, key, system, user, strict_json=True):
        calls.append(strict_json)
        raise RuntimeError("HTTP 400: messages must not be empty")

    monkeypatch.setattr(llm, "_call", fake)
    with pytest.raises(RuntimeError, match="must not be empty"):
        llm.complete("s", "u")
    assert calls == [True], "only a JSON-mode failure justifies a retry"


def test_speed_change_invalidates_the_prerendered_cache(tmp_path):
    """A holding line cached at the old rate would keep playing after a change."""
    from quorum.tts import CachedVoice

    class Inner:
        sample_rate = 24000
        clones = True
        enrolled = True
        speed = 1.0

    v = CachedVoice(Inner(), cache_dir=tmp_path)
    slow = v._path("Let me check on that.")
    v.inner.speed = 1.20
    fast = v._path("Let me check on that.")
    assert slow != fast


def test_same_text_and_speed_hits_the_same_cache_entry(tmp_path):
    from quorum.tts import CachedVoice

    class Inner:
        sample_rate = 24000
        clones = True
        enrolled = True
        speed = 1.08

    v = CachedVoice(Inner(), cache_dir=tmp_path)
    assert v._path("hello") == v._path("hello")


def test_removing_a_settings_field_does_not_wipe_saved_values(tmp_path, monkeypatch):
    """Regression: an unknown key raised, the handler swallowed it, and every
    saved setting silently reverted to defaults."""
    import json
    from quorum import config

    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    (tmp_path / "settings.json").write_text(json.dumps({
        "retired_field": "gone",
        "capture_device": "5",
        "wake_phrase": "AI Sam",
    }))
    s = config.Settings.load()
    assert s.capture_device == "5"
    assert s.wake_phrase == "AI Sam"
    assert not hasattr(s, "retired_field")


def test_corrupt_settings_file_falls_back_to_defaults(tmp_path, monkeypatch):
    from quorum import config

    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    (tmp_path / "settings.json").write_text("{ not json")
    assert config.Settings.load().wake_phrase == config.Settings().wake_phrase
