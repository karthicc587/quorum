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
