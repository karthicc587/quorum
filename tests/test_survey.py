import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from quorum.survey import (
    PAIRED_KEYS, POST, PRE, Response, SurveyStore, as_markdown,
    construct_means, format_report, paired_shift, report, score,
    straight_lining, _by_key,
)


# ------------------------------------------------------------- instrument
def test_paired_items_are_worded_identically():
    pre, post = _by_key("pre"), _by_key("post")
    for k in PAIRED_KEYS:
        assert k in post, f"{k} missing from the post survey"
        assert pre[k].text == post[k].text, f"{k} wording drifted between halves"
        assert pre[k].reverse == post[k].reverse


def test_reverse_items_exist_in_both_halves():
    assert any(i.reverse for i in PRE)
    assert any(i.reverse for i in POST)


def test_reverse_scoring_inverts():
    rev = next(i for i in POST if i.reverse)
    fwd = next(i for i in POST if not i.reverse and i.kind == "likert")
    assert score(rev, 1) == 5 and score(rev, 5) == 1
    assert score(fwd, 4) == 4


def test_consent_is_the_first_thing_asked():
    assert PRE[0].key == "consent"
    assert PRE[0].options == ["Yes", "No"]


def test_escalation_is_measured_separately_from_the_agent_overall():
    esc = [i for i in POST if i.construct == "escalation"]
    assert len(esc) >= 3
    assert any(i.reverse for i in esc), "needs a reversed item to catch acquiescence"


def test_open_questions_are_few_and_last():
    keys = [i.key for i in POST]
    opens = [i.key for i in POST if i.kind == "text"]
    assert len(opens) == 3
    assert keys[-3:] == opens, "free text belongs at the end"


def test_printable_versions_render():
    for phase in ("pre", "post"):
        md = as_markdown(phase)
        assert "Participant code" in md
        assert "Strongly agree" in md
        assert len(md.splitlines()) > 20


# ---------------------------------------------------------------- scoring
def _pair(pid, pre_ans, post_ans):
    return [Response(pid, "pre", answers=pre_ans),
            Response(pid, "post", answers=post_ans)]


def test_paired_shift_reports_direction():
    rs = _pair("p1", {"A1": 2}, {"A1": 4}) + _pair("p2", {"A1": 3}, {"A1": 4})
    sh = paired_shift(rs)
    assert sh["n"] == 2
    a1 = sh["items"]["A1"]
    assert a1["before"] == 2.5 and a1["after"] == 4.0
    assert a1["shift"] == 1.5 and a1["moved_up"] == 2


def test_shift_on_a_reversed_item_is_direction_corrected():
    """A3 is 'would get in the way' — agreeing less is a favourable shift."""
    rs = _pair("p1", {"A3": 5}, {"A3": 1})
    a3 = paired_shift(rs)["items"]["A3"]
    assert a3["before"] == 1 and a3["after"] == 5
    assert a3["shift"] == 4, "less agreement with a negative statement is positive"


def test_unpaired_participants_are_excluded():
    rs = [Response("only_pre", "pre", answers={"A1": 3}),
          Response("only_post", "post", answers={"A1": 5})]
    assert paired_shift(rs)["n"] == 0
    assert "note" in paired_shift(rs)


def test_missing_answers_do_not_break_pairing():
    rs = _pair("p1", {"A1": 2}, {"A2": 4})       # no overlap on any single item
    sh = paired_shift(rs)
    assert sh["n"] == 1
    assert "A1" not in sh["items"] and "A2" not in sh["items"]


def test_construct_means_group_correctly():
    rs = [Response("p1", "post", answers={"B1": 5, "B2": 1, "B4": 4})]
    m = construct_means(rs, "post")
    assert m["escalation"] == 5.0, "B2 reversed: 1 becomes 5"
    assert m["bounds"] == 4.0


def test_straight_lining_flagged():
    flat = Response("p1", "post", answers={"A1": 4, "A2": 4, "A3": 4, "A4": 4, "B1": 4})
    varied = Response("p2", "post", answers={"A1": 4, "A2": 5, "A3": 2, "A4": 3, "B1": 4})
    assert straight_lining(flat)
    assert not straight_lining(varied)


def test_straight_lining_needs_enough_items():
    assert not straight_lining(Response("p1", "post", answers={"A1": 4, "A2": 4}))


def test_report_surfaces_the_headline_numbers():
    rs = _pair("p1", {"A1": 2, "A3": 5}, {"A1": 4, "A3": 2, "B1": 5, "B2": 2,
                                          "B3": 1, "B4": 5, "C1": 5, "C2": 1,
                                          "C3": "The name label",
                                          "E1": "It never interrupted anyone."})
    r = report(rs)
    assert r["n_pre"] == 1 and r["n_post"] == 1
    assert r["escalation_accepted"] is not None
    assert r["stayed_in_bounds"] == 5.0
    assert r["disclosure_salient"] == 5.0
    assert r["first_noticed"] == {"The name label": 1}
    assert r["open_responses"][0]["E1"].startswith("It never")


def test_report_on_no_data_does_not_crash():
    r = report([])
    assert r["n_post"] == 0
    assert r["escalation_accepted"] is None
    assert "note" in r["shift"]
    assert format_report(r)


def test_formatted_report_is_readable():
    rs = _pair("p1", {"A1": 2}, {"A1": 5, "B1": 4, "B4": 4, "C1": 5})
    out = format_report(report(rs))
    assert "ATTITUDE SHIFT" in out and "BOUNDED AUTONOMY" in out
    assert "+3.00" in out


# ---------------------------------------------------------------- storage
def test_store_round_trip(tmp_path):
    s = SurveyStore(tmp_path)
    s.write(Response("p1", "pre", answers={"A1": 3}))
    s.write(Response("p1", "post", answers={"A1": 5}))
    rs = s.read()
    assert len(rs) == 2 and rs[1].answers["A1"] == 5


def test_store_tolerates_a_truncated_line(tmp_path):
    s = SurveyStore(tmp_path)
    s.write(Response("p1", "pre", answers={"A1": 3}))
    with s.path.open("a") as f:
        f.write('{"participant": "cut')
    assert len(s.read()) == 1


def test_participant_codes_are_unique_and_short():
    codes = {Response.new_participant() for _ in range(200)}
    assert len(codes) == 200
    assert all(len(c) == 6 for c in codes)
