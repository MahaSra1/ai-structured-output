"""Day 11 tests: no Ollama needed. Run: python -m pytest -q"""
import json

import pytest
from pydantic import ValidationError

from extractor import clean_json, extract
from ticket_schema import IssueType, Ticket

GOOD = {"customer_name": "Priya", "tracking_number": "SP1002",
        "issues": ["damaged", "refund"], "customer_upset": True,
        "summary": "Parcel arrived smashed, wants refund."}
MSG = "Hi, I'm Priya. Parcel SP1002 arrived smashed, I want my money back!"


def make(**changes):
    return {**GOOD, **changes}


# ---------- schema ----------
def test_good_ticket_passes():
    t = Ticket.model_validate(GOOD, context={"message": MSG})
    assert t.issues == [IssueType.DAMAGED, IssueType.REFUND]


def test_tracking_is_tidied():
    assert Ticket.model_validate(make(tracking_number=" sp 1002 ")).tracking_number == "SP1002"


@pytest.mark.parametrize("v", ["null", "N/A", "", None])
def test_tracking_placeholders_become_none(v):
    assert Ticket.model_validate(make(tracking_number=v)).tracking_number is None


def test_bad_tracking_format_fails():
    with pytest.raises(ValidationError, match="SP1234"):
        Ticket.model_validate(make(tracking_number="XY12"))


def test_invented_tracking_fails_grounding():
    with pytest.raises(ValidationError, match="not in the customer message"):
        Ticket.model_validate(make(tracking_number="SP9999"), context={"message": MSG})


def test_missed_tracking_fails_completeness():
    with pytest.raises(ValidationError, match="returned null"):
        Ticket.model_validate(make(tracking_number=None), context={"message": MSG})


def test_missed_tracking_with_spaces_detected():
    with pytest.raises(ValidationError, match="SP3301"):
        Ticket.model_validate(make(tracking_number=None),
                              context={"message": "Tracking sp 3301 please"})


def test_null_ok_when_no_tracking_in_message():
    t = Ticket.model_validate(make(tracking_number=None),
                              context={"message": "Where is my parcel?"})
    assert t.tracking_number is None


def test_missing_tracking_key_fails():
    data = {k: v for k, v in GOOD.items() if k != "tracking_number"}
    with pytest.raises(ValidationError, match="tracking_number"):
        Ticket.model_validate(data, context={"message": MSG})


def test_all_fields_required_in_schema():
    assert set(Ticket.model_json_schema()["required"]) == set(Ticket.model_fields)


def test_unknown_issue_fails():
    with pytest.raises(ValidationError):
        Ticket.model_validate(make(issues=["stolen"]))


def test_empty_issues_fails():
    with pytest.raises(ValidationError):
        Ticket.model_validate(make(issues=[]))


def test_unclear_must_be_alone():
    with pytest.raises(ValidationError, match="unclear"):
        Ticket.model_validate(make(issues=["unclear", "late"]))


def test_duplicate_issues_removed():
    assert Ticket.model_validate(make(issues=["late", "late"])).issues == [IssueType.LATE]


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        Ticket.model_validate(make(priority="high"))


# ---------- clean_json ----------
def test_clean_json_strips_fences_and_prose():
    raw = "Sure! Here you go:\n```json\n{\"a\": 1}\n```\nHope that helps."
    assert clean_json(raw) == '{"a": 1}'


# ---------- retry loop with a fake LLM ----------
def fake_llm(replies):
    calls = []

    def chat(messages, model, fmt):
        calls.append(messages)
        return {"content": replies[len(calls) - 1]}
    return chat, calls


def test_valid_first_time():
    chat, _ = fake_llm([json.dumps(GOOD)])
    r = extract(MSG, chat_fn=chat)
    assert r.status == "valid" and len(r.attempts) == 1


def test_fixed_by_retry_and_error_sent_back():
    chat, calls = fake_llm([json.dumps(make(tracking_number="SP9999")), json.dumps(GOOD)])
    r = extract(MSG, chat_fn=chat)
    assert r.status == "fixed" and len(r.attempts) == 2
    assert "not in the customer message" in calls[1][-1]["content"]


def test_rejected_after_max_retries():
    chat, calls = fake_llm(["not json"] * 3)
    r = extract(MSG, max_retries=2, chat_fn=chat)
    assert r.status == "rejected" and len(calls) == 3
    assert "json" in r.attempts[0].error.lower()
