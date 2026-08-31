"""Defensive parsing of model output.

Each case here is a failure mode observed from real models, not an invented
one. The suite's real assertion is the last group: **when the text cannot be
recovered, parsing fails rather than returning a partly-filled object.** A
`contract_rate` silently defaulted to 0.0 would flow into the arithmetic check
and produce a confident, precise, wrong statement about public money -- worse
than an error, because nothing downstream would mark it as suspect.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.llm.parsing import coerce_number, extract_json, parse_model
from app.models import ClauseInterpretation
from app.core.interpret import _normalise_interpretation


class Tiny(BaseModel):
    a: int
    b: str = ""


# --- recovering JSON from whatever the model actually emitted ---------------

@pytest.mark.parametrize(
    "text, expected_strategy",
    [
        ('{"a": 1}', "direct"),
        ('```json\n{"a": 1}\n```', "fenced"),
        ('```\n{"a": 1}\n```', "fenced"),
        ('Here is the analysis:\n{"a": 1}\nHope that helps!', "balanced-scan"),
        ('{"a": 1,}', "repaired"),
        ('{"a": 1, "b": None}', "repaired"),
        ('{"a": 1, "b": "x"', "closed-truncated"),
    ],
)
def test_recovers_json_from_common_model_output(text, expected_strategy):
    value, strategy, _ = extract_json(text)
    assert value is not None, f"failed to recover from {text!r}"
    assert strategy == expected_strategy
    assert value["a"] == 1


def test_braces_inside_strings_do_not_confuse_the_scanner():
    """The balanced-block scan must respect string literals and escapes.

    A reasoning field is free text and routinely contains braces or quotes;
    a naive brace counter truncates the object at the first one.
    """
    text = 'Analysis: {"a": 1, "b": "a clause with a } brace and a \\" quote"} done'
    value, _, _ = extract_json(text)
    assert value["a"] == 1
    assert "}" in value["b"] and '"' in value["b"]


def test_repairs_are_reported_not_hidden():
    """Repairs are traced so the repair rate is measurable rather than guessed."""
    result = parse_model('```json\n{"a": 1,}\n```', Tiny)
    assert result.ok
    assert result.was_repaired
    assert any("trailing comma" in r for r in result.repairs)


# --- numbers the way contracts and models actually write them --------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        (1150, 1150.0),
        (1150.5, 1150.5),
        ("1150", 1150.0),
        ("1 150", 1150.0),            # Swedish thousands separator: a space
        ("1\u00a0150", 1150.0),       # ...often a non-breaking space
        ("1 150,50", 1150.5),         # decimal comma
        ("1,150.50", 1150.5),         # anglicised: comma groups, dot decimal
        ("1.150,50", 1150.5),         # continental: dot groups, comma decimal
        ("1150,50", 1150.5),          # bare decimal comma
        ("1,150", 1150.0),            # comma as a thousands group, not a decimal
        ("1.150", 1150.0),            # dot as a thousands group
        ("1 234 567,89", 1234567.89),
        ("SEK 1 150 per hour", 1150.0),  # the contract's own phrasing echoed back
        ("-500", -500.0),
        ("null", None),
        ("n/a", None),
        ("", None),
        (None, None),
        (True, None),                 # bool is an int in Python; not a rate
    ],
)
def test_coerce_number_handles_contract_formatting(raw, expected):
    assert coerce_number(raw) == expected


def test_a_correctly_read_rate_survives_bad_formatting():
    """Format errors must not be treated as reading errors.

    A model answering "SEK 1 150 per hour" has the rate right and the schema
    wrong. Rejecting it would throw away a correct reading and send a
    checkable line to a human for no reason.
    """
    text = '{"line_no": 1, "governing_clause_id": "C-1-§4.2", '\
           '"contract_rate": "SEK 1 150 per hour", "confidence": "0.8"}'
    result = parse_model(text, ClauseInterpretation, normalise=_normalise_interpretation)
    assert result.ok
    assert result.value.contract_rate == 1150.0
    assert result.value.confidence == 0.8


def test_string_null_becomes_a_real_null():
    text = '{"line_no": 1, "governing_clause_id": "null", "contract_rate": "unknown"}'
    result = parse_model(text, ClauseInterpretation, normalise=_normalise_interpretation)
    assert result.ok
    assert result.value.governing_clause_id is None
    assert result.value.contract_rate is None


# --- failing closed --------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I'm sorry, I cannot help with that request.",
        "The rate appears to be 1150 SEK per hour.",   # prose, no object
        "[1, 2, 3]",                                   # JSON, but not an object
    ],
)
def test_unrecoverable_output_fails_rather_than_defaulting(text):
    result = parse_model(text, ClauseInterpretation, normalise=_normalise_interpretation)
    assert not result.ok
    assert result.value is None
    assert result.error


def test_a_missing_rate_is_never_silently_zero():
    """The specific defaulting bug this whole module exists to prevent."""
    result = parse_model("not json at all", ClauseInterpretation,
                         normalise=_normalise_interpretation)
    assert result.value is None, "must not produce an object with contract_rate=0.0"


def test_raw_output_is_kept_for_the_trace():
    """A parse failure is only debuggable if the text that caused it is retained."""
    result = parse_model("I'm sorry, I cannot help.", Tiny)
    assert not result.ok
    assert result.raw == "I'm sorry, I cannot help."
