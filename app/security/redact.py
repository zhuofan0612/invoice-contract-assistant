"""Lightweight PII redaction for traces and logs.

Scope, stated honestly: this covers the identifier shapes that actually occur
in Swedish municipal procurement documents -- personnummer, org.nr, IBAN/bankgiro,
email, phone. It is regex-based and is not a substitute for Microsoft Presidio,
which the design names for the full guardrail phase. It is here because traces
persist to disk and a trace file is an easier target than the database, so
shipping raw invoice text into it by default is the wrong posture.

Personnummer is the one worth getting right: YYMMDD-NNNN and YYYYMMDD-NNNN both
occur, and the 10-digit form collides with other numbers, so the separator is
required rather than optional.
"""

from __future__ import annotations

import re

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Swedish personal identity number: 19850101-1234 / 850101-1234 / 850101+1234
    #
    # A personnummer and an organisation number share a shape (NNNNNN-NNNN), so
    # the date is what separates them: the middle pair must be a real month.
    # 556677-8899 is a company, not a person -- without this constraint it was
    # matching here first and being labelled PERSONNUMMER. Both get redacted
    # either way, so this is not a leak, but a company registration number is
    # public record rather than personal data, and mislabelling it as personal
    # data is the kind of error that makes a retention argument unreviewable.
    #
    # The day range allows 61-91 as well as 01-31: a samordningsnummer, issued
    # to people without a personnummer, is a real date with 60 added to the day.
    ("PERSONNUMMER", re.compile(
        r"\b(?:19|20)?\d{2}(?:0[1-9]|1[0-2])"
        r"(?:0[1-9]|[12]\d|3[01]|6[1-9]|[78]\d|9[01])[-+]\d{4}\b"
    )),
    # Swedish organisation number: 556712-4408
    ("ORGNR", re.compile(r"\b\d{6}-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    ("BANKGIRO", re.compile(r"\b\d{3,4}-\d{4}\b")),
    ("PHONE", re.compile(r"\b(?:\+46|0)\s?(?:\d[\s-]?){7,11}\d\b")),
]


def redact(text: str) -> str:
    """Replace identifiers with typed placeholders, preserving readability."""
    if not text:
        return text
    out = text
    for label, pattern in PATTERNS:
        out = pattern.sub(f"[{label}]", out)
    return out


def find_pii(text: str) -> list[str]:
    """Report which identifier types are present, without echoing the values."""
    return [label for label, pattern in PATTERNS if pattern.search(text or "")]
