"""Defensive parsing of structured LLM output.

A model asked for JSON returns JSON *almost* always. The residual few per cent
is where a naive `json.loads(response)` turns into a 500 in production, and the
failures are boringly repetitive:

    ```json fences around the object
    "Here is the analysis:" before the object
    a trailing comma after the last field
    1 150   -- a Swedish-formatted number the contract itself writes that way
    "null" as a string instead of null
    truncation when max_tokens is hit

So parsing is a ladder, cheapest strategy first, and every rung records what it
had to do. `ParseResult.repairs` is written to the trace, because "the model
returned malformed JSON 4% of the time and we repaired it" is an operational
fact worth being able to measure rather than guess at.

The last rung matters most: if nothing yields a valid object, this returns a
failure rather than a half-populated one. A silently defaulted `contract_rate`
of 0.0 would flow into the arithmetic check and produce a confident,
wrong answer about someone's money. Failing closed sends it to a human instead.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
# "1 150", "1 150,50", "1150.50", "1,150.50", "SEK 1 150 per hour"
#
# Grab the whole run of digits and separators and disambiguate afterwards. An
# earlier version stopped the integer part at the first comma, so "1,150.50"
# matched only "1,150" and came back as 1.15 -- a factor-of-1000 error on a
# figure that is someone's money.
NUMBER_RE = re.compile(r"-?\d[\d\s\u00a0.,]*\d|-?\d")


@dataclass
class ParseResult(Generic[T]):
    value: T | None = None
    ok: bool = False
    strategy: str = ""
    repairs: list[str] = field(default_factory=list)
    error: str | None = None
    raw: str = ""

    @property
    def was_repaired(self) -> bool:
        return bool(self.repairs)


# ---------------------------------------------------------------------------
# Stage 1: get *some* JSON value out of the text
# ---------------------------------------------------------------------------


def extract_json(text: str) -> tuple[Any | None, str, list[str]]:
    """Return (value, strategy, repairs). Value is None if nothing parsed."""
    if not text or not text.strip():
        return None, "empty", []

    candidate = text.strip()

    # Rung 1: it is already valid JSON.
    try:
        return json.loads(candidate), "direct", []
    except json.JSONDecodeError:
        pass

    repairs: list[str] = []

    # Rung 2: pull it out of a markdown fence.
    fence = FENCE_RE.search(candidate)
    if fence:
        inner = fence.group(1).strip()
        repairs.append("stripped markdown code fence")
        try:
            return json.loads(inner), "fenced", repairs
        except json.JSONDecodeError:
            candidate = inner

    # Rung 3: scan for the first balanced object/array, ignoring prose either side.
    block = _first_balanced_block(candidate)
    if block is not None:
        if block != candidate:
            repairs.append("extracted balanced JSON block from surrounding prose")
        try:
            return json.loads(block), "balanced-scan", repairs
        except json.JSONDecodeError:
            candidate = block

    # Rung 4: mechanical repairs of the usual suspects.
    repaired, applied = _repair(candidate)
    if applied:
        repairs.extend(applied)
        try:
            return json.loads(repaired), "repaired", repairs
        except json.JSONDecodeError:
            pass

    # Rung 5: the response was truncated mid-object; close it and retry.
    closed, applied = _close_truncated(repaired)
    if applied:
        repairs.extend(applied)
        try:
            return json.loads(closed), "closed-truncated", repairs
        except json.JSONDecodeError:
            pass

    return None, "failed", repairs


def _first_balanced_block(text: str) -> str | None:
    """Find the first complete {...} or [...], respecting strings and escapes."""
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _repair(text: str) -> tuple[str, list[str]]:
    applied: list[str] = []
    out = text

    if "“" in out or "”" in out or "’" in out:
        out = out.replace("“", '"').replace("”", '"').replace("’", "'")
        applied.append("normalised smart quotes")

    if TRAILING_COMMA_RE.search(out):
        out = TRAILING_COMMA_RE.sub(r"\1", out)
        applied.append("removed trailing comma")

    for literal, replacement in (("None", "null"), ("True", "true"), ("False", "false")):
        pattern = rf"(?<![\"\w]){literal}(?![\"\w])"
        if re.search(pattern, out):
            out = re.sub(pattern, replacement, out)
            applied.append(f"replaced Python literal {literal}")

    if re.search(r"\bNaN\b|\bInfinity\b", out):
        out = re.sub(r"\b(?:NaN|-?Infinity)\b", "null", out)
        applied.append("replaced non-JSON float literal with null")

    return out, applied


def _close_truncated(text: str) -> tuple[str, list[str]]:
    """Balance unclosed brackets from a response cut off at max_tokens."""
    depth_curly = depth_square = 0
    in_string = escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        depth_curly += (ch == "{") - (ch == "}")
        depth_square += (ch == "[") - (ch == "]")

    if not in_string and depth_curly <= 0 and depth_square <= 0:
        return text, []

    out = text.rstrip().rstrip(",")
    applied = []
    if in_string:
        out += '"'
        applied.append("closed unterminated string")
    out += "}" * max(0, depth_curly) + "]" * max(0, depth_square)
    applied.append("closed truncated JSON structure")
    return out, applied


# ---------------------------------------------------------------------------
# Stage 2: coerce into the expected schema
# ---------------------------------------------------------------------------


def coerce_number(value: Any) -> float | None:
    """Best-effort number from a model that wrote '1 150 SEK' or '1 150,50'.

    Contracts in this dataset print rates as "SEK 1 150 per hour", so a model
    echoing the contract's own formatting is the expected case, not an edge one.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text or text.lower() in {"null", "none", "n/a", "unknown", "-"}:
        return None

    match = NUMBER_RE.search(text)
    if not match:
        return None

    number = re.sub(r"[\s\u00a0]", "", match.group(0))
    return _to_float(number)


def _to_float(number: str) -> float | None:
    """Resolve which of `.` and `,` is the decimal point.

    Swedish contracts write "1 150,50"; a model may echo that, normalise it to
    "1,150.50", or emit a plain float. All three mean the same amount, and
    guessing wrong is a 100x or 1000x error rather than a rounding one, so the
    rule is explicit:

      * both separators present -> whichever comes *last* is the decimal point
      * one separator, followed by exactly three digits and nothing else ->
        a thousands group ("1,150" and "1.150" are both 1150)
      * otherwise -> it is the decimal point ("1150,50" is 1150.50)

    The three-digit rule is genuinely ambiguous in one case: "1,150" could be
    one thousand one hundred and fifty, or one point one five. Contracts here
    quote whole-krona rates, and thousands grouping is overwhelmingly the
    intent, so it resolves that way -- and the rate is cross-checked against
    the clause text downstream regardless.
    """
    has_comma, has_dot = "," in number, "." in number

    if has_comma and has_dot:
        decimal_sep = "," if number.rfind(",") > number.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        number = number.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif has_comma or has_dot:
        sep = "," if has_comma else "."
        head, _, tail = number.rpartition(sep)
        if len(tail) == 3 and tail.isdigit() and head:
            number = number.replace(sep, "")      # thousands group
        else:
            number = number.replace(sep, ".")     # decimal point

    try:
        return float(number)
    except ValueError:
        return None


def coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"null", "none", "n/a", "unknown"}:
            return None
        return text
    return str(value)


def parse_model(
    text: str,
    model_cls: type[T],
    normalise: Callable[[dict], dict] | None = None,
) -> ParseResult[T]:
    """Parse text into `model_cls`.

    `normalise` runs on the recovered dict before schema validation, which is
    where field-level coercion belongs -- a model that answers "1 150 SEK" is
    not wrong about the rate, it is wrong about the format, and rejecting it
    outright would throw away a correct reading.
    """
    value, strategy, repairs = extract_json(text)
    if value is None:
        return ParseResult(
            ok=False, strategy=strategy, repairs=repairs, raw=text,
            error="no JSON object could be recovered from the response",
        )
    if not isinstance(value, dict):
        return ParseResult(
            ok=False, strategy=strategy, repairs=repairs, raw=text,
            error=f"expected a JSON object, got {type(value).__name__}",
        )
    if normalise is not None:
        before = dict(value)
        value = normalise(value)
        if value != before:
            repairs.append("coerced field types to the expected schema")

    try:
        return ParseResult(
            value=model_cls.model_validate(value), ok=True,
            strategy=strategy, repairs=repairs, raw=text,
        )
    except ValidationError as exc:
        return ParseResult(
            ok=False, strategy=strategy, repairs=repairs, raw=text,
            error=f"schema validation failed: {exc.errors()[:3]}",
        )
