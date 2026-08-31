"""Offline LLM stand-in: a lexical baseline that speaks the same protocol.

Two jobs.

**Job one: the project runs with no API key.** Clone, install, `make demo`, and
the full pipeline executes. Nothing about ingestion, retrieval, parsing,
checking, evaluation or the API needs a credential to be exercised.

**Job two: it is the deterministic baseline.** The design calls for
cross-checking LLM judgements against a non-LLM baseline instead of trusting an
LLM-as-judge on its own. This is that baseline. Running the eval twice, once
with LLM_BACKEND=stub and once with anthropic, prices what the language model
is actually buying.

It matches on words, prefixes and units. It has no notion of meaning, so it
resolves rates, tables, tiers and caps reliably, and it is expected to lose on
`ambiguous_mapping`, where deciding between "strategic advisory" and
"implementation" needs a two-hop inference through §3.1 rather than shared
vocabulary. That gap is the measurement, not a defect to paper over.

It emits JSON as text and goes through the same defensive parser as the real
client, so the offline path tests the real code path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from app.llm.base import LLMResponse, ToolCall
from app.llm.parsing import coerce_number

TAG_RE = {
    "line": re.compile(r"<invoice_line>\s*(\{.*?\})\s*</invoice_line>", re.DOTALL),
    "contract": re.compile(r"<contract>\s*(\{.*?\})\s*</contract>", re.DOTALL),
    "clauses": re.compile(r"<candidate_clauses>\s*(\[.*?\])\s*</candidate_clauses>", re.DOTALL),
}

# "a fixed hourly rate of SEK 1 450 per hour excluding VAT"
RATE_RE = re.compile(
    r"SEK\s+(?P<amount>\d[\d\s\u00a0]*(?:[.,]\d+)?)\s+per\s+(?P<unit>[a-zA-Z][a-zA-Z ]*?)"
    r"(?=\s+excluding|\s+and\s+month|[,.;]|\s+shall|\s+may|$)",
    re.IGNORECASE,
)
CAP_TRIGGER_RE = re.compile(r"(?:shall not exceed|maximum of)\s+(?P<amount>\d[\d\s\u00a0]*)", re.I)
PER_RE = re.compile(r"per\s+(?P<per>calendar month|contract year|month|week|year|day)", re.I)
# Deliberately does NOT match the bare stem "exclud": every priced clause in
# this corpus ends "...excluding VAT", so a looser pattern marks every rate
# clause as an exclusion and the system reports that nothing is under contract.
EXCLUSION_RE = re.compile(
    r"\bexcluded\b|not covered|does not cover|do not cover|falls? outside|"
    r"outside the scope|shall not invoice\b|are not covered|is not covered",
    re.I,
)
RANGE_RE = re.compile(r"(\d[\d\s\u00a0]*)\s*(?:[-\u2013\u2014]|to)\s*(\d[\d\s\u00a0]*)")
OPEN_RANGE_RE = re.compile(r"(\d[\d\s\u00a0]*)\s*(?:and above|or more|\+)", re.I)

STOPWORDS = {
    "the", "and", "for", "shall", "with", "that", "this", "under", "per", "sek",
    "excluding", "vat", "rate", "supplier", "municipality", "agreement", "contract",
    "invoice", "invoiced", "paid", "price", "cost", "work", "services", "service",
    "described", "applies", "apply", "not", "any", "all", "each", "from", "which",
    "has", "have", "been", "are", "was", "shall", "may", "must", "including", "incl",
}

UNIT_ALIASES = {
    "hours": "hour", "hour": "hour", "hrs": "hour", "h": "hour",
    "occasions": "occasion", "occasion": "occasion", "visits": "occasion", "visit": "occasion",
    "days": "day", "day": "day",
    "devices": "device", "device": "device", "endpoints": "device", "endpoint": "device",
    "portions": "portion", "portion": "portion",
    "metres": "metre", "metre": "metre", "meters": "metre", "m": "metre",
    "units": "unit", "unit": "unit", "pcs": "unit",
    "participants": "participant", "participant": "participant",
    "project": "project", "projects": "project",
}


@dataclass
class _RateCandidate:
    clause_id: str
    rate: float
    unit: str
    overlap: int
    source: str  # prose | table


class StubLLMClient:
    name = "stub"

    def complete(
        self,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        prompt = _last_text(messages)

        if tools:
            return self._agent_turn(prompt, messages, tools)

        payload = _extract_payload(prompt)
        if payload is None:
            return LLMResponse(text="{}", model="stub", stop_reason="end_turn")

        line, _contract, clauses = payload
        return LLMResponse(
            text=json.dumps(_interpret(line, clauses), ensure_ascii=False),
            model="stub",
            stop_reason="end_turn",
            usage={"input_tokens": len(prompt) // 4, "output_tokens": 120},
        )

    def _agent_turn(self, prompt, messages, tools) -> LLMResponse:
        """Minimal policy so the bounded tool loop is exercisable offline.

        Turn 1: fetch clauses. Turn 2+: answer from what came back.
        """
        used_tools = any(
            block.get("type") == "tool_result"
            for m in messages
            if isinstance(m.get("content"), list)
            for block in m["content"]
            if isinstance(block, dict)
        )
        tool_names = {t["name"] for t in tools}

        if not used_tools and "retrieve_clauses" in tool_names:
            payload = _extract_payload(prompt)
            line = payload[0] if payload else {}
            contract = payload[1] if payload else {}
            return LLMResponse(
                stop_reason="tool_use",
                model="stub",
                tool_calls=[
                    ToolCall(
                        id="stub-1",
                        name="retrieve_clauses",
                        arguments={
                            "query": str(line.get("description", "")),
                            "contract_id": str(contract.get("contract_id", "")),
                        },
                    )
                ],
            )

        payload = _extract_payload(prompt)
        if payload is None:
            return LLMResponse(text="{}", model="stub")
        line, _c, clauses = payload
        clauses = clauses or _clauses_from_tool_results(messages)
        return LLMResponse(
            text=json.dumps(_interpret(line, clauses), ensure_ascii=False), model="stub"
        )


# ---------------------------------------------------------------------------
# The heuristic itself
# ---------------------------------------------------------------------------


def _interpret(line: dict, clauses: list[dict]) -> dict:
    line_no = int(line.get("line_no", 1))
    description = str(line.get("description", ""))
    quantity = coerce_number(line.get("quantity")) or 0.0
    line_unit = _normalise_unit(str(line.get("unit", "")))
    desc_tokens = _tokens(description)

    rate_candidates: list[_RateCandidate] = []
    exclusions: list[tuple[str, int]] = []

    for clause in clauses:
        clause_id = clause.get("clause_id") or ""
        text = clause.get("text", "")
        heading = clause.get("heading", "")
        blob = f"{heading}\n{text}"
        overlap = _overlap(desc_tokens, _tokens(blob))

        if EXCLUSION_RE.search(blob):
            exclusions.append((clause_id, overlap))

        for rate, unit, source, row_overlap in _rates_in(blob, desc_tokens, quantity):
            if line_unit and not _units_compatible(line_unit, unit):
                continue
            rate_candidates.append(
                _RateCandidate(clause_id, rate, unit, overlap + row_overlap, source)
            )

    best_rate_overlap = max((c.overlap for c in rate_candidates), default=-1)
    best_exclusion = max(exclusions, key=lambda e: e[1], default=None)

    cap, cap_unit, cap_clause = _find_cap(clauses, line_unit)

    # An exclusion clause wins only by a clear margin, not a bare majority.
    # Exclusion clauses discuss the very things the priced clauses price, so
    # they pick up incidental overlap: C-2024-001 §3.2 excludes hardware but
    # uses the word "procured", which beats the real rate clause 2-1 on a line
    # reading "go-live of procured software". Requiring a margin keeps the
    # rule for cases where the description genuinely is the excluded item.
    if best_exclusion and best_exclusion[1] > 0 and best_exclusion[1] >= best_rate_overlap + 2:
        return _result(
            line_no, None, False, None, "", cap, cap_unit, cap_clause,
            exclusion_clause_id=best_exclusion[0], confidence=0.8,
            reasoning="A clause in this contract expressly places this item outside its scope.",
        )

    if not rate_candidates:
        return _result(
            line_no, None, False, None, "", cap, cap_unit, cap_clause,
            confidence=0.3,
            reasoning=(
                "No supplied clause states a rate in a unit compatible with this "
                f"line (billed per {line_unit or 'unspecified unit'})."
            ),
        )

    rate_candidates.sort(key=lambda c: c.overlap, reverse=True)
    best = rate_candidates[0]

    distinct = {c.rate for c in rate_candidates}
    tied = [c for c in rate_candidates if c.overlap == best.overlap and c.rate != best.rate]

    if len(distinct) > 1 and tied:
        alternatives = sorted({c.clause_id for c in rate_candidates if c.clause_id})
        return _result(
            line_no, best.clause_id, True, best.rate, best.unit,
            cap, cap_unit, cap_clause, alternatives=alternatives, confidence=0.45,
            reasoning=(
                "Several clauses state different rates in a compatible unit and the "
                "description does not clearly favour one of them."
            ),
        )

    confidence = 0.9 if best.overlap >= 2 else 0.7 if best.overlap == 1 else 0.55
    return _result(
        line_no, best.clause_id, True, best.rate, best.unit,
        cap, cap_unit, cap_clause, confidence=confidence,
        reasoning=f"Matched on shared terminology with the {best.source} rate in {best.clause_id}.",
    )


def _rates_in(blob: str, desc_tokens: set[str], quantity: float):
    """Yield (rate, unit, source, extra_overlap) for prose rates and table rows."""
    for match in RATE_RE.finditer(blob):
        amount = coerce_number(match.group("amount"))
        if amount is not None:
            yield amount, _normalise_unit(match.group("unit")), "prose", 0

    yield from _table_rates(blob, desc_tokens, quantity)


def _table_rates(blob: str, desc_tokens: set[str], quantity: float):
    """Read rates out of a Markdown table.

    Two shapes appear in this corpus: rates keyed by a role ("Senior
    technician ... 895") and tiered pricing keyed by a quantity band ("51-200
    ... 175"). The band case is resolved by the line's own quantity, which is
    exactly the kind of rule that belongs in code rather than in a prompt.
    """
    rows = [r for r in blob.splitlines() if r.strip().startswith("|")]
    if len(rows) < 3:
        return

    header = rows[0]
    unit = "hour" if re.search(r"/\s*hour|per hour", header, re.I) else ""
    if re.search(r"device", header, re.I):
        unit = "device"

    for row in rows[2:]:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        rate = coerce_number(cells[-1])
        if rate is None:
            continue

        # Test each cell separately. Joining them first breaks the band parse:
        # a tier row "| 2 | 51-200 | 175 |" joins to "2 51-200", and the range
        # regex then reads the low bound as 251.
        bands = [(_band_for(cell, quantity), cell) for cell in cells[:-1]]
        verdicts = [b for b, _ in bands if b is not None]

        extra = 0
        if verdicts:
            if not any(verdicts):
                continue      # a tier this quantity cannot fall in
            extra += 3        # the quantity lands in this tier: decisive

        labels = " ".join(cell for band, cell in bands if band is None)
        extra += _overlap(desc_tokens, _tokens(labels))

        yield rate, unit, "table", extra


def _band_for(label: str, quantity: float) -> bool | None:
    """True/False if the label is a quantity band, None if it is not a band."""
    open_range = OPEN_RANGE_RE.search(label)
    if open_range:
        return quantity >= (coerce_number(open_range.group(1)) or 0)
    closed = RANGE_RE.search(label)
    if closed:
        low = coerce_number(closed.group(1)) or 0
        high = coerce_number(closed.group(2)) or 0
        return low <= quantity <= high
    return None


def _find_cap(clauses: list[dict], line_unit: str) -> tuple[float | None, str, str | None]:
    for clause in clauses:
        blob = f"{clause.get('heading', '')}\n{clause.get('text', '')}"
        trigger = CAP_TRIGGER_RE.search(blob)
        if not trigger:
            continue
        amount = coerce_number(trigger.group("amount"))
        if amount is None:
            continue

        tail = blob[trigger.end() : trigger.end() + 80]
        cap_unit_match = re.match(r"\s*([a-zA-Z][a-zA-Z ]*?)(?=\s+per\b|[.,]|\s+without|$)", tail)
        cap_unit = _normalise_unit(cap_unit_match.group(1)) if cap_unit_match else ""
        if not cap_unit:
            # "shall not exceed 400 without a written amendment" -- the unit is
            # named earlier in the sentence instead.
            head = blob[max(0, trigger.start() - 120) : trigger.start()]
            for token in reversed(re.findall(r"[a-zA-Z]+", head.lower())):
                if token in UNIT_ALIASES:
                    cap_unit = UNIT_ALIASES[token]
                    break

        period = PER_RE.search(tail)
        unit_label = f"{cap_unit} per {period.group('per')}" if period else cap_unit

        if line_unit and cap_unit and not _units_compatible(line_unit, cap_unit):
            continue
        return amount, unit_label, clause.get("clause_id")
    return None, "", None


def _result(
    line_no, clause_id, covered, rate, rate_unit, cap, cap_unit, cap_clause,
    exclusion_clause_id=None, alternatives=None, confidence=0.5, reasoning="",
) -> dict:
    return {
        "line_no": line_no,
        "governing_clause_id": clause_id,
        "covered_by_contract": covered,
        "contract_rate": rate,
        "rate_unit": rate_unit,
        "quantity_cap": cap,
        "cap_unit": cap_unit,
        "cap_clause_id": cap_clause,
        "exclusion_clause_id": exclusion_clause_id,
        "alternative_clause_ids": alternatives or [],
        "confidence": confidence,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-ZåäöÅÄÖ]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in STOPWORDS}


def _overlap(a: set[str], b: set[str]) -> int:
    """Shared terms, counting prefix matches so 'config' meets 'configuration'."""
    score = 0
    for token in a:
        if token in b:
            score += 1
            continue
        if any(
            (token.startswith(other) or other.startswith(token)) and min(len(token), len(other)) >= 5
            for other in b
        ):
            score += 1
    return score


def _normalise_unit(unit: str) -> str:
    words = re.findall(r"[a-zA-Z]+", unit.lower())
    for word in words:
        if word in UNIT_ALIASES:
            return UNIT_ALIASES[word]
    return words[-1] if words else ""


def _units_compatible(line_unit: str, clause_unit: str) -> bool:
    if not clause_unit:
        return True
    return _normalise_unit(line_unit) == _normalise_unit(clause_unit)


def _last_text(messages: Sequence[dict[str, Any]]) -> str:
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if parts:
                return "\n".join(parts)
    return ""


def _extract_payload(prompt: str):
    line_m = TAG_RE["line"].search(prompt)
    clause_m = TAG_RE["clauses"].search(prompt)
    contract_m = TAG_RE["contract"].search(prompt)
    if not line_m:
        return None
    try:
        line = json.loads(line_m.group(1))
        contract = json.loads(contract_m.group(1)) if contract_m else {}
        clauses = json.loads(clause_m.group(1)) if clause_m else []
    except json.JSONDecodeError:
        return None
    return line, contract, clauses


def _clauses_from_tool_results(messages) -> list[dict]:
    collected: list[dict] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                try:
                    data = json.loads(block.get("content", "[]"))
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(data, list):
                    collected.extend(d for d in data if isinstance(d, dict))
    return collected
