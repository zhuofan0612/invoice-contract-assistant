"""Prompts for the interpretation step.

Two properties are deliberate.

**The model is fenced in.** The system prompt tells it what it may report and,
explicitly, what it must not do: no arithmetic, no verdict. Those belong to
`app/core/check.py`. Asking a language model whether 120 x 1150 equals 138000
invites a fluent wrong answer about public money; asking it what rate §4.2
states plays to what it is actually good at.

**"I don't know" is a first-class answer.** Nulls and a low confidence are
valid, encouraged outputs. A system that must always produce a clause will
always produce a clause, including when none applies -- which is exactly the
`no_matching_clause` category in the golden set.

The user prompt puts the invoice line and the candidate clauses in tagged JSON
blocks. That is partly for the model's benefit and partly so the offline stub
client can read the same structured payload out of the same prompt, which keeps
both backends on one code path.
"""

from __future__ import annotations

import json
from typing import Sequence

from app.models import LineItem, RetrievedChunk

INTERPRETATION_SYSTEM = """\
You are assisting a procurement caseworker at a Swedish municipality. You read \
contract clauses and report what they say about a single invoice line.

YOUR JOB
Report, strictly from the candidate clauses supplied:
  - which clause governs this invoice line, if any
  - the rate that clause states, and the unit that rate is expressed in
  - any volume cap or ceiling that applies, and the clause stating it
  - whether the clause set positively excludes this item from the contract

WHAT YOU MUST NOT DO
  - Do not perform arithmetic. Do not multiply quantity by rate, do not check
    the invoice total, do not compute differences. Downstream code does that.
  - Do not decide whether the invoice is correct, approved, or fraudulent.
  - Do not use knowledge outside the supplied clauses. If the rate is not in
    the text you were given, it is unknown.

AMBIGUITY IS AN ACCEPTABLE ANSWER
  - If no supplied clause governs the line, set governing_clause_id to null and
    covered_by_contract to false, and explain why in reasoning.
  - If two clauses plausibly govern it at different rates, put the more likely
    one in governing_clause_id, list every plausible clause in
    alternative_clause_ids, and set confidence at or below 0.6. Do not break
    the tie by guessing; a human will resolve it.
  - Set confidence honestly. It expresses how sure you are that
    governing_clause_id is the right clause for this line.

A rate only governs a line if its unit is compatible with what the line bills.
An hourly rate does not govern a line billed per metre.

OUTPUT
Return one JSON object and nothing else. No prose, no markdown fence.

{
  "line_no": <int>,
  "governing_clause_id": <string|null>,
  "covered_by_contract": <bool>,
  "contract_rate": <number|null>,
  "rate_unit": <string>,
  "quantity_cap": <number|null>,
  "cap_unit": <string>,
  "cap_clause_id": <string|null>,
  "exclusion_clause_id": <string|null>,
  "alternative_clause_ids": [<string>],
  "confidence": <number 0.0-1.0>,
  "reasoning": <string, max 2 sentences>
}
"""


# The agent variant differs from the above in exactly one respect: the clauses
# are not handed over, the model must go and find them. The output contract is
# byte-identical, deliberately -- both paths feed the same parser, the same
# hallucinated-citation guardrail and the same deterministic checker, so the
# eval can attribute any score difference to the search strategy alone.
AGENT_SYSTEM = (
    INTERPRETATION_SYSTEM.replace(
        "Report, strictly from the candidate clauses supplied:",
        "Find the relevant clauses using your tools, then report, strictly "
        "from the clauses those tools returned:",
    )
    + """
FINDING THE CLAUSES
You are given an invoice line but no clauses. Use retrieve_clauses to search
the contract named in the prompt.

  - The facts you need usually live in different clauses. What the work is,
    what it costs, and how much of it may be bought are typically three
    separate clauses, and one search rarely surfaces all three. Search
    separately for the rate and for any ceiling.
  - Search in the contract's vocabulary, not the invoice's. The line may say
    "snr tech"; the contract says "senior technician".
  - Use read_clause when an excerpt is cut off or points at a clause you have
    not seen.
  - If a search returns nothing useful, change the wording. If two or three
    attempts return nothing, the contract most likely does not address this:
    say so with governing_clause_id null. That is a correct answer, not a
    failure.
  - Never cite a clause ID you have not seen in a tool result. Do not guess an
    ID by pattern, even an obvious-looking one.

You have a limited number of turns. Stop searching and answer as soon as you
can name the governing clause and its rate, or conclude there is none.
"""
)


def build_agent_prompt(invoice, line) -> str:
    """The agent's opening turn: the line and the contract, and no clauses."""
    payload = {
        "line_no": line.line_no,
        "description": line.description,
        "quantity": line.quantity,
        "unit": line.unit,
        "unit_price_billed": line.unit_price,
        "line_total_billed": line.line_total,
        "billing_period": invoice.period,
    }
    return f"""\
<invoice_line>
{json.dumps(payload, ensure_ascii=False, indent=2)}
</invoice_line>

<contract>
{json.dumps({"contract_id": invoice.contract_id, "supplier": invoice.supplier},
            ensure_ascii=False, indent=2)}
</contract>

Search contract {invoice.contract_id} for the clauses governing this line, \
then return only the JSON object described in your instructions."""


def build_interpretation_prompt(
    line: LineItem,
    contract_id: str,
    supplier: str,
    period: str,
    clauses: Sequence[RetrievedChunk],
) -> str:
    line_payload = {
        "line_no": line.line_no,
        "description": line.description,
        "quantity": line.quantity,
        "unit": line.unit,
        "unit_price_billed": line.unit_price,
        "line_total_billed": line.line_total,
        "billing_period": period,
    }
    contract_payload = {"contract_id": contract_id, "supplier": supplier}
    clause_payload = [
        {
            "clause_id": c.clause_id or c.chunk_id,
            "heading": c.heading,
            "text": c.text,
            "contains_table": c.has_table,
        }
        for c in clauses
    ]

    return f"""\
<invoice_line>
{json.dumps(line_payload, ensure_ascii=False, indent=2)}
</invoice_line>

<contract>
{json.dumps(contract_payload, ensure_ascii=False, indent=2)}
</contract>

<candidate_clauses>
{json.dumps(clause_payload, ensure_ascii=False, indent=2)}
</candidate_clauses>

Report what these clauses say about this invoice line. Return only the JSON \
object described in your instructions."""
