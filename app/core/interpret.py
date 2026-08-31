"""Step 4a: LLM interpretation of the retrieved clauses.

This is the only stage where a language model is in the loop, and it is scoped
to the one job that is genuinely fuzzy: mapping a terse invoice line ("snr
tech", "go-live") onto a clause written in legal prose, and reading the rate
and terms back out.

**Three queries per line, not one.** The checker needs three facts -- is this
covered, at what rate, up to what ceiling -- and contracts put each in a
different clause family (§3 scope, §4 fees, §5 volume). An invoice line
describes *work*, so its vocabulary matches the scope clause and misses the
other two. "CPD delivery - digital tools in teaching" ranks §3.1 first and the
clause that actually states the price sixth; "§5.1 Annual ceiling" shares no
vocabulary with the work at all.

So the query is split by the fact being sought rather than by the text
available: one query from the line description, one targeted at rates, one at
ceilings, merged into a single candidate set. A single query fails *quietly* --
the system would simply never flag an over-cap invoice, and would report a
missing rate as "no clause governs this line" when the clause was there and
merely ranked seventh. Retrieval misses are invisible in the output, which is
why the eval harness scores retrieval separately from decisions.

**Failure is closed, not defaulted.** If the model's output cannot be parsed
into the schema, this returns an error and the line goes to a human. The
tempting alternative -- default `contract_rate` to 0.0 and carry on -- would
produce a confident, arithmetically flawless, completely wrong statement about
public money.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.llm.base import LLMClient, get_llm_client
from app.llm.parsing import ParseResult, coerce_number, coerce_optional_str, parse_model
from app.llm.prompts import INTERPRETATION_SYSTEM, build_interpretation_prompt
from app.models import ClauseInterpretation, Invoice, LineItem, Principal, RetrievedChunk
from app.retrieval.retriever import ClauseRetriever

CAP_QUERY = (
    "maximum volume ceiling authorised quantity limit per calendar month "
    "shall not exceed hours occasions units"
)

RATE_QUERY = (
    "price rate fee per unit SEK amount payable excluding VAT "
    "shall pay per hour per day per occasion"
)


@dataclass
class InterpretationOutcome:
    line_no: int
    interpretation: ClauseInterpretation | None = None
    clauses: list[RetrievedChunk] = field(default_factory=list)
    parse: ParseResult | None = None
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.interpretation is not None and self.error is None

    @property
    def retrieved_clause_ids(self) -> list[str]:
        return [c.clause_id for c in self.clauses if c.clause_id]


class Interpreter:
    def __init__(
        self,
        retriever: ClauseRetriever | None = None,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or ClauseRetriever(settings=self.settings)
        self.llm = llm or get_llm_client(self.settings)

    def gather_clauses(
        self, line: LineItem, contract_id: str, principal: Principal
    ) -> list[RetrievedChunk]:
        primary = self.retriever.retrieve(
            line.description, principal, contract_id=contract_id
        ).chunks
        rates = self.retriever.retrieve(
            RATE_QUERY, principal, contract_id=contract_id, top_k=3
        ).chunks
        caps = self.retriever.retrieve(
            CAP_QUERY, principal, contract_id=contract_id, top_k=2
        ).chunks

        merged: dict[str, RetrievedChunk] = {}
        for chunk in [*primary, *rates, *caps]:
            merged.setdefault(chunk.chunk_id, chunk)
        return list(merged.values())

    def interpret_line(
        self,
        invoice: Invoice,
        line: LineItem,
        principal: Principal,
        clauses: list[RetrievedChunk] | None = None,
    ) -> InterpretationOutcome:
        clauses = clauses if clauses is not None else self.gather_clauses(
            line, invoice.contract_id, principal
        )

        if not clauses:
            return InterpretationOutcome(
                line_no=line.line_no, clauses=[],
                error="no readable clauses were retrieved for this contract",
            )

        prompt = build_interpretation_prompt(
            line=line,
            contract_id=invoice.contract_id,
            supplier=invoice.supplier,
            period=invoice.period,
            clauses=clauses,
        )

        try:
            response = self.llm.complete(
                system=INTERPRETATION_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            return InterpretationOutcome(
                line_no=line.line_no, clauses=clauses,
                error=f"LLM call failed: {exc}",
            )

        parsed = parse_model(
            response.text, ClauseInterpretation, normalise=_normalise_interpretation
        )
        if not parsed.ok or parsed.value is None:
            return InterpretationOutcome(
                line_no=line.line_no, clauses=clauses, parse=parsed,
                usage=response.usage,
                error=f"could not parse model output: {parsed.error}",
            )

        interpretation = parsed.value
        interpretation.line_no = line.line_no

        # Guardrail: the model may only cite clauses it was actually shown.
        # A citation that is not in the candidate set is a hallucinated
        # reference, and a caseworker clicking it would find nothing.
        visible = {c.clause_id for c in clauses if c.clause_id}
        if interpretation.governing_clause_id and interpretation.governing_clause_id not in visible:
            return InterpretationOutcome(
                line_no=line.line_no, clauses=clauses, parse=parsed, usage=response.usage,
                error=(
                    f"model cited clause {interpretation.governing_clause_id!r}, which was "
                    "not among the retrieved clauses"
                ),
            )
        interpretation.alternative_clause_ids = [
            c for c in interpretation.alternative_clause_ids if c in visible
        ]

        return InterpretationOutcome(
            line_no=line.line_no, interpretation=interpretation,
            clauses=clauses, parse=parsed, usage=response.usage,
        )


def _normalise_interpretation(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce the fields models most often format differently than requested."""
    out = dict(payload)

    for key in ("contract_rate", "quantity_cap", "confidence"):
        if key in out:
            out[key] = coerce_number(out[key])

    for key in ("governing_clause_id", "cap_clause_id", "exclusion_clause_id"):
        if key in out:
            out[key] = coerce_optional_str(out[key])

    for key in ("rate_unit", "cap_unit", "reasoning"):
        if key in out and out[key] is None:
            out[key] = ""

    if out.get("confidence") is None:
        out["confidence"] = 0.0

    alternatives = out.get("alternative_clause_ids")
    if isinstance(alternatives, str):
        out["alternative_clause_ids"] = [alternatives]
    elif alternatives is None:
        out["alternative_clause_ids"] = []

    covered = out.get("covered_by_contract")
    if isinstance(covered, str):
        out["covered_by_contract"] = covered.strip().lower() in {"true", "yes", "1"}

    return out
