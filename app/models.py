"""Shared schemas.

The important boundary in this file is between `ClauseInterpretation` (what the
LLM is allowed to tell us) and `LineDecision` (what the deterministic checker
concludes). The LLM may report what a clause *says*; it never reports whether
the invoice passes.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Decision(StrEnum):
    MATCH = "match"
    FLAG = "flag"


class Finding(StrEnum):
    """Why a line was flagged. Mirrors the dataset's ground-truth categories."""

    CLEAN_MATCH = "clean_match"
    OVERBILLING_WRONG_TOTAL = "overbilling_wrong_total"
    WRONG_RATE = "wrong_rate"
    UNAUTHORIZED_QUANTITY = "unauthorized_quantity"
    ITEM_NOT_IN_CONTRACT = "item_not_in_contract"
    AMBIGUOUS_MAPPING = "ambiguous_mapping"
    NO_MATCHING_CLAUSE = "no_matching_clause"
    INVALID_INVOICE = "invalid_invoice"
    INTERPRETATION_FAILED = "interpretation_failed"


# --------------------------------------------------------------------------
# Invoice intake
# --------------------------------------------------------------------------


class LineItem(BaseModel):
    line_no: int
    description: str
    quantity: float
    unit: str = ""
    unit_price: float
    line_total: float

    @field_validator("quantity", "unit_price", "line_total")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must not be negative")
        return v


class Invoice(BaseModel):
    invoice_id: str
    supplier: str
    contract_id: str
    issue_date: str = ""
    period: str = ""
    currency: str = "SEK"
    line_items: list[LineItem] = Field(min_length=1)
    invoice_total: float


class Principal(BaseModel):
    """The caseworker making the request; source of the access-control filter."""

    user_id: str = "anonymous"
    groups: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    chunk_id: str
    contract_id: str
    supplier: str
    clause_id: str | None = None
    heading: str = ""
    text: str
    has_table: bool = False
    candidate_score: float = 0.0
    rerank_score: float = 0.0


# --------------------------------------------------------------------------
# LLM interpretation — the ONLY thing the model is trusted to produce
# --------------------------------------------------------------------------


class ClauseInterpretation(BaseModel):
    """What the model read out of the retrieved clauses for one invoice line.

    Note what is absent: no pass/fail, no computed total, no decision. The model
    reports the contract's terms; `app.core.check` decides.
    """

    line_no: int
    governing_clause_id: str | None = None
    covered_by_contract: bool = True
    contract_rate: float | None = None
    rate_unit: str = ""
    quantity_cap: float | None = None
    cap_unit: str = ""
    cap_clause_id: str | None = None
    exclusion_clause_id: str | None = None
    alternative_clause_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# --------------------------------------------------------------------------
# Deterministic decision
# --------------------------------------------------------------------------


class LineDecision(BaseModel):
    line_no: int
    description: str
    status: Decision
    finding: Finding
    cited_clause_id: str | None = None
    candidate_clause_ids: list[str] = Field(default_factory=list)
    explanation: str = ""
    stated_line_total: float | None = None
    expected_line_total: float | None = None
    stated_unit_price: float | None = None
    contract_rate: float | None = None
    delta: float | None = None
    confidence: float = 0.0


class InvoiceDecision(BaseModel):
    invoice_id: str
    supplier: str
    contract_id: str
    decision: Decision
    requires_human_review: bool = True
    findings: list[Finding] = Field(default_factory=list)
    line_decisions: list[LineDecision] = Field(default_factory=list)
    retrieved_clause_ids: list[str] = Field(default_factory=list)
    total_delta: float = 0.0
    trace_id: str = ""
    timings_ms: dict[str, float] = Field(default_factory=dict)
    backend: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class CheckRequest(BaseModel):
    invoice: Invoice
    principal: Principal = Field(default_factory=Principal)
    mode: Literal["pipeline", "agent"] = "pipeline"
