"""Step 4b: the deterministic checker. This is where the decision is made.

Everything above this file gathers evidence. This file, and only this file,
concludes anything -- and it contains no LLM call, no embedding, no
probability. Given a rate, a cap and an invoice line, the verdict is a fixed
function of the inputs, reproducible and explainable to a caseworker in one
sentence.

That split is the whole safety argument. A hallucinated multiplication would
approve an overpayment of public money and would look exactly as confident as a
correct one. Arithmetic is free and exact in Python, so there is no reason to
delegate it to a system that is merely usually right.

Note what the checker does with uncertainty: `AMBIGUOUS_MAPPING` and
`NO_MATCHING_CLAUSE` are outcomes, not errors. The system is allowed to say it
does not know, and saying so routes the line to a human. The design's error
tolerance is low in both directions -- a wrong "match" approves an overpayment,
a wrong "mismatch" wastes the team's time -- so "I am not sure" is the honest
third answer.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.core.interpret import InterpretationOutcome
from app.models import Decision, Finding, LineDecision, LineItem

# Most-severe-first. A line can trip several rules; this decides which one the
# caseworker sees as the headline, and the explanation lists the rest.
PRECEDENCE = [
    Finding.INTERPRETATION_FAILED,
    Finding.ITEM_NOT_IN_CONTRACT,
    Finding.NO_MATCHING_CLAUSE,
    Finding.AMBIGUOUS_MAPPING,
    Finding.UNAUTHORIZED_QUANTITY,
    Finding.WRONG_RATE,
    Finding.OVERBILLING_WRONG_TOTAL,
]


def check_line(
    line: LineItem, outcome: InterpretationOutcome, settings: Settings | None = None
) -> LineDecision:
    settings = settings or get_settings()

    if not outcome.ok or outcome.interpretation is None:
        return LineDecision(
            line_no=line.line_no, description=line.description,
            status=Decision.FLAG, finding=Finding.INTERPRETATION_FAILED,
            candidate_clause_ids=outcome.retrieved_clause_ids,
            explanation=(
                f"The contract terms for this line could not be established "
                f"({outcome.error}). Sent for manual review rather than guessed at."
            ),
            stated_line_total=line.line_total, delta=line.line_total,
        )

    interp = outcome.interpretation
    findings: list[tuple[Finding, str]] = []
    delta = 0.0
    expected_total: float | None = None

    # --- 1. the contract positively excludes this item ---------------------
    if interp.exclusion_clause_id and not interp.covered_by_contract:
        return LineDecision(
            line_no=line.line_no, description=line.description,
            status=Decision.FLAG, finding=Finding.ITEM_NOT_IN_CONTRACT,
            cited_clause_id=interp.exclusion_clause_id,
            candidate_clause_ids=outcome.retrieved_clause_ids,
            explanation=(
                f"{interp.exclusion_clause_id} places this item outside the contract, "
                f"so the full {line.line_total:,.2f} is not payable under it. {interp.reasoning}"
            ),
            stated_line_total=line.line_total, stated_unit_price=line.unit_price,
            delta=line.line_total, confidence=interp.confidence,
        )

    # --- 2. nothing in the contract governs this line ----------------------
    if interp.governing_clause_id is None or interp.contract_rate is None:
        return LineDecision(
            line_no=line.line_no, description=line.description,
            status=Decision.FLAG, finding=Finding.NO_MATCHING_CLAUSE,
            candidate_clause_ids=outcome.retrieved_clause_ids,
            explanation=(
                "No clause in this contract states a rate governing this line, so the "
                f"system cannot verify {line.line_total:,.2f}. {interp.reasoning}"
            ),
            stated_line_total=line.line_total, stated_unit_price=line.unit_price,
            delta=line.line_total, confidence=interp.confidence,
        )

    # --- 3. more than one clause plausibly applies -------------------------
    alternatives = [c for c in interp.alternative_clause_ids if c != interp.governing_clause_id]
    if alternatives and interp.confidence < settings.ambiguity_confidence_threshold:
        return LineDecision(
            line_no=line.line_no, description=line.description,
            status=Decision.FLAG, finding=Finding.AMBIGUOUS_MAPPING,
            cited_clause_id=interp.governing_clause_id,
            candidate_clause_ids=[interp.governing_clause_id, *alternatives],
            explanation=(
                f"This line could be governed by {interp.governing_clause_id} or by "
                f"{', '.join(alternatives)}, which carry different rates. Confidence "
                f"{interp.confidence:.2f} is below the {settings.ambiguity_confidence_threshold:.2f} "
                f"threshold, so a caseworker should decide. {interp.reasoning}"
            ),
            stated_line_total=line.line_total, stated_unit_price=line.unit_price,
            contract_rate=interp.contract_rate, delta=line.line_total,
            confidence=interp.confidence,
        )

    # --- 4. the arithmetic on the invoice itself ---------------------------
    internal_total = round(line.quantity * line.unit_price, 2)
    if abs(line.line_total - internal_total) > settings.total_tolerance:
        overstatement = round(line.line_total - internal_total, 2)
        findings.append((
            Finding.OVERBILLING_WRONG_TOTAL,
            f"Line total {line.line_total:,.2f} does not equal quantity {line.quantity:g} "
            f"x unit price {line.unit_price:,.2f} = {internal_total:,.2f} "
            f"(overstated by {overstatement:,.2f}).",
        ))
        delta += overstatement

    # --- 5. billed rate against the contract rate --------------------------
    expected_total = round(line.quantity * interp.contract_rate, 2)
    if abs(line.unit_price - interp.contract_rate) > settings.rate_tolerance:
        rate_delta = round((line.unit_price - interp.contract_rate) * line.quantity, 2)
        findings.append((
            Finding.WRONG_RATE,
            f"Billed {line.unit_price:,.2f} per {line.unit or 'unit'}; "
            f"{interp.governing_clause_id} states {interp.contract_rate:,.2f} "
            f"{interp.rate_unit or ''}".rstrip()
            + f". Difference over {line.quantity:g} = {rate_delta:,.2f}.",
        ))
        delta += rate_delta

    # --- 6. quantity against the contractual ceiling -----------------------
    if interp.quantity_cap is not None and line.quantity > interp.quantity_cap:
        excess = line.quantity - interp.quantity_cap
        cap_delta = round(excess * interp.contract_rate, 2)
        findings.append((
            Finding.UNAUTHORIZED_QUANTITY,
            f"Quantity {line.quantity:g} exceeds the ceiling of {interp.quantity_cap:g} "
            f"{interp.cap_unit or ''}".rstrip()
            + f" set by {interp.cap_clause_id or 'the contract'} "
            f"({excess:g} over, worth {cap_delta:,.2f}).",
        ))
        delta += cap_delta

    if not findings:
        return LineDecision(
            line_no=line.line_no, description=line.description,
            status=Decision.MATCH, finding=Finding.CLEAN_MATCH,
            cited_clause_id=interp.governing_clause_id,
            candidate_clause_ids=outcome.retrieved_clause_ids,
            explanation=(
                f"Agrees with {interp.governing_clause_id}: {line.quantity:g} x "
                f"{interp.contract_rate:,.2f} = {expected_total:,.2f}, within the "
                "contractual ceiling."
            ),
            stated_line_total=line.line_total, expected_line_total=expected_total,
            stated_unit_price=line.unit_price, contract_rate=interp.contract_rate,
            delta=0.0, confidence=interp.confidence,
        )

    primary = min(findings, key=lambda f: PRECEDENCE.index(f[0]))
    return LineDecision(
        line_no=line.line_no, description=line.description,
        status=Decision.FLAG, finding=primary[0],
        cited_clause_id=interp.governing_clause_id,
        candidate_clause_ids=outcome.retrieved_clause_ids,
        explanation=" ".join(message for _, message in findings),
        stated_line_total=line.line_total, expected_line_total=expected_total,
        stated_unit_price=line.unit_price, contract_rate=interp.contract_rate,
        delta=round(delta, 2), confidence=interp.confidence,
    )
