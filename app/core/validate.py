"""Step 1: structural validation of the incoming invoice.

Cheap, contract-free checks that run before anything expensive. The point is to
fail fast on malformed input rather than spend an embedding call and an LLM
call discovering that `contract_id` was blank.

This stage deliberately does not compare anything to the contract. Whether the
rate is right is a question for the checker; whether there *is* a rate field at
all is a question for here.
"""

from __future__ import annotations

from app.models import Invoice

TOLERANCE = 0.01


def validate_invoice(invoice: Invoice) -> list[str]:
    """Return a list of structural problems. Empty means the invoice is usable."""
    issues: list[str] = []

    if not invoice.contract_id.strip():
        issues.append("contract_id is missing, so no governing contract can be identified")
    if not invoice.supplier.strip():
        issues.append("supplier is missing")
    if not invoice.line_items:
        issues.append("invoice has no line items")

    for line in invoice.line_items:
        if not line.description.strip():
            issues.append(f"line {line.line_no}: description is empty, cannot be matched to a clause")
        if line.quantity == 0:
            issues.append(f"line {line.line_no}: quantity is zero")

    stated_sum = round(sum(li.line_total for li in invoice.line_items), 2)
    if abs(stated_sum - invoice.invoice_total) > TOLERANCE:
        issues.append(
            f"invoice_total {invoice.invoice_total:.2f} does not equal the sum of "
            f"line totals {stated_sum:.2f}"
        )

    seen = [li.line_no for li in invoice.line_items]
    if len(set(seen)) != len(seen):
        issues.append("duplicate line numbers")

    return issues
