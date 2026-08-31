"""Step 2: find the governing contract. Deterministically.

This is the clearest example of the project's central rule. Finding the
contract is *not* a semantic search problem: the invoice states its contract ID
and its supplier, so the answer is knowable by exact match. Embedding the
invoice and hoping the right contract ranks first would introduce a failure
mode that simply does not need to exist -- and it would fail silently, quietly
checking an invoice against another supplier's rates.

Semantic search starts one level down, in `retrieval/`, for the question that
genuinely is language-shaped: which clause inside this contract applies?

The supplier cross-check matters too. If the invoice's contract ID and supplier
name disagree, that is either a clerical error or someone billing under
another supplier's agreement. Both need a human.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models import Invoice, Principal
from app.retrieval.retriever import ClauseRetriever
from app.security.access import filter_for


@dataclass
class LookupResult:
    contract_id: str
    found: bool
    readable: bool
    supplier_on_contract: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.found and self.readable and self.error is None


def lookup_contract(
    invoice: Invoice, principal: Principal, retriever: ClauseRetriever
) -> LookupResult:
    contract_id = invoice.contract_id.strip()

    # The filter is applied here too, so "does this contract exist?" is answered
    # in terms of what this caller may see. A caseworker in another department
    # gets "not found", not "found but forbidden" -- the latter leaks the fact
    # that the contract exists.
    search_filter = filter_for(principal, contract_id=contract_id)
    if not retriever.store.contract_exists(contract_id, search_filter):
        return LookupResult(
            contract_id=contract_id, found=False, readable=False,
            error=(
                f"No contract {contract_id} is available to this user. It may not "
                "exist, or it may belong to a department the user cannot access."
            ),
        )

    clauses = retriever.contract_clauses(contract_id, principal)
    suppliers = {c.supplier for c in clauses if c.supplier}
    supplier_on_contract = next(iter(suppliers), "")

    if supplier_on_contract and not _same_supplier(supplier_on_contract, invoice.supplier):
        return LookupResult(
            contract_id=contract_id, found=True, readable=True,
            supplier_on_contract=supplier_on_contract,
            error=(
                f"Invoice is from {invoice.supplier!r} but contract {contract_id} is "
                f"held by {supplier_on_contract!r}."
            ),
        )

    return LookupResult(
        contract_id=contract_id, found=True, readable=True,
        supplier_on_contract=supplier_on_contract,
    )


def _same_supplier(a: str, b: str) -> bool:
    """Compare supplier names tolerantly of company-form and punctuation noise."""
    return _normalise(a) == _normalise(b)


def _normalise(name: str) -> str:
    cleaned = name.lower().replace(",", " ").replace(".", " ")
    dropped = {"ab", "hb", "kb", "publ", "aktiebolag", "the"}
    return " ".join(w for w in cleaned.split() if w not in dropped)
