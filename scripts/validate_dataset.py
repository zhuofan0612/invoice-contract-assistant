"""Verify the golden set is internally consistent.

An evaluation harness is only as good as its labels. A `governing_clause_id`
pointing at a clause that does not exist would silently make retrieval recall
unreachable, and the eval would report a model failure that is really a data
bug. This script fails loudly instead.

It resolves clause IDs through the *real* chunker, so if chunking changes in a
way that breaks citation IDs, this catches it.

Run:  python scripts/validate_dataset.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.ingest.pipeline import build_chunks
from app.models import Invoice

EXPECTED_COUNTS = {
    "clean_match": 4,
    "overbilling_wrong_total": 3,
    "wrong_rate": 3,
    "unauthorized_quantity": 2,
    "item_not_in_contract": 2,
    "ambiguous_mapping": 2,
    "no_matching_clause": 2,
}
TOLERANCE = 0.01


def main() -> int:
    settings = get_settings()
    errors: list[str] = []
    notes: list[str] = []

    chunks, _ = build_chunks(settings)
    known_clauses = {c.clause_id for c in chunks if c.clause_id}
    known_contracts = {c.contract_id for c in chunks}

    labels = json.loads(settings.labels_path.read_text(encoding="utf-8"))
    access = json.loads(settings.access_path.read_text(encoding="utf-8"))
    labels_by_id = {lb["invoice_id"]: lb for lb in labels}

    invoice_paths = sorted(settings.invoices_dir.glob("*.json"))
    invoice_ids = set()

    # -- invoices parse, and their arithmetic matches their category ---------
    for path in invoice_paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        try:
            invoice = Invoice.model_validate(raw)
        except Exception as exc:
            errors.append(f"{path.name}: does not validate against Invoice schema: {exc}")
            continue

        invoice_ids.add(invoice.invoice_id)
        label = labels_by_id.get(invoice.invoice_id)
        if label is None:
            errors.append(f"{invoice.invoice_id}: no entry in labels.json")
            continue

        if invoice.contract_id != label["contract_id"]:
            errors.append(f"{invoice.invoice_id}: contract_id disagrees with its label")
        if invoice.supplier != label["supplier"]:
            errors.append(f"{invoice.invoice_id}: supplier disagrees with its label")

        summed = round(sum(li.line_total for li in invoice.line_items), 2)
        if abs(summed - invoice.invoice_total) > TOLERANCE:
            errors.append(
                f"{invoice.invoice_id}: invoice_total {invoice.invoice_total} "
                f"!= sum of line totals {summed}"
            )

        inconsistent = [
            li.line_no for li in invoice.line_items
            if abs(li.quantity * li.unit_price - li.line_total) > TOLERANCE
        ]
        category = label["category"]
        if category == "overbilling_wrong_total":
            if not inconsistent:
                errors.append(
                    f"{invoice.invoice_id}: labelled overbilling but every line's "
                    "quantity x unit_price equals its stated total"
                )
        elif inconsistent:
            errors.append(
                f"{invoice.invoice_id}: labelled {category} but line(s) {inconsistent} "
                "have arithmetic errors, which would make the category ambiguous"
            )

    for label in labels:
        if label["invoice_id"] not in invoice_ids:
            errors.append(f"{label['invoice_id']}: label has no invoice file")

    # -- labels are self-consistent and cite real clauses --------------------
    for label in labels:
        inv = label["invoice_id"]
        category = label["category"]
        clause = label.get("governing_clause_id")
        candidates = label.get("ambiguous_clause_candidates", [])

        expected = "match" if category == "clean_match" else "flag"
        if label["expected_decision"] != expected:
            errors.append(f"{inv}: expected_decision should be '{expected}' for {category}")

        if label["contract_id"] not in known_contracts:
            errors.append(f"{inv}: contract {label['contract_id']} was not ingested")

        if category == "no_matching_clause":
            if clause is not None:
                errors.append(f"{inv}: no_matching_clause must have a null governing_clause_id")
        else:
            if clause is None:
                errors.append(f"{inv}: {category} requires a governing_clause_id")
            elif clause not in known_clauses:
                errors.append(f"{inv}: governing_clause_id {clause} is not a real clause")
            elif not clause.startswith(label["contract_id"]):
                errors.append(f"{inv}: governing_clause_id {clause} belongs to another contract")

        if category == "ambiguous_mapping":
            if len(candidates) < 2:
                errors.append(f"{inv}: ambiguous_mapping needs >= 2 candidate clauses")
            for cand in candidates:
                if cand not in known_clauses:
                    errors.append(f"{inv}: candidate clause {cand} is not a real clause")
        elif candidates:
            errors.append(f"{inv}: only ambiguous_mapping should list candidate clauses")

        if category != "clean_match" and not label.get("discrepancy"):
            errors.append(f"{inv}: flagged categories must describe the discrepancy")

    # -- category distribution ----------------------------------------------
    counts = Counter(lb["category"] for lb in labels)
    for category, expected_n in EXPECTED_COUNTS.items():
        if counts.get(category, 0) != expected_n:
            errors.append(
                f"category {category}: expected {expected_n}, found {counts.get(category, 0)}"
            )
    unknown = set(counts) - set(EXPECTED_COUNTS)
    if unknown:
        errors.append(f"unexpected categories: {sorted(unknown)}")

    # -- access metadata -----------------------------------------------------
    access_ids = {a["contract_id"] for a in access}
    missing = known_contracts - access_ids
    if missing:
        errors.append(f"contracts with no access.json entry: {sorted(missing)}")

    group_sets = {tuple(sorted(a["allowed_groups"])) for a in access}
    if len(group_sets) < 3:
        errors.append("allowed_groups barely vary; access-control tests would be weak")
    notes.append(f"{len(group_sets)} distinct access-group combinations across {len(access)} contracts")

    all_groups = sorted({g for a in access for g in a["allowed_groups"]})
    notes.append(f"groups in use: {', '.join(all_groups)}")

    # -- report --------------------------------------------------------------
    print(f"contracts ingested     : {len(known_contracts)}")
    print(f"clause chunks          : {len(chunks)}")
    print(f"distinct clause IDs    : {len(known_clauses)}")
    print(f"invoices               : {len(invoice_paths)}")
    print()
    print("category distribution")
    for category, n in sorted(counts.items()):
        print(f"  {category:26s} {n}")
    print()
    for note in notes:
        print(f"note: {note}")

    if errors:
        print(f"\nFAILED with {len(errors)} problem(s):")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("\nOK: every governing_clause_id resolves to a real clause, every")
    print("invoice's arithmetic matches its labelled category, and the category")
    print("distribution is as specified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
