"""Check one invoice and print the decision the way a caseworker would read it.

    python scripts/check_invoice.py INV-008
    python scripts/check_invoice.py INV-008 --mode agent
    python scripts/check_invoice.py path/to/invoice.json --groups it-dept

This is the fastest way to see what the system actually produces, and the
output is deliberately shaped like the review screen rather than like JSON:
what was billed, what the contract says, the difference, and the clause to go
and read. If a flag cannot be explained in those four lines it is not worth
showing to anyone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.core.pipeline import DecisionPipeline  # noqa: E402
from app.models import Invoice, Principal  # noqa: E402


def resolve(reference: str, settings) -> Path:
    path = Path(reference)
    if path.exists():
        return path
    candidate = settings.invoices_dir / f"{reference}.json"
    if candidate.exists():
        return candidate
    raise SystemExit(f"no invoice found for {reference!r}")


def all_groups(settings) -> list[str]:
    access = json.loads(settings.access_path.read_text(encoding="utf-8"))
    rows = access["contracts"] if isinstance(access, dict) else access
    return sorted({g for row in rows for g in row.get("allowed_groups", [])})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("invoice", help="an invoice ID (INV-008) or a path to a JSON file")
    parser.add_argument("--mode", choices=["pipeline", "agent"], default="pipeline")
    parser.add_argument("--groups", default=None,
                        help="comma-separated groups for the caller "
                             "(default: every group, i.e. full access)")
    parser.add_argument("--json", action="store_true", help="print the raw decision")
    args = parser.parse_args()

    settings = get_settings()
    invoice = Invoice.model_validate(
        json.loads(resolve(args.invoice, settings).read_text(encoding="utf-8"))
    )
    groups = (
        [g.strip() for g in args.groups.split(",") if g.strip()]
        if args.groups else all_groups(settings)
    )
    principal = Principal(user_id="cli", groups=groups)

    decision = DecisionPipeline(settings=settings).run(invoice, principal, mode=args.mode)

    if args.json:
        print(decision.model_dump_json(indent=2))
        return 0

    money = f"{decision.total_delta:,.2f} {invoice.currency}"
    print()
    print(f"  {invoice.invoice_id}   {invoice.supplier}   contract {invoice.contract_id}")
    print(f"  period {invoice.period}   invoiced {invoice.invoice_total:,.2f} "
          f"{invoice.currency}")
    print()
    print(f"  DECISION: {decision.decision.value.upper()}"
          + (f"   findings: {', '.join(f.value for f in decision.findings)}"
             if decision.findings else ""))
    if decision.total_delta:
        print(f"  amount in question: {money}")
    print(f"  requires human review: {decision.requires_human_review}")
    print()

    for item in decision.line_decisions:
        mark = "OK  " if item.status.value == "match" else "FLAG"
        print(f"  [{mark}] line {item.line_no}: {item.description}")
        if item.stated_unit_price is not None:
            billed = f"{item.stated_unit_price:,.2f}"
            contract = (f"{item.contract_rate:,.2f}"
                        if item.contract_rate is not None else "not established")
            print(f"         billed {billed} / contract {contract}")
        print(f"         {item.explanation}")
        if item.cited_clause_id:
            print(f"         source: {item.cited_clause_id}")
        elif item.candidate_clause_ids:
            print(f"         considered: {', '.join(item.candidate_clause_ids[:4])}")
        print()

    if decision.errors:
        print("  errors:")
        for err in decision.errors:
            print(f"    - {err}")
        print()

    print(f"  trace: {decision.trace_id}   "
          f"backend: {decision.backend.get('llm_client')}/"
          f"{decision.backend.get('embedder')}/{decision.backend.get('store')}   "
          f"{decision.timings_ms.get('total_ms')} ms")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
