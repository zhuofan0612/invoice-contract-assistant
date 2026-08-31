"""Generate the synthetic invoice dataset, ground-truth labels, and access metadata.

The contracts under data/contracts/ are hand-written prose. The invoices are
generated here so that the arithmetic in each test category is exactly what the
label claims: for "overbilling" the stated line_total deliberately disagrees
with quantity x unit_price, for "wrong_rate" the invoice is internally
consistent but the rate disagrees with the contract, and so on.

Run:  python scripts/make_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ---------------------------------------------------------------------------
# Invoice definitions.
#
# `stated_total` overrides quantity x unit_price. It is set ONLY for the
# overbilling category, where the supplier's stated line total is inflated.
# ---------------------------------------------------------------------------

INVOICES = [
    # -- 1. clean match -----------------------------------------------------
    dict(
        id="INV-001", supplier="Nordisk Konsult AB", contract="C-2024-001",
        issue_date="2025-04-05", period="2025-03",
        lines=[("Integration work, case mgmt system - implementation phase", 120, "hours", 1150.0, None)],
        category="clean_match", clause="C-2024-001-§4.2",
        discrepancy="", note="Rate, quantity and arithmetic all agree with the contract.",
    ),
    dict(
        id="INV-002", supplier="Björkdal Städservice AB", contract="C-2024-002",
        issue_date="2025-04-03", period="2025-03",
        lines=[("Routine cleaning, primary school units - scheduled visits", 18, "occasions", 485.0, None)],
        category="clean_match", clause="C-2024-002-§3.1",
        discrepancy="", note="18 occasions is within the 22/month ceiling.",
    ),
    dict(
        id="INV-003", supplier="Vasaberg Teknik AB", contract="C-2024-003",
        issue_date="2025-04-08", period="2025-03",
        lines=[("Preventive maintenance, ventilation plant - technician", 64, "hours", 745.0, None)],
        category="clean_match", clause="C-2024-003-§4.1",
        discrepancy="", note="Technician rate read from the rate table; within the 120h ceiling.",
    ),
    dict(
        id="INV-004", supplier="Kvarnholmen Livsmedel AB", contract="C-2024-007",
        issue_date="2025-03-24", period="2025-W12",
        lines=[("School lunch portions delivered, week 12", 3850, "portions", 68.50, None)],
        category="clean_match", clause="C-2024-007-§3.1",
        discrepancy="", note="3 850 portions is within the 4 000/week ceiling.",
    ),

    # -- 2. overbilling / wrong total --------------------------------------
    dict(
        id="INV-005", supplier="Sundberg IT-Drift AB", contract="C-2024-005",
        issue_date="2025-04-02", period="2025-03",
        lines=[("Managed client operations & support, 120 endpoints", 120, "devices", 175.0, 24000.0)],
        category="overbilling_wrong_total", clause="C-2024-005-§4.1",
        discrepancy="Line total stated as 24 000 SEK; 120 devices x 175 SEK = 21 000 SEK. Overstated by 3 000 SEK.",
        note="Correct tier (51-200) and correct unit price; the multiplication is wrong.",
    ),
    dict(
        id="INV-006", supplier="Björkdal Städservice AB", contract="C-2024-002",
        issue_date="2025-04-11", period="2025-03",
        lines=[("Window cleaning, interior and exterior surfaces", 4, "occasions", 1250.0, 6250.0)],
        category="overbilling_wrong_total", clause="C-2024-002-§3.2",
        discrepancy="Line total stated as 6 250 SEK; 4 occasions x 1 250 SEK = 5 000 SEK. Overstated by 1 250 SEK.",
        note="Looks like a fifth occasion was priced in but only four were performed.",
    ),
    dict(
        id="INV-007", supplier="Almgren Utbildning HB", contract="C-2024-006",
        issue_date="2025-04-15", period="2025-03",
        lines=[("CPD delivery - digital tools in teaching, training days", 6, "days", 12500.0, 82500.0)],
        category="overbilling_wrong_total", clause="C-2024-006-§4.1",
        discrepancy="Line total stated as 82 500 SEK; 6 days x 12 500 SEK = 75 000 SEK. Overstated by 7 500 SEK.",
        note="Contract is the scanned/OCR case, so this also exercises the OCR ingestion path.",
    ),

    # -- 3. wrong rate ------------------------------------------------------
    dict(
        id="INV-008", supplier="Nordisk Konsult AB", contract="C-2024-001",
        issue_date="2025-04-06", period="2025-03",
        lines=[("Config and go-live of procured software", 80, "hours", 1350.0, None)],
        category="wrong_rate", clause="C-2024-001-§4.2",
        discrepancy="Billed 1 350 SEK/h; contract implementation rate under §4.2 is 1 150 SEK/h.",
        note="Arithmetic is internally consistent, so only a rate comparison catches this.",
    ),
    dict(
        id="INV-009", supplier="Vasaberg Teknik AB", contract="C-2024-003",
        issue_date="2025-04-09", period="2025-03",
        lines=[("Statutory ventilation inspection (OVK), snr tech", 40, "hours", 985.0, None)],
        category="wrong_rate", clause="C-2024-003-§4.1",
        discrepancy="Billed 985 SEK/h; the rate table gives 895 SEK/h for senior technician.",
        note="Requires reading the correct row out of a Markdown table.",
    ),
    dict(
        id="INV-010", supplier="Kvarnholmen Livsmedel AB", contract="C-2024-007",
        issue_date="2025-04-01", period="2025-W13",
        lines=[("School lunch portions incl. special diets", 3000, "portions", 74.00, None)],
        category="wrong_rate", clause="C-2024-007-§3.1",
        discrepancy="Billed 74.00 SEK/portion; contract price is 68.50 SEK/portion.",
        note="§3.2 forbids a surcharge for special diets, which is the likely excuse for the uplift.",
    ),

    # -- 4. unauthorized quantity ------------------------------------------
    dict(
        id="INV-011", supplier="Nordisk Konsult AB", contract="C-2024-001",
        issue_date="2025-04-04", period="2025-03",
        lines=[("Integration work, running hours March", 195, "hours", 1150.0, None)],
        category="unauthorized_quantity", clause="C-2024-001-§5.1",
        discrepancy="195 hours invoiced; §5.1 caps call-offs at 160 hours per calendar month.",
        note="Correct rate and correct arithmetic; only the ceiling is breached.",
    ),
    dict(
        id="INV-012", supplier="Björkdal Städservice AB", contract="C-2024-002",
        issue_date="2025-05-05", period="2025-04",
        lines=[("Cleaning visits April, primary school units", 28, "occasions", 485.0, None)],
        category="unauthorized_quantity", clause="C-2024-002-§5.1",
        discrepancy="28 cleaning occasions invoiced; §5.1 allows a maximum of 22 per calendar month.",
        note="No written call-off for the additional occasions is referenced on the invoice.",
    ),

    # -- 5. item not in contract -------------------------------------------
    dict(
        id="INV-013", supplier="Nordisk Konsult AB", contract="C-2024-001",
        issue_date="2025-04-18", period="2025-03",
        lines=[("Purchase of 3 laptops for the project team", 3, "units", 9400.0, None)],
        category="item_not_in_contract", clause="C-2024-001-§3.2",
        discrepancy="Hardware purchase invoiced; §3.2 expressly excludes supply of computers and equipment.",
        note="The system should cite the exclusion clause, not merely fail to find a rate.",
    ),
    dict(
        id="INV-014", supplier="Björkdal Städservice AB", contract="C-2024-002",
        issue_date="2025-03-08", period="2025-02",
        lines=[("Snow clearance and gritting, car park at Ekskolan", 6, "occasions", 1850.0, None)],
        category="item_not_in_contract", clause="C-2024-002-§3.3",
        discrepancy="Snow clearance and gritting invoiced; §3.3 places outdoor/seasonal services outside the agreement.",
        note="Belongs under the grounds maintenance agreement instead.",
    ),

    # -- 6. ambiguous mapping ----------------------------------------------
    dict(
        id="INV-015", supplier="Nordisk Konsult AB", contract="C-2024-001",
        issue_date="2025-04-07", period="2025-03",
        lines=[("Advisory and technical support ahead of system replacement", 40, "hours", 1450.0, None)],
        category="ambiguous_mapping", clause="C-2024-001-§4.1",
        candidates=["C-2024-001-§4.1", "C-2024-001-§4.2"],
        discrepancy="Description mixes advisory (§4.1, 1 450 SEK/h) with technical support ahead of system replacement (§4.2, 1 150 SEK/h). The 300 SEK/h difference cannot be resolved from the invoice alone.",
        note="System must surface both candidate clauses and let a caseworker adjudicate.",
    ),
    dict(
        id="INV-016", supplier="Trelleborg Säkerhet AB", contract="C-2024-008",
        issue_date="2025-04-12", period="2025-03",
        lines=[("Security assignment, Radhuset site - perimeter protection review", 60, "hours", 1180.0, None)],
        category="ambiguous_mapping", clause="C-2024-008-§4.2",
        candidates=["C-2024-008-§4.1", "C-2024-008-§4.2"],
        discrepancy="'Security assignment' suggests manned guarding (§4.1, 520 SEK/h) while 'perimeter protection review' suggests consultancy (§4.2, 1 180 SEK/h). §4.3 requires the call-off to state the split; it is absent.",
        note="Ambiguity is worth 39 600 SEK, so guessing is exactly the wrong behaviour.",
    ),

    # -- 7. no matching contract clause ------------------------------------
    dict(
        id="INV-017", supplier="Lindqvist Anläggning AB", contract="C-2024-004",
        issue_date="2025-04-14", period="2025-03",
        lines=[("Design meetings and consultation with authorities", 12, "hours", 1100.0, None)],
        category="no_matching_clause", clause=None,
        discrepancy="Contract remunerates drainage on a per-running-metre basis only; it contains no hourly rate for meetings or authority consultation.",
        note="Needs human review: no clause governs this line at all.",
    ),
    dict(
        id="INV-018", supplier="Sundberg IT-Drift AB", contract="C-2024-005",
        issue_date="2025-04-16", period="2025-03",
        lines=[("One-off migration project fee, new platform", 1, "project", 145000.0, None)],
        category="no_matching_clause", clause=None,
        discrepancy="Contract prices per registered device per month and out-of-hours support per hour; it has no clause for one-off project fees.",
        note="Needs human review: a 145 000 SEK charge with no governing clause.",
    ),
]

ACCESS = [
    dict(contract_id="C-2024-001", department="IT", allowed_groups=["procurement", "it-dept"]),
    dict(contract_id="C-2024-002", department="Facilities", allowed_groups=["procurement", "facilities"]),
    dict(contract_id="C-2024-003", department="Facilities", allowed_groups=["procurement", "facilities", "property-mgmt"]),
    dict(contract_id="C-2024-004", department="Urban Development", allowed_groups=["procurement", "urban-dev"]),
    dict(contract_id="C-2024-005", department="IT", allowed_groups=["procurement", "it-dept"]),
    dict(contract_id="C-2024-006", department="Education", allowed_groups=["procurement", "education"]),
    dict(contract_id="C-2024-007", department="Education", allowed_groups=["procurement", "education", "school-catering"]),
    dict(contract_id="C-2024-008", department="Security", allowed_groups=["procurement", "security-dept"]),
]


def build_invoice(spec: dict) -> dict:
    lines = []
    for i, (desc, qty, unit, price, stated) in enumerate(spec["lines"], start=1):
        computed = round(qty * price, 2)
        lines.append({
            "line_no": i,
            "description": desc,
            "quantity": qty,
            "unit": unit,
            "unit_price": price,
            "line_total": stated if stated is not None else computed,
        })
    return {
        "invoice_id": spec["id"],
        "supplier": spec["supplier"],
        "contract_id": spec["contract"],
        "issue_date": spec["issue_date"],
        "period": spec["period"],
        "currency": "SEK",
        "line_items": lines,
        "invoice_total": round(sum(x["line_total"] for x in lines), 2),
    }


def build_label(spec: dict) -> dict:
    return {
        "invoice_id": spec["id"],
        "supplier": spec["supplier"],
        "contract_id": spec["contract"],
        "category": spec["category"],
        "expected_decision": "match" if spec["category"] == "clean_match" else "flag",
        "governing_clause_id": spec["clause"],
        "discrepancy": spec["discrepancy"],
        "ambiguous_clause_candidates": spec.get("candidates", []),
        "notes": spec["note"],
    }


def main() -> None:
    (DATA / "invoices").mkdir(parents=True, exist_ok=True)

    for spec in INVOICES:
        invoice = build_invoice(spec)
        path = DATA / "invoices" / f"{spec['id']}.json"
        path.write_text(json.dumps(invoice, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    labels = [build_label(s) for s in INVOICES]
    (DATA / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (DATA / "access.json").write_text(
        json.dumps(ACCESS, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    counts: dict[str, int] = {}
    for spec in INVOICES:
        counts[spec["category"]] = counts.get(spec["category"], 0) + 1
    print(f"Wrote {len(INVOICES)} invoices, labels.json, access.json")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:26s} {n}")


if __name__ == "__main__":
    main()
