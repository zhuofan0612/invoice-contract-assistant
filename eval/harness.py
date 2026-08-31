"""Runs the labelled set through the pipeline and scores the result.

This exists so that "does it work?" has an answer that is a number rather than
a demo. Every claim made about this system -- that the second cap query fixed
the volume check, that the agent path buys accuracy at the cost of latency --
came from re-running this and comparing, and any of them can be re-checked by
running it again.

The harness deliberately calls `DecisionPipeline` in-process rather than over
HTTP. Scoring the transport is not interesting, and going direct means an eval
run needs no server and can be a test.

**On the eval principal.** The harness runs as a user in every group, so
retrieval is never the constraint. Access control is not measured here; it is
tested separately in `tests/test_access.py`, because mixing "can this user see
it" into an accuracy number produces a figure that means neither thing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.core.pipeline import DecisionPipeline
from app.models import Invoice, Principal
from eval.metrics import recall_at_k, reciprocal_rank, summarise


@dataclass
class EvalCase:
    invoice: Invoice
    category: str
    expected_decision: str
    governing_clause_id: str | None


def load_cases(settings: Settings | None = None) -> list[EvalCase]:
    settings = settings or get_settings()
    labels = json.loads(Path(settings.labels_path).read_text(encoding="utf-8"))
    rows = labels["invoices"] if isinstance(labels, dict) else labels
    by_id = {row["invoice_id"]: row for row in rows}

    cases = []
    for path in sorted(Path(settings.invoices_dir).glob("*.json")):
        invoice = Invoice.model_validate(json.loads(path.read_text(encoding="utf-8")))
        label = by_id.get(invoice.invoice_id)
        if label is None:
            continue
        cases.append(EvalCase(
            invoice=invoice,
            category=label["category"],
            expected_decision=label["expected_decision"],
            governing_clause_id=label.get("governing_clause_id") or None,
        ))
    return cases


def eval_principal(settings: Settings | None = None) -> Principal:
    """A user in every group: makes retrieval, not permissions, the variable."""
    settings = settings or get_settings()
    access = json.loads(Path(settings.access_path).read_text(encoding="utf-8"))
    rows = access["contracts"] if isinstance(access, dict) else access
    groups = sorted({g for row in rows for g in row.get("allowed_groups", [])})
    return Principal(user_id="eval-harness", groups=groups)


def run_eval(
    mode: str = "pipeline",
    pipeline: DecisionPipeline | None = None,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    pipeline = pipeline or DecisionPipeline(settings=settings)
    principal = eval_principal(settings)

    rows = []
    for case in load_cases(settings):
        started = time.perf_counter()
        decision = pipeline.run(case.invoice, principal, mode=mode)
        elapsed = (time.perf_counter() - started) * 1000

        got_category = (
            decision.findings[0].value if decision.findings else "clean_match"
        )
        retrieved = decision.retrieved_clause_ids

        rows.append({
            "invoice_id": case.invoice.invoice_id,
            "expected_category": case.category,
            "got_category": got_category,
            "expected_decision": case.expected_decision,
            "got_decision": decision.decision.value,
            "category_correct": int(got_category == case.category),
            "decision_correct": int(decision.decision.value == case.expected_decision),
            "gold_clause": case.governing_clause_id,
            "cited_clause": next(
                (d.cited_clause_id for d in decision.line_decisions if d.cited_clause_id),
                None,
            ),
            "recall": recall_at_k(retrieved, case.governing_clause_id),
            "reciprocal_rank": reciprocal_rank(retrieved, case.governing_clause_id),
            "latency_ms": elapsed,
            "trace_id": decision.trace_id,
            "errors": decision.errors,
        })

    report = summarise(rows)
    report["mode"] = mode
    report["backend"] = settings.describe()
    report["rows"] = rows
    return report
