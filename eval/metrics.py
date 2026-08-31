"""Scoring functions, kept separate from the harness that calls them.

The organising idea is that **retrieval and generation fail differently and
must be measured separately**. A wrong answer has two possible causes: the
governing clause was never retrieved, or it was retrieved and misread. Those
have opposite fixes -- change the chunking or the reranker, versus change the
prompt or the model -- and a single end-to-end accuracy number tells you
nothing about which to do.

So `recall_at_k` is computed against `governing_clause_id` from the labels,
independently of what the model then concluded. When accuracy is 80%, the
first question is what recall was. If recall is 100%, every error is the
model's. If recall is 80%, the model may be doing fine with what it was given.

`ceiling` in the report makes that explicit: it is the accuracy the system
could reach if the interpreter were perfect, given the retrieval it actually
got. The gap between accuracy and ceiling is the interpretation problem; the
gap between ceiling and 1.0 is the retrieval problem.
"""

from __future__ import annotations

from collections import defaultdict


def recall_at_k(retrieved: list[str], gold: str | None) -> float:
    """Did the clause the label names appear anywhere in the candidate set?

    Binary per invoice: there is exactly one governing clause per labelled
    line, so this is recall, not precision. Precision is not scored because
    retrieving a few extra clauses is nearly free here -- the interpreter is
    asked to choose among them, and being shown an irrelevant clause is far
    cheaper than not being shown the right one.
    """
    if not gold:
        return 1.0
    return 1.0 if gold in retrieved else 0.0


def reciprocal_rank(retrieved: list[str], gold: str | None) -> float:
    """1/rank of the governing clause, 0 if absent.

    Recall says whether the clause was there; MRR says whether it was near the
    top. They diverge in the case that matters: if the right clause is
    consistently retrieved at rank 8, recall@10 looks perfect while the model
    is being handed seven distractors first.
    """
    if not gold:
        return 1.0
    for i, clause_id in enumerate(retrieved, start=1):
        if clause_id == gold:
            return 1.0 / i
    return 0.0


def summarise(rows: list[dict]) -> dict:
    """Aggregate per-invoice results into the report."""
    n = len(rows) or 1

    decision_correct = sum(r["decision_correct"] for r in rows)
    category_correct = sum(r["category_correct"] for r in rows)
    retrieval_hits = sum(r["recall"] for r in rows)
    mrr = sum(r["reciprocal_rank"] for r in rows) / n

    # The two errors are not symmetric and are never averaged into one number.
    # A false match is an overpayment of public money that nobody looks at; a
    # false flag is wasted caseworker time. Reporting one "accuracy" would hide
    # the trade the tolerances are actually tuned on.
    false_matches = [
        r["invoice_id"] for r in rows
        if r["expected_decision"] == "flag" and r["got_decision"] == "match"
    ]
    false_flags = [
        r["invoice_id"] for r in rows
        if r["expected_decision"] == "match" and r["got_decision"] == "flag"
    ]

    # What accuracy would have been achievable given the retrieval that
    # happened: an invoice whose governing clause was never retrieved could not
    # have been categorised correctly by any interpreter.
    ceiling = sum(1 for r in rows if r["recall"] == 1.0) / n

    by_category: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0})
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        bucket = by_category[row["expected_category"]]
        bucket["n"] += 1
        bucket["correct"] += row["category_correct"]
        confusion[row["expected_category"]][row["got_category"]] += 1

    return {
        "n": len(rows),
        "decision_accuracy": round(decision_correct / n, 4),
        "category_accuracy": round(category_correct / n, 4),
        "retrieval_recall": round(retrieval_hits / n, 4),
        "retrieval_mrr": round(mrr, 4),
        "category_ceiling_given_retrieval": round(ceiling, 4),
        "false_matches": false_matches,
        "false_flags": false_flags,
        "by_category": {
            k: {**v, "accuracy": round(v["correct"] / v["n"], 4)}
            for k, v in sorted(by_category.items())
        },
        "confusion": {k: dict(v) for k, v in sorted(confusion.items())},
        "latency_ms": _percentiles([r["latency_ms"] for r in rows]),
    }


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    def pick(p: float) -> float:
        idx = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
        return round(ordered[idx], 2)
    return {
        "p50": pick(0.5), "p95": pick(0.95),
        "max": round(ordered[-1], 2),
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def format_report(report: dict, title: str = "") -> str:
    """Human-readable summary for the terminal."""
    lines = []
    if title:
        lines += [title, "=" * len(title)]

    lines += [
        f"invoices                     : {report['n']}",
        f"decision accuracy            : {report['decision_accuracy']:.2%}",
        f"category accuracy            : {report['category_accuracy']:.2%}",
        "",
        "-- retrieval (scored independently of the model) --",
        f"recall of governing clause   : {report['retrieval_recall']:.2%}",
        f"MRR                          : {report['retrieval_mrr']:.3f}",
        f"category ceiling given that  : {report['category_ceiling_given_retrieval']:.2%}",
        "",
        "-- the two errors, kept apart --",
        f"false MATCHES (missed money) : {len(report['false_matches'])} "
        f"{report['false_matches'] or ''}",
        f"false FLAGS (wasted time)    : {len(report['false_flags'])} "
        f"{report['false_flags'] or ''}",
        "",
        "-- by category --",
    ]
    for name, stats in report["by_category"].items():
        lines.append(f"  {name:<26} {stats['correct']}/{stats['n']}  {stats['accuracy']:.0%}")

    misread = [
        f"    {exp} -> {got} ({n})"
        for exp, got_map in report["confusion"].items()
        for got, n in got_map.items()
        if exp != got
    ]
    if misread:
        lines += ["", "-- misclassifications --", *misread]

    lat = report["latency_ms"]
    if lat:
        lines += [
            "",
            f"latency ms  p50 {lat['p50']}  p95 {lat['p95']}  max {lat['max']}",
        ]
    return "\n".join(lines)
