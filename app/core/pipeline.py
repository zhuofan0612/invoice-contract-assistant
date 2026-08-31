"""The orchestration: one invoice in, one reviewable decision out.

    validate -> lookup contract -> retrieve clauses -> interpret -> check -> flag

Two properties hold no matter which branch runs.

**Nothing is ever auto-approved.** `requires_human_review` is always True.
A "match" from this system means "no discrepancy found", not "pay it". The
design's error tolerance is low in both directions, and the value on offer is
cutting the time a caseworker spends looking, not removing the caseworker.

**Every path produces a citation or an explicit admission that there is none.**
A flag a caseworker cannot check is worse than no flag, because it costs them
the time the system was supposed to save.
"""

from __future__ import annotations

import time

from app.config import Settings, get_settings
from app.core.check import check_line
from app.core.interpret import Interpreter
from app.core.lookup import lookup_contract
from app.core.validate import validate_invoice
from app.models import (
    Decision,
    Finding,
    Invoice,
    InvoiceDecision,
    LineDecision,
    Principal,
)
from app.obs.metrics import METRICS
from app.obs.tracing import Tracer
from app.retrieval.retriever import ClauseRetriever


class DecisionPipeline:
    def __init__(
        self,
        retriever: ClauseRetriever | None = None,
        interpreter: Interpreter | None = None,
        tracer: Tracer | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or ClauseRetriever(settings=self.settings)
        self.interpreter = interpreter or Interpreter(
            retriever=self.retriever, settings=self.settings
        )
        self.tracer = tracer or Tracer(self.settings.trace_path, self.settings.redact_pii)
        self._agent_loop = None

    def _agent(self):
        """Built lazily so the default path never constructs an agent it won't use."""
        if self._agent_loop is None:
            from app.agent.loop import AgentLoop

            self._agent_loop = AgentLoop(
                retriever=self.retriever, llm=self.interpreter.llm,
                settings=self.settings,
            )
        return self._agent_loop

    def run(
        self, invoice: Invoice, principal: Principal, mode: str = "pipeline"
    ) -> InvoiceDecision:
        """Check one invoice.

        `mode` selects how clauses are found, and nothing else. Both modes feed
        the same `check_line`, so a difference in results is always a
        difference in evidence gathering, never in how the evidence is judged.
        """
        started = time.perf_counter()
        backend = {
            **self.settings.describe(),
            "mode": mode,
            "llm_client": self.interpreter.llm.name,
            "embedder": self.interpreter.retriever.embedder.name,
            "reranker": self.interpreter.retriever.reranker.name,
            "store": self.interpreter.retriever.store.name,
        }

        with self.tracer.trace(
            "check_invoice",
            invoice_id=invoice.invoice_id,
            contract_id=invoice.contract_id,
            user_id=principal.user_id,
            user_groups=principal.groups,
            backend=backend,
        ) as trace:

            # -- step 1: structural validation ------------------------------
            with trace.span("validate"):
                issues = validate_invoice(invoice)
            if issues:
                trace.event("rejected", reason="invalid_invoice", issues=issues)
                METRICS.increment("invoice_checks_total", outcome="invalid")
                return self._terminal(
                    invoice, Finding.INVALID_INVOICE,
                    "This invoice cannot be checked as submitted: " + "; ".join(issues),
                    trace, backend, started, errors=issues,
                )

            # -- step 2: deterministic contract lookup ----------------------
            with trace.span("contract_lookup"):
                lookup = lookup_contract(invoice, principal, self.retriever)
            if not lookup.ok:
                trace.event("rejected", reason="contract_lookup_failed", detail=lookup.error)
                METRICS.increment("invoice_checks_total", outcome="lookup_failed")
                return self._terminal(
                    invoice, Finding.NO_MATCHING_CLAUSE,
                    lookup.error or "the governing contract could not be identified",
                    trace, backend, started, errors=[lookup.error or "lookup failed"],
                )

            # -- steps 3-5: per line, retrieve -> interpret -> check --------
            line_decisions: list[LineDecision] = []
            retrieved: list[str] = []
            errors: list[str] = []

            for line in invoice.line_items:
                if mode == "agent":
                    # The agent chooses its own queries, so retrieval and
                    # interpretation are one span here rather than two.
                    with trace.span("agent", line_no=line.line_no) as span:
                        outcome, run = self._agent().interpret_line(
                            invoice, line, principal
                        )
                        span.attributes["iterations"] = run.iterations
                        span.attributes["stop_reason"] = run.stop_reason
                        span.attributes["tool_calls"] = run.tool_calls
                        span.attributes["clause_ids"] = run.clause_ids
                        METRICS.increment(
                            "agent_runs_total", stop_reason=run.stop_reason
                        )
                        METRICS.observe("agent_iterations", run.iterations)
                        errors.extend(self._record_outcome(span, outcome, line))
                else:
                    with trace.span("retrieve", line_no=line.line_no) as span:
                        clauses = self.interpreter.gather_clauses(
                            line, invoice.contract_id, principal
                        )
                        span.attributes["clause_ids"] = [c.clause_id for c in clauses]
                        span.attributes["scores"] = [
                            {"clause_id": c.clause_id,
                             "candidate": round(c.candidate_score, 4),
                             "rerank": round(c.rerank_score, 4)}
                            for c in clauses
                        ]

                    with trace.span("interpret", line_no=line.line_no) as span:
                        outcome = self.interpreter.interpret_line(
                            invoice, line, principal, clauses=clauses
                        )
                        errors.extend(self._record_outcome(span, outcome, line))

                with trace.span("check", line_no=line.line_no) as span:
                    decision = check_line(line, outcome, self.settings)
                    span.attributes["finding"] = decision.finding.value
                    span.attributes["status"] = decision.status.value
                    span.attributes["cited_clause_id"] = decision.cited_clause_id
                    span.attributes["delta"] = decision.delta

                line_decisions.append(decision)
                retrieved.extend(outcome.retrieved_clause_ids)
                METRICS.increment("decisions_total", finding=decision.finding.value)

            flagged = [d for d in line_decisions if d.status is Decision.FLAG]
            overall = Decision.FLAG if flagged else Decision.MATCH
            total_delta = round(sum(d.delta or 0.0 for d in line_decisions), 2)

            trace.set(
                decision=overall.value,
                findings=[d.finding.value for d in flagged],
                total_delta=total_delta,
            )
            METRICS.increment("invoice_checks_total", outcome=overall.value)
            METRICS.observe("invoice_check_latency_ms", (time.perf_counter() - started) * 1000)

            return InvoiceDecision(
                invoice_id=invoice.invoice_id,
                supplier=invoice.supplier,
                contract_id=invoice.contract_id,
                decision=overall,
                # Always true: the system flags, a human decides. A clean result
                # is "no discrepancy found", not an approval to pay.
                requires_human_review=True,
                findings=[d.finding for d in flagged],
                line_decisions=line_decisions,
                retrieved_clause_ids=list(dict.fromkeys(retrieved)),
                total_delta=total_delta,
                trace_id=trace.trace_id,
                timings_ms={
                    **trace.timings_ms,
                    "total_ms": round((time.perf_counter() - started) * 1000, 2),
                },
                backend=backend,
                errors=errors,
            )

    def _record_outcome(self, span, outcome, line) -> list[str]:
        """Write the interpretation onto the span that produced it.

        Must be called *inside* the span's `with` block. The tracer redacts
        attributes when the span closes, and `raw_model_output` is the one
        attribute most likely to contain PII echoed back out of a contract, so
        recording it after the span exits would write unredacted text to disk.

        Returns any error to surface on the decision, rather than appending to
        a captured list, so the caller keeps ownership of the errors it reports.
        """
        span.attributes["ok"] = outcome.ok
        if outcome.parse is not None:
            span.attributes["parse_strategy"] = outcome.parse.strategy
            span.attributes["parse_repairs"] = outcome.parse.repairs
            span.attributes["raw_model_output"] = outcome.parse.raw[:1500]
            if outcome.parse.was_repaired:
                METRICS.increment(
                    "llm_output_repairs_total", strategy=outcome.parse.strategy
                )
        if outcome.interpretation is not None:
            span.attributes["interpretation"] = outcome.interpretation.model_dump()
        for key, value in (outcome.usage or {}).items():
            METRICS.increment(f"llm_{key}_total", float(value))

        if outcome.error:
            span.attributes["error"] = outcome.error
            return [f"line {line.line_no}: {outcome.error}"]
        return []

    def _terminal(
        self, invoice: Invoice, finding: Finding, explanation: str,
        trace, backend: dict, started: float, errors: list[str],
    ) -> InvoiceDecision:
        """Short-circuit result for invoices that cannot reach the checker."""
        return InvoiceDecision(
            invoice_id=invoice.invoice_id,
            supplier=invoice.supplier,
            contract_id=invoice.contract_id,
            decision=Decision.FLAG,
            requires_human_review=True,
            findings=[finding],
            line_decisions=[
                LineDecision(
                    line_no=line.line_no, description=line.description,
                    status=Decision.FLAG, finding=finding, explanation=explanation,
                    stated_line_total=line.line_total, delta=line.line_total,
                )
                for line in invoice.line_items
            ],
            total_delta=round(sum(li.line_total for li in invoice.line_items), 2),
            trace_id=trace.trace_id,
            timings_ms={"total_ms": round((time.perf_counter() - started) * 1000, 2)},
            backend=backend,
            errors=errors,
        )
