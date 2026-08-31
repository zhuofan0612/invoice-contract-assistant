"""The bounded tool-calling loop.

A loop that works when the model cooperates is not the interesting case. These
tests use deliberately badly-behaved models -- one that never stops, one that
asks the same question forever, one that cites a clause it never saw -- because
those are the behaviours the bounds exist for, and they are hard to provoke on
demand from a real model.

The through-line: **every way of running out must fail closed.** Exhaustion
produces a line routed to a human, never a guess.
"""

from __future__ import annotations

import pytest

from app.agent.loop import AgentLoop
from app.agent.tools import ToolBox
from app.config import get_settings
from app.llm.base import LLMResponse, ToolCall
from app.models import Invoice, LineItem, Principal

SETTINGS = get_settings()
PRINCIPAL = Principal(user_id="anna", groups=["it-dept", "procurement"])


@pytest.fixture
def invoice() -> Invoice:
    import json
    return Invoice.model_validate(
        json.loads((SETTINGS.invoices_dir / "INV-008.json").read_text(encoding="utf-8"))
    )


class ScriptedLLM:
    """A model whose behaviour is fixed in advance."""

    name = "scripted"

    def __init__(self, *, mode: str):
        self.mode = mode
        self.calls = 0

    def complete(self, system, messages, tools=None, max_tokens=None):
        self.calls += 1
        if not tools:
            # The forced final turn.
            return LLMResponse(text=self._final(), model="scripted")
        if self.mode == "never_stops":
            return self._tool_call(f"query {self.calls}")
        if self.mode == "repeats":
            return self._tool_call("the same query")
        # The cooperative modes search first, then answer -- the guardrail
        # rejects any citation the agent did not actually retrieve, so
        # answering on turn one would (correctly) be refused.
        if self.calls == 1:
            return self._tool_call("consultancy implementation hourly rate")
        if self.mode == "hallucinates":
            return LLMResponse(text=self._final(clause="C-2024-001-§99.9"),
                               model="scripted")
        return LLMResponse(text=self._final(), model="scripted")

    def _tool_call(self, query: str) -> LLMResponse:
        return LLMResponse(
            stop_reason="tool_use", model="scripted",
            tool_calls=[ToolCall(id=f"c{self.calls}", name="retrieve_clauses",
                                 arguments={"query": query,
                                            "contract_id": "C-2024-001"})],
        )

    def _final(self, clause: str = "C-2024-001-§4.2") -> str:
        return (
            '{"line_no": 1, "governing_clause_id": "%s", "covered_by_contract": true,'
            ' "contract_rate": 1150, "rate_unit": "hour", "confidence": 0.9,'
            ' "reasoning": "scripted"}' % clause
        )


def run(mode: str, invoice: Invoice):
    loop = AgentLoop(llm=ScriptedLLM(mode=mode), settings=SETTINGS)
    return loop.interpret_line(invoice, invoice.line_items[0], PRINCIPAL)


def test_a_cooperative_model_finishes_and_is_trusted(invoice):
    outcome, agent_run = run("answers", invoice)
    assert agent_run.stop_reason == "answered"
    assert outcome.ok
    assert outcome.interpretation.contract_rate == 1150.0


def test_a_model_that_never_stops_hits_the_iteration_cap(invoice):
    outcome, agent_run = run("never_stops", invoice)
    assert agent_run.stop_reason == "iteration_cap"
    assert agent_run.iterations == SETTINGS.max_agent_iterations
    assert not outcome.ok, "an exhausted run must not be treated as an answer"


def test_a_model_repeating_itself_is_stopped_early(invoice):
    """Cheaper than waiting out the cap, and catches the common two-call cycle."""
    outcome, agent_run = run("repeats", invoice)
    assert agent_run.stop_reason == "repeated_calls"
    assert agent_run.iterations < SETTINGS.max_agent_iterations
    assert not outcome.ok


def test_exhaustion_routes_the_line_to_a_human(invoice):
    """The decision-level consequence of running out of budget."""
    from app.core.check import check_line

    outcome, _ = run("never_stops", invoice)
    decision = check_line(invoice.line_items[0], outcome, SETTINGS)
    assert decision.status.value == "flag"
    assert decision.finding.value == "interpretation_failed"


def test_a_cited_clause_that_was_never_retrieved_is_rejected(invoice):
    """The hallucination guardrail.

    §99.9 does not exist, but it is well-formed enough that a caseworker would
    have to go and look before finding that out -- which is exactly the time
    the system is meant to save.
    """
    outcome, _ = run("hallucinates", invoice)
    assert not outcome.ok
    assert "never retrieved" in outcome.error


def test_tool_errors_are_returned_to_the_model_not_raised():
    """A raised exception ends the run; a message lets the model correct itself."""
    box = ToolBox(PRINCIPAL)
    assert box.execute("no_such_tool", {}).ok is False
    assert box.execute("retrieve_clauses", {}).ok is False
    assert box.execute("read_clause", {"clause_id": "C-2024-001-§99.9"}).ok is True


def test_tool_results_are_size_capped():
    """Bounds context growth, so a model cannot inflate its own prompt."""
    from app.agent.tools import MAX_RESULT_CHARS

    box = ToolBox(PRINCIPAL)
    result = box.execute("retrieve_clauses",
                         {"query": "rate hour work", "contract_id": "C-2024-001"})
    assert len(result.content) <= MAX_RESULT_CHARS + 20


def test_the_toolbox_offers_no_way_to_decide_anything():
    """The safety boundary: tools read, `check_line` decides.

    A `calculate` or `approve` tool would move arithmetic back inside the
    model, undoing the split the whole design rests on.
    """
    names = {schema["name"] for schema in ToolBox(PRINCIPAL).schemas}
    assert names == {"retrieve_clauses", "read_clause"}
    forbidden = {"calculate", "compute", "decide", "approve", "flag", "pay"}
    assert not any(f in n for n in names for f in forbidden)
