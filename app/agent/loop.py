"""The bounded tool-calling loop.

Written by hand rather than taken from a framework, because the interesting
part of an agent is not that it calls tools -- that is twenty lines -- but what
happens when it will not stop. The loop below is the same twenty lines plus the
answers to "what if it doesn't terminate", and those answers are the design.

**Four independent bounds, because each fails differently.**

1. *Iteration cap.* The hard stop. A model that keeps calling tools forever is
   spending money and a caseworker's afternoon on nothing.
2. *Repeat-call detection.* A model that asks the identical question twice has
   not learned anything from the first answer and will not learn from the
   third. Cheaper to detect than to wait out the iteration cap, and it catches
   the common failure -- a loop of two alternating calls -- much earlier.
3. *No-progress detection.* Tool calls that return nothing useful (empty
   results, unknown clause IDs) mean the model is searching for something that
   is not there. After a few of those the honest answer is "not found".
4. *A final forced turn.* On hitting a bound, the model is asked once more
   *without tools*, so it must answer from what it already has.

**Running out of budget is not an error.** It returns the model's best
available answer marked `exhausted`, and downstream that produces
`INTERPRETATION_FAILED` -- a line routed to a human. The system is allowed to
run out of ideas; it is not allowed to invent a clause because it did.

**The loop cannot decide anything.** It gathers evidence and returns a parsed
interpretation. `check_line` still renders the verdict afterwards, exactly as
in the non-agent path, so both paths share one arithmetic implementation and
one set of tolerances.

Why have this path at all when the fixed pipeline scores well? The pipeline
issues a fixed set of queries; the agent can notice that §4.1 refers to a
schedule in §4.3 and go read it. That matters for contracts whose clauses
cross-reference, and it is measurably worse on latency and cost. Both paths
exist so the eval harness can put a number on that trade rather than an
opinion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.agent.tools import ToolBox
from app.config import Settings, get_settings
from app.core.interpret import InterpretationOutcome, _normalise_interpretation
from app.llm.base import LLMClient, get_llm_client
from app.llm.parsing import parse_model
from app.llm.prompts import AGENT_SYSTEM, build_agent_prompt
from app.models import ClauseInterpretation, Invoice, LineItem, Principal
from app.retrieval.retriever import ClauseRetriever

MAX_REPEATED_CALLS = 2
MAX_EMPTY_CALLS = 3


@dataclass
class AgentRun:
    """Everything needed to explain afterwards what the agent did and why it stopped."""

    iterations: int = 0
    stop_reason: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    clause_ids: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    final_text: str = ""

    @property
    def exhausted(self) -> bool:
        return self.stop_reason in {"iteration_cap", "repeated_calls", "no_progress"}


class AgentLoop:
    def __init__(
        self,
        retriever: ClauseRetriever | None = None,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or ClauseRetriever(settings=self.settings)
        self.llm = llm or get_llm_client(self.settings)

    def interpret_line(
        self, invoice: Invoice, line: LineItem, principal: Principal
    ) -> tuple[InterpretationOutcome, AgentRun]:
        toolbox = ToolBox(principal, retriever=self.retriever, settings=self.settings)
        run = AgentRun()

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": build_agent_prompt(invoice, line)}
        ]

        seen_calls: dict[str, int] = {}
        empty_calls = 0

        while run.iterations < self.settings.max_agent_iterations:
            run.iterations += 1

            response = self.llm.complete(
                system=AGENT_SYSTEM, messages=messages, tools=toolbox.schemas
            )
            _accumulate(run.usage, response.usage)

            if not response.wants_tools:
                run.stop_reason = "answered"
                run.final_text = response.text
                return self._finish(run, line, toolbox), run

            # Record the assistant's tool-use turn before appending results, so
            # the transcript stays in the shape the API requires.
            messages.append(_assistant_turn(response))

            results = []
            for call in response.tool_calls:
                signature = _signature(call.name, call.arguments)
                seen_calls[signature] = seen_calls.get(signature, 0) + 1

                result = toolbox.execute(call.name, call.arguments)
                run.clause_ids.extend((result.meta or {}).get("clause_ids", []))

                if seen_calls[signature] > MAX_REPEATED_CALLS:
                    # Say so in the transcript rather than silently dropping it.
                    # A model told "you already asked this" usually stops; a
                    # model whose tool call vanishes tends to retry it.
                    result.content = (
                        "You have already made this exact call and received an "
                        "answer. Do not repeat it. Answer with what you have, "
                        "or state that the contract does not say."
                    )
                if not (result.meta or {}).get("clause_ids", ["x"]):
                    empty_calls += 1

                results.append(_tool_result_block(call.id, result.content, result.ok))

            messages.append({"role": "user", "content": results})

            if max(seen_calls.values(), default=0) > MAX_REPEATED_CALLS:
                run.stop_reason = "repeated_calls"
                break
            if empty_calls >= MAX_EMPTY_CALLS:
                run.stop_reason = "no_progress"
                break
        else:
            run.stop_reason = "iteration_cap"

        # Bound hit: one last turn with no tools offered, so the only thing the
        # model can do is answer from the evidence already in the transcript.
        messages.append({"role": "user", "content": FORCED_ANSWER})
        try:
            final = self.llm.complete(system=AGENT_SYSTEM, messages=messages)
            _accumulate(run.usage, final.usage)
            run.final_text = final.text
        except Exception as exc:  # noqa: BLE001
            run.final_text = ""
            run.stop_reason = f"{run.stop_reason}+final_call_failed:{exc}"

        run.tool_calls = toolbox.calls
        return self._finish(run, line, toolbox), run

    def _finish(
        self, run: AgentRun, line: LineItem, toolbox: ToolBox
    ) -> InterpretationOutcome:
        run.tool_calls = toolbox.calls
        chunks = list(toolbox.seen.values())
        seen = set(toolbox.seen)

        parsed = parse_model(
            run.final_text, ClauseInterpretation, normalise=_normalise_interpretation
        )
        if not parsed.ok or parsed.value is None:
            return InterpretationOutcome(
                line_no=line.line_no, clauses=chunks, parse=parsed, usage=run.usage,
                error=(
                    f"agent produced no usable interpretation after "
                    f"{run.iterations} iterations ({run.stop_reason}): {parsed.error}"
                ),
            )

        interpretation = parsed.value
        interpretation.line_no = line.line_no

        # Same guardrail as the fixed pipeline: a clause the agent never
        # actually retrieved is a hallucinated citation, and it is worse here
        # because the agent chose its own queries and may have invented an ID
        # that merely looks like the ones it saw.
        if interpretation.governing_clause_id and interpretation.governing_clause_id not in seen:
            return InterpretationOutcome(
                line_no=line.line_no, clauses=chunks, parse=parsed, usage=run.usage,
                error=(
                    f"agent cited clause {interpretation.governing_clause_id!r}, "
                    "which it never retrieved"
                ),
            )
        interpretation.alternative_clause_ids = [
            c for c in interpretation.alternative_clause_ids if c in seen
        ]

        if run.exhausted:
            return InterpretationOutcome(
                line_no=line.line_no, clauses=chunks, parse=parsed, usage=run.usage,
                error=f"agent stopped without converging ({run.stop_reason})",
            )

        return InterpretationOutcome(
            line_no=line.line_no, interpretation=interpretation,
            clauses=chunks, parse=parsed, usage=run.usage,
        )


FORCED_ANSWER = (
    "Stop searching and answer now with the JSON object, using only the "
    "clauses you have already retrieved. If they do not establish a governing "
    "rate, say so with governing_clause_id set to null. Do not invent a clause ID."
)


def _signature(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"


def _assistant_turn(response) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if response.text:
        content.append({"type": "text", "text": response.text})
    for call in response.tool_calls:
        content.append({
            "type": "tool_use", "id": call.id,
            "name": call.name, "input": call.arguments,
        })
    return {"role": "assistant", "content": content}


def _tool_result_block(call_id: str, content: str, ok: bool) -> dict[str, Any]:
    block = {"type": "tool_result", "tool_use_id": call_id, "content": content}
    if not ok:
        block["is_error"] = True
    return block


def _accumulate(total: dict[str, int], usage: dict[str, int] | None) -> None:
    for key, value in (usage or {}).items():
        total[key] = total.get(key, 0) + int(value)
