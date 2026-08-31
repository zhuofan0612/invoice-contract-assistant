"""The tools the model is allowed to call, and the boundary around them.

Two rules govern what appears in this file, and both are about what is *not*
here.

**Every tool is a read.** There is no `decide`, no `compute_total`, no
`flag_line`. The model can look things up and nothing else; the verdict is
still produced by `app/core/check.py` after the loop has finished. Giving a
model a tool is giving it an action, so the safety argument that keeps
arithmetic out of the LLM would collapse the moment a `calculate` tool existed.

**The principal is bound at construction, not passed as an argument.** The
model chooses *what* to look up and never *who is asking*. If `contract_id`
were joined by a `groups` parameter, then prompt injection in a contract PDF
-- a supplier-supplied document, i.e. untrusted input that lands directly in
the model's context -- could widen its own access by asking for a different
group. Here, the worst an injected instruction can do is retrieve something
the caseworker was already entitled to read.

Tool results are also the reason the loop terminates cleanly: each result is
capped in size, so a model cannot inflate its own context by requesting the
same clause fifty times.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from app.config import Settings, get_settings
from app.models import Principal, RetrievedChunk
from app.retrieval.retriever import ClauseRetriever

MAX_RESULT_CHARS = 6000


@dataclass
class ToolResult:
    name: str
    content: str
    ok: bool = True
    meta: dict[str, Any] | None = None


# The schemas sent to the model. Kept deliberately narrow: a tool the model
# cannot misuse is better documentation than a warning in the prompt.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "retrieve_clauses",
        "description": (
            "Search the governing contract for clauses relevant to a query. "
            "Use this to find the clause that states a rate, a ceiling, or an "
            "exclusion. Returns clause IDs with their text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What you are looking for, in the contract's own "
                        "vocabulary. Prefer the language a contract would use "
                        "('maximum number of hours per month') over the "
                        "invoice's shorthand ('snr tech')."
                    ),
                },
                "contract_id": {
                    "type": "string",
                    "description": "The contract to search, e.g. C-2024-001.",
                },
            },
            "required": ["query", "contract_id"],
        },
    },
    {
        "name": "read_clause",
        "description": (
            "Read one clause in full by its ID, e.g. C-2024-001-§4.2. Use this "
            "when a retrieved excerpt looks truncated or refers to another "
            "clause you have not seen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "clause_id": {"type": "string"},
                "contract_id": {"type": "string"},
            },
            "required": ["clause_id", "contract_id"],
        },
    },
]


class ToolBox:
    """Executes tool calls on behalf of one principal.

    Construction binds the identity. `execute` takes only the name and the
    model's arguments, so there is no code path in which the model's output
    influences the access filter.
    """

    def __init__(
        self,
        principal: Principal,
        retriever: ClauseRetriever | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or ClauseRetriever(settings=self.settings)
        self.principal = principal
        self.calls: list[dict[str, Any]] = []
        # Every chunk the agent actually saw, in first-seen order. This is the
        # evidence set the hallucinated-citation guardrail checks against, and
        # what the decision reports as its candidate clauses.
        self.seen: dict[str, RetrievedChunk] = {}
        self._handlers: dict[str, Callable[[dict], ToolResult]] = {
            "retrieve_clauses": self._retrieve_clauses,
            "read_clause": self._read_clause,
        }

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run one tool call, converting every failure into a message.

        A tool that raises kills the loop; a tool that returns "that clause
        does not exist" lets the model correct itself on the next turn. Model
        arguments are untrusted input, so bad ones are an expected case rather
        than an exception.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(
                name=name, ok=False,
                content=f"No tool named {name!r}. Available: {', '.join(self._handlers)}.",
            )
        try:
            result = handler(arguments or {})
        except Exception as exc:  # noqa: BLE001 - reported to the model, not raised
            result = ToolResult(name=name, ok=False, content=f"Tool failed: {exc}")

        if len(result.content) > MAX_RESULT_CHARS:
            result.content = result.content[:MAX_RESULT_CHARS] + "\n...[truncated]"

        self.calls.append(
            {"name": name, "arguments": arguments, "ok": result.ok, **(result.meta or {})}
        )
        return result

    def _retrieve_clauses(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or "").strip()
        contract_id = str(args.get("contract_id") or "").strip()
        if not query or not contract_id:
            return ToolResult(
                name="retrieve_clauses", ok=False,
                content="Both 'query' and 'contract_id' are required.",
            )

        # principal is self.principal -- never from args.
        result = self.retriever.retrieve(query, self.principal, contract_id=contract_id)
        if not result.chunks:
            return ToolResult(
                name="retrieve_clauses",
                content=(
                    f"No clauses found in {contract_id} for that query. The "
                    "contract may not exist, may not be readable by this user, "
                    "or may simply not address this. Do not guess a clause ID."
                ),
                meta={"clause_ids": []},
            )

        for chunk in result.chunks:
            self.seen.setdefault(chunk.clause_id, chunk)

        payload = [
            {
                "clause_id": c.clause_id,
                "heading": c.heading,
                "text": c.text,
                "relevance": round(c.rerank_score, 3),
            }
            for c in result.chunks
        ]
        return ToolResult(
            name="retrieve_clauses",
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            meta={"clause_ids": [c.clause_id for c in result.chunks]},
        )

    def _read_clause(self, args: dict[str, Any]) -> ToolResult:
        clause_id = str(args.get("clause_id") or "").strip()
        contract_id = str(args.get("contract_id") or "").strip()
        if not clause_id:
            return ToolResult(
                name="read_clause", ok=False, content="'clause_id' is required."
            )
        if not contract_id and "-§" in clause_id:
            contract_id = clause_id.split("-§")[0]

        for chunk in self.retriever.contract_clauses(contract_id, self.principal):
            if chunk.clause_id == clause_id:
                self.seen.setdefault(chunk.clause_id, chunk)
                return ToolResult(
                    name="read_clause",
                    content=json.dumps(
                        {"clause_id": chunk.clause_id, "heading": chunk.heading,
                         "text": chunk.text},
                        ensure_ascii=False, indent=2,
                    ),
                    meta={"clause_ids": [clause_id]},
                )

        return ToolResult(
            name="read_clause",
            content=(
                f"No clause {clause_id!r} is readable in {contract_id!r}. If you "
                "inferred this ID rather than seeing it in a search result, it "
                "probably does not exist."
            ),
            meta={"clause_ids": []},
        )
