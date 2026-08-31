"""The HTTP surface.

Thin on purpose. Every route does three things and no more: turn a request
into a `Principal`, hand it to a component that already knows how to do the
work, and hand back the result. There is no business logic in this file, which
is why the same decision can be produced from the CLI, the eval harness or a
test without going through HTTP at all.

**Where identity comes from.** The caseworker's groups arrive as headers here,
which is a deliberate stand-in, not a claim to have built authentication. In a
municipal deployment these would be OIDC claims validated against the identity
provider on every request, and this file is the only place that would change:
`_principal()` becomes a dependency that verifies a bearer token and reads the
same two fields out of it. Everything downstream already takes a `Principal`
and never asks where it came from.

The important consequence is that no route lets a caller widen its own access.
There is no `groups` parameter in any request body. `X-User-Groups` is what a
proxy asserts about the user, not what the user asks for.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, Response

from app.config import get_settings
from app.core.pipeline import DecisionPipeline
from app.ingest.pipeline import ingest
from app.models import CheckRequest, InvoiceDecision, Principal
from app.obs.metrics import METRICS
from app.retrieval.retriever import ClauseRetriever

app = FastAPI(
    title="Invoice-Contract Decision Assistant",
    version="0.1.0",
    description=(
        "Flags discrepancies between supplier invoices and the contracts that "
        "govern them. Never auto-approves: every result is for human review."
    ),
)

_retriever = ClauseRetriever()
_pipeline = DecisionPipeline(retriever=_retriever)


def _principal(
    x_user_id: str = Header(default="anonymous"),
    x_user_groups: str = Header(default=""),
) -> Principal:
    """Build the caller's identity from headers.

    Stand-in for OIDC. The groups string is the *asserted* group membership
    from an upstream proxy; it becomes the access-control filter applied
    inside the SQL search, and there is no way to override it per-request.
    """
    groups = [g.strip() for g in x_user_groups.split(",") if g.strip()]
    return Principal(user_id=x_user_id, groups=groups)


@app.get("/health")
def health() -> dict:
    """Liveness plus the resolved backends.

    The backend block is here because "which embedder was that?" is the first
    question asked of any result that looks wrong, and this service silently
    selects different implementations depending on what is installed.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "backends": settings.describe(),
        "clauses_indexed": _retriever.store.count(),
    }


@app.post("/ingest")
def run_ingest() -> dict:
    """Rebuild the clause index from `data/contracts`.

    Exposed as a route for convenience in a single-node demo. In a real
    deployment this is a job, not an endpoint, and would sit behind an
    admin-only scope -- reindexing is not a caseworker's operation.
    """
    report = ingest()
    return {
        "contracts": report.contracts,
        "chunks": report.chunks,
        "tables": report.tables,
        "ocr_documents": report.ocr_documents,
        "warnings": report.warnings,
    }


@app.post("/check", response_model=InvoiceDecision)
def check_invoice(
    request: CheckRequest, principal: Principal = Depends(_principal)
) -> InvoiceDecision:
    """Check one invoice against its contract.

    The `principal` comes from headers, never from the request body, so a
    caller cannot grant itself sight of a contract by editing the JSON it
    posts. `CheckRequest.principal` exists for in-process callers (tests, the
    eval harness) and is deliberately ignored here.

    Note there is no 4xx for "this invoice is bad". A malformed invoice or an
    unreadable contract is a *finding*, returned at 200 with the reason
    attached, because those are answers a caseworker needs to see rather than
    errors a client needs to retry.
    """
    return _pipeline.run(request.invoice, principal, mode=request.mode)


@app.get("/contracts/{contract_id}/clauses")
def contract_clauses(
    contract_id: str, principal: Principal = Depends(_principal)
) -> dict:
    """Read a contract's clauses, subject to the caller's access.

    This is the endpoint a caseworker follows when a decision cites §4.2 and
    they want to read §4.2 themselves. A flag they cannot verify is worth
    less than no flag, so being able to get from a citation to its source
    is part of the product, not a debug affordance.
    """
    clauses = _retriever.contract_clauses(contract_id, principal)
    if not clauses:
        # Deliberately not distinguishing "no such contract" from "you may not
        # read it". The difference leaks which suppliers the municipality has
        # contracts with, which is exactly what the access rules protect.
        raise HTTPException(status_code=404, detail="contract not found")
    return {
        "contract_id": contract_id,
        "clauses": [
            {"clause_id": c.clause_id, "heading": c.heading, "text": c.text}
            for c in clauses
        ],
    }


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus exposition format."""
    return Response(content=METRICS.render_prometheus(), media_type="text/plain")
