"""Access control: the tests that would matter most if they broke.

Everything else in this suite guards accuracy. These guard a data leak, which
is why they check the property from several directions rather than once. The
claim under test is not "the API returns 404" -- that is a symptom. It is that
**the filter is applied during retrieval, so unauthorised text never enters
the process**, and therefore cannot reach a prompt, a trace file or a log line
even if the layers above are wrong.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.models import Invoice, Principal
from app.retrieval.retriever import ClauseRetriever
from app.security.access import filter_for

IT_CONTRACT = "C-2024-001"       # allowed_groups includes it-dept
EDU_CONTRACT = "C-2024-006"      # education only


@pytest.fixture(scope="module")
def retriever() -> ClauseRetriever:
    return ClauseRetriever()


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_search_returns_nothing_for_a_user_outside_the_group(retriever):
    """The core property: no chunks come back, rather than coming back and being hidden."""
    outsider = Principal(user_id="bo", groups=["education"])
    result = retriever.retrieve("consultancy rate", outsider, contract_id=IT_CONTRACT)
    assert result.chunks == []


def test_same_query_succeeds_for_an_authorised_user(retriever):
    """Guards against the test above passing because retrieval is simply broken."""
    insider = Principal(user_id="anna", groups=["it-dept"])
    result = retriever.retrieve("consultancy rate", insider, contract_id=IT_CONTRACT)
    assert result.chunks, "authorised user should see clauses"
    assert all(c.clause_id.startswith(IT_CONTRACT) for c in result.chunks)


def test_a_user_with_no_groups_sees_nothing(retriever):
    """Deny by default. An empty group list must not mean 'unfiltered'."""
    nobody = Principal(user_id="nobody", groups=[])
    for contract in (IT_CONTRACT, EDU_CONTRACT):
        assert retriever.retrieve("rate", nobody, contract_id=contract).chunks == []


def test_unfiltered_search_still_respects_groups(retriever):
    """Without a contract_id the filter must still apply.

    Omitting contract_id widens the search to every contract, which is exactly
    where a missing group predicate would show up as a cross-department leak.
    """
    edu_only = Principal(user_id="bo", groups=["education"])
    chunks = retriever.retrieve("hourly rate excluding VAT", edu_only).chunks
    assert chunks, "should find something in the contracts this user may read"
    settings = get_settings()
    access = json.loads(settings.access_path.read_text(encoding="utf-8"))
    rows = access["contracts"] if isinstance(access, dict) else access
    permitted = {
        row["contract_id"] for row in rows
        if "education" in row.get("allowed_groups", [])
    }
    assert {c.clause_id.split("-§")[0] for c in chunks} <= permitted


def test_filter_carries_the_groups_it_was_given():
    """The filter object itself, since every backend depends on it being right."""
    principal = Principal(user_id="anna", groups=["it-dept", "procurement"])
    filters = filter_for(principal, contract_id=IT_CONTRACT)
    assert set(filters.allowed_groups) == {"it-dept", "procurement"}
    assert filters.contract_id == IT_CONTRACT


def test_request_body_cannot_widen_access(client):
    """Identity comes from headers; the body is not trusted to supply it.

    A caller who could name its own groups in the JSON it posts would have no
    access control at all, so this asserts the body's principal is ignored.
    """
    invoice = json.loads(
        (get_settings().invoices_dir / "INV-008.json").read_text(encoding="utf-8")
    )
    response = client.post(
        "/check",
        json={
            "invoice": invoice,
            "principal": {"user_id": "attacker", "groups": ["it-dept", "procurement"]},
        },
        headers={"X-User-Id": "bo", "X-User-Groups": "education"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["findings"] == ["no_matching_clause"]
    # Nothing *from the contract* leaked. Note the response does echo the
    # contract_id, which is not a disclosure: the caller supplied it in the
    # invoice they posted, so it tells them nothing they did not already
    # assert. What must not appear is anything only readable from the store --
    # clause IDs, headings, or the rates themselves.
    assert body["retrieved_clause_ids"] == []
    rendered = json.dumps(body["line_decisions"])
    assert "§" not in rendered
    assert "1 150" not in rendered and "1150" not in rendered


def test_forbidden_contract_is_indistinguishable_from_a_missing_one(client):
    """Not a 403: 'you may not read this' confirms the contract exists.

    Which supplier the municipality contracts with is itself the information
    the access rules protect, so both cases answer 404.
    """
    headers = {"X-User-Groups": "education"}
    forbidden = client.get(f"/contracts/{IT_CONTRACT}/clauses", headers=headers)
    missing = client.get("/contracts/C-9999-999/clauses", headers=headers)
    assert forbidden.status_code == missing.status_code == 404
    assert forbidden.json() == missing.json()


def test_decision_for_an_unreadable_contract_reveals_nothing(client):
    """A denied check must not leak the contract through the explanation text."""
    invoice = json.loads(
        (get_settings().invoices_dir / "INV-001.json").read_text(encoding="utf-8")
    )
    body = client.post(
        "/check", json={"invoice": invoice},
        headers={"X-User-Groups": "school-catering"},
    ).json()

    assert body["decision"] == "flag"
    assert body["requires_human_review"] is True
    text = " ".join(d["explanation"] for d in body["line_decisions"])
    # No rate, clause or supplier detail from the contract they cannot read.
    assert "1 150" not in text and "1150" not in text
    assert "§" not in text


def test_agent_tools_cannot_be_told_which_groups_to_use(retriever):
    """Prompt injection defence.

    Contract text is supplier-supplied and lands in the model's context, so an
    instruction hidden in a PDF is untrusted input the model may well follow.
    The tool signature is what makes that harmless: the model chooses what to
    look up, never who is asking, so injected arguments cannot widen access.
    """
    from app.agent.tools import ToolBox

    outsider = Principal(user_id="bo", groups=["education"])
    box = ToolBox(outsider, retriever=retriever)

    result = box.execute(
        "retrieve_clauses",
        {
            "query": "consultancy rate",
            "contract_id": IT_CONTRACT,
            # A model that had been injection-prompted into trying this:
            "groups": ["it-dept", "procurement"],
            "allowed_groups": ["it-dept"],
            "principal": {"groups": ["it-dept"]},
        },
    )
    assert "No clauses found" in result.content
    assert box.seen == {}


def test_pipeline_decision_is_never_an_auto_approval():
    """The one invariant that holds on every path through the system."""
    from app.core.pipeline import DecisionPipeline

    settings = get_settings()
    pipeline = DecisionPipeline(settings=settings)
    principal = Principal(user_id="anna", groups=["it-dept", "procurement"])

    for name in ("INV-001.json", "INV-008.json"):
        invoice = Invoice.model_validate(
            json.loads((settings.invoices_dir / name).read_text(encoding="utf-8"))
        )
        assert pipeline.run(invoice, principal).requires_human_review is True
