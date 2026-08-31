"""The HTTP surface.

Thin routes still need tests, because "thin" is where nothing catches a typo.
The `/ingest` route shipped referencing a field the report did not have -- a
guaranteed 500 that no other test touched, because every other test calls the
pipeline in-process. These exist so that every route is executed at least once.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

SETTINGS = get_settings()
AUTHORISED = {"X-User-Id": "anna", "X-User-Groups": "procurement,it-dept"}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def invoice() -> dict:
    return json.loads((SETTINGS.invoices_dir / "INV-008.json").read_text(encoding="utf-8"))


def test_health_reports_the_backends_that_actually_resolved(client):
    """The service starts happily on fallbacks, so this is how you tell which ran."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["clauses_indexed"] > 0
    for key in ("store_backend", "embedding_backend", "reranker_backend", "llm_backend"):
        assert key in body["backends"]


def test_ingest_route_runs_and_reports_what_it_indexed(client):
    body = client.post("/ingest").json()
    assert body["contracts"] == 8
    assert body["chunks"] > 40
    assert body["tables"] == 2
    assert body["ocr_documents"] == 1


def test_check_returns_a_reviewable_decision(client, invoice):
    body = client.post("/check", json={"invoice": invoice}, headers=AUTHORISED).json()

    assert body["decision"] == "flag"
    assert body["findings"] == ["wrong_rate"]
    assert body["requires_human_review"] is True
    assert body["total_delta"] == 16000.0
    # Every flag must be checkable: a citation and a sentence explaining it.
    line = body["line_decisions"][0]
    assert line["cited_clause_id"] == "C-2024-001-§4.2"
    assert line["explanation"]
    assert body["trace_id"]


def test_agent_mode_is_reachable_over_http(client, invoice):
    body = client.post(
        "/check", json={"invoice": invoice, "mode": "agent"}, headers=AUTHORISED
    ).json()
    assert body["backend"]["mode"] == "agent"
    assert body["requires_human_review"] is True


def test_an_uncheckable_invoice_is_a_finding_not_an_error(client, invoice):
    """The boundary between 422 and a finding, which is worth being explicit about.

    *Malformed* (422): the payload is not a well-formed invoice document --
    missing fields, negative amounts, no line items. Pydantic rejects it and
    the client has a bug to fix.

    *Well-formed but uncheckable* (200 + `invalid_invoice`): it parses, but
    something makes it impossible to compare against a contract -- here a line
    with no description, which cannot be matched to any clause. That is not a
    client error, it is a fact about the invoice, and a caseworker needs to see
    it with the reason attached. Returning 4xx would push the explanation into
    an error handler and out of the decision record, where it is neither traced
    nor reviewable.
    """
    broken = {
        **invoice,
        "line_items": [{**invoice["line_items"][0], "description": "   "}],
    }
    response = client.post("/check", json={"invoice": broken}, headers=AUTHORISED)
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "flag"
    assert body["findings"] == ["invalid_invoice"]
    assert body["requires_human_review"] is True
    assert "description is empty" in body["line_decisions"][0]["explanation"]


@pytest.mark.parametrize(
    "mutation",
    [
        {"line_items": []},                                  # no lines at all
        {"line_items": [{"line_no": 1, "description": "x",   # negative money
                         "quantity": 1.0, "unit_price": -5.0, "line_total": -5.0}]},
        {"supplier": None},                                  # wrong type
    ],
)
def test_a_malformed_invoice_document_is_a_422(client, invoice, mutation):
    """Schema violations are the client's problem; invoice *content* is not."""
    response = client.post("/check", json={"invoice": {**invoice, **mutation}},
                           headers=AUTHORISED)
    assert response.status_code == 422


def test_clauses_route_lets_a_caseworker_follow_a_citation(client):
    """The path from 'cites §4.2' to reading §4.2, which is part of the product."""
    body = client.get("/contracts/C-2024-001/clauses", headers=AUTHORISED).json()
    ids = [c["clause_id"] for c in body["clauses"]]
    assert "C-2024-001-§4.2" in ids
    clause = next(c for c in body["clauses"] if c["clause_id"] == "C-2024-001-§4.2")
    assert "1 150" in clause["text"]


def test_metrics_are_prometheus_formatted(client, invoice):
    client.post("/check", json={"invoice": invoice}, headers=AUTHORISED)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "invoice_checks_total" in response.text
    assert "decisions_total" in response.text
