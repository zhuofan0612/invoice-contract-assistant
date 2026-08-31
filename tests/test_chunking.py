"""Structure-aware chunking.

Chunking is where a contract QA system is usually lost, quietly. A fixed-size
window splits "SEK 1 150 per hour" from the clause heading that says what the
rate is *for*, and the model then reads a plausible number out of the wrong
context. Nothing errors; the answers just get worse, and the cause is three
layers below where the symptom appears.

So the tests here assert three properties the rest of the system depends on:

  * a chunk is a clause, so a retrieved passage is a complete legal statement
  * the clause ID survives, so every citation is one a caseworker can look up
  * tables are not split, so a rate row keeps the header that labels it
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.ingest.chunk import chunk_document
from app.ingest.parse import parse_contract
from app.ingest.pipeline import build_chunks

SETTINGS = get_settings()


@pytest.fixture(scope="module")
def chunks():
    built, _ = build_chunks(SETTINGS)
    return built


def test_every_contract_produces_clauses(chunks):
    contracts = {c.contract_id for c in chunks}
    assert len(contracts) == 8
    assert len(chunks) > 40


def test_clause_ids_are_unique_and_resolvable(chunks):
    """A duplicate ID would make a citation ambiguous, which defeats its purpose."""
    ids = [c.clause_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all("§" in cid for cid in ids)


def test_clause_id_encodes_its_contract(chunks):
    """`C-2024-001-§4.2` is self-describing, so a cited ID needs no lookup table."""
    for chunk in chunks:
        assert chunk.clause_id.startswith(chunk.contract_id + "-§")


def test_no_chunk_is_empty(chunks):
    """Parent headings whose prose lives in subsections must not become empty chunks.

    An empty chunk is retrievable, scores unpredictably, and gives the model
    nothing to read.
    """
    assert all(chunk.text.strip() for chunk in chunks)


def test_a_rate_clause_keeps_its_rate_and_its_heading(chunks):
    """The property fixed-size chunking breaks."""
    clause = next(c for c in chunks if c.clause_id == "C-2024-001-§4.2")
    assert "1 150" in clause.text
    assert "§4.2" in clause.heading
    # And the embedded form carries the context needed to tell near-identical
    # clauses in different contracts apart.
    assert "C-2024-001" in clause.embed_text
    assert "Nordisk Konsult AB" in clause.embed_text


def test_tables_stay_intact_with_their_header_row(chunks):
    """A rate table split across chunks yields rows whose columns are unlabelled."""
    clause = next(c for c in chunks if c.clause_id == "C-2024-003-§4.1")
    assert clause.has_table
    for role, rate in [("Technician", "745"), ("Senior technician", "895"),
                       ("Emergency", "1 340")]:
        assert role in clause.text, f"{role} row was split out of the table"
        assert rate in clause.text
    # The header must be in the same chunk, or 745 is a number with no meaning.
    assert "Rate" in clause.text or "SEK" in clause.text


def test_tiered_quantity_bands_survive_as_one_chunk(chunks):
    clause = next(c for c in chunks if c.clause_id == "C-2024-005-§4.1")
    assert clause.has_table
    for band in ("1-50", "51-200", "201"):
        assert band.replace("-", "") in clause.text.replace("–", "").replace("-", "")


def test_the_scanned_contract_goes_through_the_ocr_path():
    """Marked scanned, so it must be reported as OCR rather than native text.

    The flag is honest bookkeeping: text extracted by OCR is lower confidence
    than text read from a PDF's own layer, and a caseworker reviewing a flag
    that hinges on a mis-OCR'd digit needs to know which they are looking at.
    """
    parsed = parse_contract(SETTINGS.contracts_dir / "C-2024-006.md")
    assert parsed.source_mode == "pdf-ocr"
    assert parsed.warnings


def test_chunking_is_deterministic():
    """Same input, same chunks -- otherwise eval numbers are not comparable."""
    path = SETTINGS.contracts_dir / "C-2024-001.md"
    first = chunk_document(parse_contract(path))
    second = chunk_document(parse_contract(path))
    assert [c.clause_id for c in first] == [c.clause_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_labelled_governing_clauses_all_exist(chunks):
    """Ties the golden set to the chunker.

    If chunking changes such that a labelled clause ID stops existing, the
    eval would silently start scoring against clauses that cannot be
    retrieved. This fails loudly instead.
    """
    import json

    labels = json.loads(SETTINGS.labels_path.read_text(encoding="utf-8"))
    rows = labels["invoices"] if isinstance(labels, dict) else labels
    available = {c.clause_id for c in chunks}
    missing = {
        row["governing_clause_id"] for row in rows
        if row.get("governing_clause_id")
    } - available
    assert not missing, f"labels reference clauses the chunker does not produce: {missing}"
