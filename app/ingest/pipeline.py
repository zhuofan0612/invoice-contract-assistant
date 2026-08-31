"""Ingestion pipeline: contracts on disk -> embedded, access-tagged chunks.

Access groups are attached at ingest time, from access.json, and stored on the
chunk row. Retrieval then filters on that column. Doing it here rather than at
query time means there is exactly one place where a chunk's audience is
decided, and it is the same place the chunk is created.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings, get_settings
from app.embeddings.base import Embedder, get_embedder
from app.ingest.chunk import Chunk, chunk_document
from app.ingest.parse import parse_contract
from app.store.base import VectorStore, get_store

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    contracts: int = 0
    chunks: int = 0
    tables: int = 0
    ocr_documents: int = 0
    per_contract: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Ingested {self.contracts} contracts -> {self.chunks} clause chunks "
            f"({self.tables} contain tables, {self.ocr_documents} via OCR path)"
        ]
        for contract_id, n in sorted(self.per_contract.items()):
            lines.append(f"  {contract_id}: {n} chunks")
        lines.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(lines)


def load_access_map(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        log.warning("no access.json at %s; every chunk will be private", path)
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {e["contract_id"]: list(e.get("allowed_groups", [])) for e in entries}


def build_chunks(settings: Settings | None = None) -> tuple[list[Chunk], IngestReport]:
    settings = settings or get_settings()
    access = load_access_map(settings.access_path)
    report = IngestReport()
    chunks: list[Chunk] = []

    paths = sorted(
        p for p in settings.contracts_dir.iterdir()
        if p.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"}
    )
    for path in paths:
        doc = parse_contract(path)
        contract_id = str(doc.metadata.get("contract_id") or path.stem)
        groups = access.get(contract_id)
        if groups is None:
            report.warnings.append(
                f"{contract_id} has no entry in access.json; defaulting to no access"
            )
            groups = []

        doc_chunks = chunk_document(doc, allowed_groups=groups)
        chunks.extend(doc_chunks)

        report.contracts += 1
        report.per_contract[contract_id] = len(doc_chunks)
        report.tables += sum(1 for c in doc_chunks if c.has_table)
        if doc.source_mode == "pdf-ocr":
            report.ocr_documents += 1
        report.warnings.extend(f"{contract_id}: {w}" for w in doc.warnings)

    report.chunks = len(chunks)
    return chunks, report


def ingest(
    settings: Settings | None = None,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    rebuild: bool = True,
) -> IngestReport:
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    store = store or get_store(settings)

    chunks, report = build_chunks(settings)
    if not chunks:
        report.warnings.append("no contracts found")
        return report

    store.initialise(embedder.dim)
    if rebuild:
        store.clear()

    vectors = embedder.encode([c.embed_text for c in chunks])
    store.upsert(chunks, vectors)
    return report
