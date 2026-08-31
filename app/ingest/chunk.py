"""Structure-aware chunking.

The chunk boundary is the clause boundary. Nothing here uses a fixed character
window, and that is a deliberate decision with three consequences:

1. Every chunk has a real clause ID (`C-2024-001-§4.2`), so a citation the
   system shows a caseworker points at something that exists in the contract.
2. A rate table is never split in half, because a table always lives inside one
   clause body. Split tables are the classic way RAG silently loses a rate.
3. Retrieval can be scored against `governing_clause_id` in labels.json, which
   is what lets evaluation separate a retrieval failure from an interpretation
   failure.

The cost is that chunk sizes are uneven, and a very long clause could exceed a
comfortable embedding window. `MAX_CHARS` splits only such oversized clauses,
and the parts keep the same clause ID with a part suffix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingest.parse import ParsedDocument

# "## §4 Compensation" or "### §4.2 Implementation and integration"
HEADING_RE = re.compile(r"^(#{2,4})\s*§\s*(?P<num>\d+(?:\.\d+)*)\.?\s*(?P<title>.*)$")
DOC_TITLE_RE = re.compile(r"^#\s+(?P<title>.+)$")

MAX_CHARS = 2400


@dataclass
class Chunk:
    chunk_id: str
    contract_id: str
    supplier: str
    department: str
    clause_id: str | None
    clause_number: str | None
    heading: str
    text: str
    has_table: bool = False
    allowed_groups: list[str] = field(default_factory=list)
    source_mode: str = "native"

    @property
    def embed_text(self) -> str:
        """Text actually embedded.

        The clause body alone is ambiguous once it leaves its document: "the
        rate shall be SEK 1 150 per hour" does not say whose rate. Prepending a
        contextual header keeps supplier and clause title in the vector, which
        measurably helps when several suppliers have similar clauses.
        """
        header = f"{self.supplier} | contract {self.contract_id} | {self.heading}"
        return f"{header}\n{self.text}"


@dataclass
class _Section:
    level: int
    number: str | None
    title: str
    lines: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return "\n".join(self.lines).strip()


def chunk_document(doc: ParsedDocument, allowed_groups: list[str] | None = None) -> list[Chunk]:
    metadata = doc.metadata
    contract_id = str(metadata.get("contract_id") or doc.path.stem)
    supplier = str(metadata.get("supplier", "")).strip()
    department = str(metadata.get("department", "")).strip()
    groups = list(allowed_groups or [])

    sections = _split_sections(doc.body)

    chunks: list[Chunk] = []
    for section in sections:
        body = section.body
        if not body:
            # A parent heading whose prose lives entirely in its subsections
            # (e.g. "## §3 Scope" followed straight by "### §3.1"). Emitting it
            # would create an empty chunk that can win a similarity match.
            continue

        if section.number is None:
            clause_id = None
            chunk_id = f"{contract_id}-preamble"
            heading = section.title or "Preamble"
        else:
            clause_id = f"{contract_id}-§{section.number}"
            chunk_id = clause_id
            heading = f"§{section.number} {section.title}".strip()

        for part_no, part in enumerate(_split_oversized(body), start=1):
            suffix = "" if part_no == 1 else f"#part{part_no}"
            chunks.append(
                Chunk(
                    chunk_id=f"{chunk_id}{suffix}",
                    contract_id=contract_id,
                    supplier=supplier,
                    department=department,
                    clause_id=clause_id,
                    clause_number=section.number,
                    heading=heading,
                    text=part,
                    has_table="|" in part and "---" in part,
                    allowed_groups=groups,
                    source_mode=doc.source_mode,
                )
            )
    return chunks


def _split_sections(body: str) -> list[_Section]:
    """Cut the document at every §-numbered heading."""
    sections: list[_Section] = []
    current = _Section(level=0, number=None, title="")

    for line in body.splitlines():
        heading = HEADING_RE.match(line.strip())
        if heading:
            sections.append(current)
            current = _Section(
                level=len(heading.group(1)),
                number=heading.group("num"),
                title=heading.group("title").strip(),
            )
            continue

        title = DOC_TITLE_RE.match(line.strip())
        if title and current.number is None and not current.lines:
            current.title = title.group("title").strip()
            continue

        current.lines.append(line)

    sections.append(current)
    return [s for s in sections if s.body or s.number]


def _split_oversized(body: str) -> list[str]:
    """Split only clauses that exceed MAX_CHARS, on paragraph boundaries.

    A table is kept whole even if that makes the part oversized: a half table is
    worse than a long chunk.
    """
    if len(body) <= MAX_CHARS:
        return [body]

    parts: list[str] = []
    buffer: list[str] = []
    size = 0
    for block in body.split("\n\n"):
        block_len = len(block) + 2
        if buffer and size + block_len > MAX_CHARS:
            parts.append("\n\n".join(buffer).strip())
            buffer, size = [], 0
        buffer.append(block)
        size += block_len
    if buffer:
        parts.append("\n\n".join(buffer).strip())
    return [p for p in parts if p]
