"""Vector store interface.

`SearchFilter` is the security-relevant type in this project. It is passed into
the store and becomes part of the SQL predicate, so a clause the caller may not
see is never scored, never ranked, and never reaches the LLM prompt. Filtering
after retrieval would still leak: the forbidden text would have been loaded into
the process, and any bug in the post-filter (or a `top_k` cut that happens to
keep it) exposes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from app.ingest.chunk import Chunk


@dataclass(frozen=True)
class SearchFilter:
    """Mandatory predicate applied during candidate search."""

    contract_id: str | None = None
    allowed_groups: tuple[str, ...] = ()
    limit: int = 20

    @classmethod
    def for_principal(
        cls, groups: Sequence[str], contract_id: str | None = None, limit: int = 20
    ) -> "SearchFilter":
        return cls(contract_id=contract_id, allowed_groups=tuple(groups), limit=limit)


@dataclass
class StoredChunk:
    chunk_id: str
    contract_id: str
    supplier: str
    department: str
    clause_id: str | None
    heading: str
    text: str
    has_table: bool
    allowed_groups: list[str] = field(default_factory=list)
    source_mode: str = "native"
    score: float = 0.0


@runtime_checkable
class VectorStore(Protocol):
    name: str

    def initialise(self, dim: int) -> None: ...

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int: ...

    def search(
        self, query_vector: Sequence[float], filters: SearchFilter
    ) -> list[StoredChunk]: ...

    def get_contract_chunks(
        self, contract_id: str, filters: SearchFilter
    ) -> list[StoredChunk]: ...

    def contract_exists(self, contract_id: str, filters: SearchFilter) -> bool: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...


def get_store(settings=None) -> VectorStore:
    from app.config import get_settings

    settings = settings or get_settings()
    if settings.store_backend == "pgvector":
        from app.store.pgvector_store import PgVectorStore

        return PgVectorStore(settings.database_url)

    from app.store.sqlite_store import SqliteVectorStore

    return SqliteVectorStore(settings.sqlite_path)
