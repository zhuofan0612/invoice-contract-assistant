"""SQLite fallback store.

Same interface and the same *filter-during-retrieval* semantics as the pgvector
backend, so the retrieval and access-control logic can be developed and tested
with no infrastructure. The access predicate is a real SQL predicate here too
(via json_each over allowed_groups), not a Python post-filter.

The one thing SQLite genuinely cannot do is index the vectors: cosine
similarity is computed in Python over the rows that survive the filter. At
~50 clauses per contract that is microseconds and completely fine. It does not
scale, which is exactly why pgvector with an ivfflat index is the intended
production backend.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np

from app.ingest.chunk import Chunk
from app.store.base import SearchFilter, StoredChunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       TEXT PRIMARY KEY,
    contract_id    TEXT NOT NULL,
    supplier       TEXT NOT NULL DEFAULT '',
    department     TEXT NOT NULL DEFAULT '',
    clause_id      TEXT,
    heading        TEXT NOT NULL DEFAULT '',
    text           TEXT NOT NULL,
    has_table      INTEGER NOT NULL DEFAULT 0,
    allowed_groups TEXT NOT NULL DEFAULT '[]',
    source_mode    TEXT NOT NULL DEFAULT 'native',
    embedding      BLOB NOT NULL,
    dim            INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_contract ON chunks(contract_id);
"""


class SqliteVectorStore:
    name = "sqlite"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def initialise(self, dim: int) -> None:
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        rows = []
        for chunk, vector in zip(chunks, vectors):
            array = np.asarray(vector, dtype=np.float32)
            rows.append(
                (
                    chunk.chunk_id, chunk.contract_id, chunk.supplier, chunk.department,
                    chunk.clause_id, chunk.heading, chunk.text, int(chunk.has_table),
                    json.dumps(chunk.allowed_groups), chunk.source_mode,
                    array.tobytes(), int(array.size),
                )
            )
        self._conn.executemany(
            """
            INSERT INTO chunks (chunk_id, contract_id, supplier, department, clause_id,
                                heading, text, has_table, allowed_groups, source_mode,
                                embedding, dim)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                contract_id=excluded.contract_id, supplier=excluded.supplier,
                department=excluded.department, clause_id=excluded.clause_id,
                heading=excluded.heading, text=excluded.text,
                has_table=excluded.has_table, allowed_groups=excluded.allowed_groups,
                source_mode=excluded.source_mode, embedding=excluded.embedding,
                dim=excluded.dim
            """,
            rows,
        )
        self._conn.commit()
        return len(rows)

    # -- the access predicate ------------------------------------------------

    def _where(self, filters: SearchFilter) -> tuple[str, list]:
        clauses: list[str] = []
        params: list = []

        if filters.contract_id:
            clauses.append("contract_id = ?")
            params.append(filters.contract_id)

        # Access control, expressed in SQL. A chunk is visible only if one of
        # its allowed_groups is one of the caller's groups. Empty caller groups
        # means no access to anything -- deny by default, never allow by default.
        placeholders = ",".join("?" * len(filters.allowed_groups)) or "NULL"
        clauses.append(
            f"EXISTS (SELECT 1 FROM json_each(chunks.allowed_groups) AS g "
            f"WHERE g.value IN ({placeholders}))"
        )
        params.extend(filters.allowed_groups)

        return (" AND ".join(clauses) if clauses else "1=1"), params

    def search(self, query_vector: Sequence[float], filters: SearchFilter) -> list[StoredChunk]:
        where, params = self._where(filters)
        rows = self._conn.execute(f"SELECT * FROM chunks WHERE {where}", params).fetchall()
        if not rows:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query) or 1.0

        matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1.0
        similarities = (matrix @ query) / (norms * query_norm)

        order = np.argsort(-similarities)[: filters.limit]
        return [_to_chunk(rows[i], float(similarities[i])) for i in order]

    def get_contract_chunks(self, contract_id: str, filters: SearchFilter) -> list[StoredChunk]:
        scoped = SearchFilter(
            contract_id=contract_id,
            allowed_groups=filters.allowed_groups,
            limit=filters.limit,
        )
        where, params = self._where(scoped)
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE {where} ORDER BY chunk_id", params
        ).fetchall()
        return [_to_chunk(r, 0.0) for r in rows]

    def contract_exists(self, contract_id: str, filters: SearchFilter) -> bool:
        scoped = SearchFilter(contract_id=contract_id, allowed_groups=filters.allowed_groups)
        where, params = self._where(scoped)
        row = self._conn.execute(
            f"SELECT 1 FROM chunks WHERE {where} LIMIT 1", params
        ).fetchone()
        return row is not None

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def clear(self) -> None:
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()


def _to_chunk(row: sqlite3.Row, score: float) -> StoredChunk:
    return StoredChunk(
        chunk_id=row["chunk_id"],
        contract_id=row["contract_id"],
        supplier=row["supplier"],
        department=row["department"],
        clause_id=row["clause_id"],
        heading=row["heading"],
        text=row["text"],
        has_table=bool(row["has_table"]),
        allowed_groups=json.loads(row["allowed_groups"]),
        source_mode=row["source_mode"],
        score=score,
    )
