"""pgvector store: the intended production backend.

Why Postgres rather than a dedicated vector database:

* One query does both jobs. The access filter, the contract filter and the
  nearest-neighbour ordering are a single SQL statement, so the database can
  use the GIN index on allowed_groups to shrink the candidate set *before*
  ranking. Bolting a metadata filter onto a separate vector service usually
  means over-fetching then post-filtering.
* Row-level security. Postgres can enforce the access rule itself, underneath
  the application, which is what makes the second defence layer possible.
* Self-hostable, which the sovereignty constraint requires.

Enable with:  pip install -r requirements-pg.txt  and set DATABASE_URL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.ingest.chunk import Chunk
from app.store.base import SearchFilter, StoredChunk

SCHEMA_PATH = Path(__file__).parent / "schema_pg.sql"


class PgVectorStore:
    name = "pgvector"

    def __init__(self, dsn: str | None) -> None:
        if not dsn:
            raise ValueError("DATABASE_URL is required for the pgvector backend")
        try:
            import psycopg
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "pgvector backend needs psycopg. pip install -r requirements-pg.txt"
            ) from exc

        self._psycopg = psycopg
        self._pool = ConnectionPool(dsn, min_size=1, max_size=8, open=True)

    def initialise(self, dim: int) -> None:
        """Apply the schema, but only if nobody has applied it already.

        In a deployment the schema is applied by a privileged bootstrap step --
        `docker-entrypoint-initdb.d`, or a migration job -- because the two
        application roles deliberately hold no DDL rights and cannot run it.
        Attempting it unconditionally would fail on `ALTER TABLE ... ENABLE ROW
        LEVEL SECURITY`, which requires ownership.

        It stays here so that a developer pointing at an empty database still
        gets a working table.
        """
        with self._pool.connection() as conn:
            already = conn.execute(
                "SELECT to_regclass('public.chunks') IS NOT NULL"
            ).fetchone()[0]
            if already:
                return

            sql = SCHEMA_PATH.read_text(encoding="utf-8")
            if dim != 384:
                sql = sql.replace("vector(384)", f"vector({dim})")
            conn.execute(sql)
            conn.commit()

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        rows = [
            (
                c.chunk_id, c.contract_id, c.supplier, c.department, c.clause_id,
                c.heading, c.text, c.has_table, c.allowed_groups, c.source_mode,
                _vector_literal(v),
            )
            for c, v in zip(chunks, vectors)
        ]
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks (chunk_id, contract_id, supplier, department,
                                        clause_id, heading, text, has_table,
                                        allowed_groups, source_mode, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        contract_id=EXCLUDED.contract_id, supplier=EXCLUDED.supplier,
                        department=EXCLUDED.department, clause_id=EXCLUDED.clause_id,
                        heading=EXCLUDED.heading, text=EXCLUDED.text,
                        has_table=EXCLUDED.has_table,
                        allowed_groups=EXCLUDED.allowed_groups,
                        source_mode=EXCLUDED.source_mode, embedding=EXCLUDED.embedding
                    """,
                    rows,
                )
            conn.commit()
        return len(rows)

    def search(self, query_vector: Sequence[float], filters: SearchFilter) -> list[StoredChunk]:
        conditions = ["allowed_groups && %(groups)s"]
        params: dict = {
            "groups": list(filters.allowed_groups),
            "limit": filters.limit,
            "query": _vector_literal(query_vector),
        }
        if filters.contract_id:
            conditions.append("contract_id = %(contract_id)s")
            params["contract_id"] = filters.contract_id

        # The ORDER BY runs over rows that already survived the access and
        # contract predicates -- filtered during retrieval, not after it.
        sql = f"""
            SELECT chunk_id, contract_id, supplier, department, clause_id, heading,
                   text, has_table, allowed_groups, source_mode,
                   1 - (embedding <=> %(query)s::vector) AS score
            FROM chunks
            WHERE {' AND '.join(conditions)}
            ORDER BY embedding <=> %(query)s::vector
            LIMIT %(limit)s
        """
        with self._session(filters) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_to_chunk(r) for r in rows]

    def get_contract_chunks(self, contract_id: str, filters: SearchFilter) -> list[StoredChunk]:
        sql = """
            SELECT chunk_id, contract_id, supplier, department, clause_id, heading,
                   text, has_table, allowed_groups, source_mode, 0.0 AS score
            FROM chunks
            WHERE contract_id = %(contract_id)s AND allowed_groups && %(groups)s
            ORDER BY chunk_id
        """
        params = {"contract_id": contract_id, "groups": list(filters.allowed_groups)}
        with self._session(filters) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_to_chunk(r) for r in rows]

    def contract_exists(self, contract_id: str, filters: SearchFilter) -> bool:
        with self._session(filters) as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE contract_id = %(cid)s "
                "AND allowed_groups && %(groups)s LIMIT 1",
                {"cid": contract_id, "groups": list(filters.allowed_groups)},
            ).fetchone()
        return row is not None

    def count(self) -> int:
        """How many clauses are indexed, for /health.

        Not `COUNT(*)`: the serving role sees no rows without a principal, so a
        plain count would report 0 on a perfectly healthy service. The schema
        exposes a SECURITY DEFINER function instead, which can return the
        number without exposing a single row.
        """
        with self._pool.connection() as conn:
            return int(conn.execute("SELECT chunks_indexed()").fetchone()[0])

    def clear(self) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM chunks")
            conn.commit()

    def _session(self, filters: SearchFilter):
        """Connection with the RLS session variable bound to the caller's groups.

        SET LOCAL scopes the setting to this transaction, so a pooled
        connection cannot carry one user's groups into the next user's query.
        """
        conn = self._pool.connection()
        ctx = conn.__enter__()
        ctx.execute(
            "SELECT set_config('app.user_groups', %s, true)",
            (",".join(filters.allowed_groups),),
        )
        return _SessionWrapper(conn, ctx)


class _SessionWrapper:
    def __init__(self, cm, conn) -> None:
        self._cm, self._conn = cm, conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in vector) + "]"


def _to_chunk(row) -> StoredChunk:
    return StoredChunk(
        chunk_id=row[0], contract_id=row[1], supplier=row[2], department=row[3],
        clause_id=row[4], heading=row[5], text=row[6], has_table=row[7],
        allowed_groups=list(row[8]), source_mode=row[9], score=float(row[10]),
    )
