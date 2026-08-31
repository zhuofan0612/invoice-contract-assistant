-- pgvector schema + row-level security.
--
-- Two independent layers protect the same data:
--
--   Layer 1 (application): app/store/*.py put the caller's groups into the
--   WHERE clause of every search.
--   Layer 2 (database): the RLS policy below re-applies the same rule inside
--   Postgres, against a session variable the application cannot forge on
--   another user's behalf.
--
-- Layer 2 exists because layer 1 is code, and code has bugs. Delete the
-- application filter entirely and this policy still returns zero forbidden
-- rows. tests/test_access.py asserts exactly that.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       text PRIMARY KEY,
    contract_id    text NOT NULL,
    supplier       text NOT NULL DEFAULT '',
    department     text NOT NULL DEFAULT '',
    clause_id      text,
    heading        text NOT NULL DEFAULT '',
    text           text NOT NULL,
    has_table      boolean NOT NULL DEFAULT false,
    allowed_groups text[] NOT NULL DEFAULT '{}',
    source_mode    text NOT NULL DEFAULT 'native',
    embedding      vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_contract ON chunks (contract_id);

-- GIN index so the array-overlap access predicate stays cheap.
CREATE INDEX IF NOT EXISTS idx_chunks_groups ON chunks USING gin (allowed_groups);

-- Approximate nearest-neighbour index. Cosine distance to match the
-- normalised embeddings the application produces.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);


-- ---------------------------------------------------------------------------
-- Row-level security
-- ---------------------------------------------------------------------------

ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
-- FORCE so the table owner is not silently exempt.
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS chunks_group_read ON chunks;
CREATE POLICY chunks_group_read ON chunks
    FOR SELECT
    USING (
        allowed_groups && string_to_array(
            current_setting('app.user_groups', true), ','
        )
    );

-- Deny-by-default note: if app.user_groups is unset, current_setting(..., true)
-- returns NULL, string_to_array(NULL, ',') is NULL, and `&&` against NULL is
-- NULL rather than true. A caller who forgets to set the variable therefore
-- sees nothing at all, which is the failure mode we want.

DROP POLICY IF EXISTS chunks_ingest_write ON chunks;
CREATE POLICY chunks_ingest_write ON chunks
    FOR ALL
    TO invoice_ingest
    USING (true) WITH CHECK (true);
