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
-- rows. tests/test_rls_pg.py asserts exactly that, against a live database.
--
-- **Layer 2 only exists if the application is not a superuser.** Superusers and
-- roles with BYPASSRLS ignore row security entirely; FORCE ROW LEVEL SECURITY
-- removes the *table owner's* exemption but does nothing about that. So this
-- file creates two ordinary login roles and the application connects as one of
-- them, never as the bootstrap owner. That separation is the policy; everything
-- below is detail.

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
-- The two application roles
-- ---------------------------------------------------------------------------
--
-- Neither is a superuser and neither owns the table, so both are subject to the
-- policies below. Passwords are placeholders for local development; in a real
-- deployment these roles are created by the migration job and the credentials
-- come from a secret (see deploy/).
--
--   invoice_app     serves requests. SELECT only, and every SELECT is filtered
--                   by the caller's groups.
--   invoice_ingest  writes the index. Reads and writes every row, because it
--                   builds them, but nothing serves traffic as this role.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'invoice_app') THEN
        CREATE ROLE invoice_app LOGIN PASSWORD 'invoice_app';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'invoice_ingest') THEN
        CREATE ROLE invoice_ingest LOGIN PASSWORD 'invoice_ingest';
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO invoice_app, invoice_ingest;
GRANT SELECT ON chunks TO invoice_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON chunks TO invoice_ingest;


-- ---------------------------------------------------------------------------
-- Row-level security
-- ---------------------------------------------------------------------------

ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
-- FORCE so the table owner is not silently exempt.
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS chunks_group_read ON chunks;
CREATE POLICY chunks_group_read ON chunks
    FOR SELECT
    TO invoice_app
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

-- FOR ALL is safe here only because invoice_ingest is a separate login role
-- that never serves a request. Granting this policy to the serving role -- or
-- letting one role do both jobs -- would OR `USING (true)` into every SELECT
-- and quietly disable the group filter above.


-- ---------------------------------------------------------------------------
-- Aggregate exposure without row exposure
-- ---------------------------------------------------------------------------
--
-- GET /health reports how many clauses are indexed. Under RLS the serving role
-- can see no rows without a principal, so a plain COUNT(*) would report 0 and
-- make a healthy service look broken. SECURITY DEFINER runs this as the owner,
-- which is acceptable because the only thing it can return is a number.

CREATE OR REPLACE FUNCTION chunks_indexed() RETURNS bigint
    LANGUAGE sql
    SECURITY DEFINER
    SET search_path = public
    AS 'SELECT count(*) FROM chunks';

REVOKE ALL ON FUNCTION chunks_indexed() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION chunks_indexed() TO invoice_app, invoice_ingest;
