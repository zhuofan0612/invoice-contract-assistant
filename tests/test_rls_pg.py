"""Row-level security, asserted against a live Postgres.

These skip unless `DATABASE_URL` points at a real database, which is why they
did not exist earlier -- and why a bug survived: the RLS policy was written,
reviewed and described in the README, but nothing ever executed it. Two things
were wrong at once. The policy referenced a role that no file created, so the
schema aborted on first boot; and the application connected as the bootstrap
superuser, which bypasses row security unconditionally. Either fault alone
makes layer 2 inert, and the SQLite suite cannot see either.

The first test below is the one that matters. It does not test a query -- it
tests *who we are*, because every other guarantee here is void for a superuser.

Run them with:

    export DATABASE_URL=postgresql://invoice_app:invoice_app@localhost:5432/invoice_check
    .venv/bin/python -m pytest tests/test_rls_pg.py -v
"""

from __future__ import annotations

import os

import pytest

DSN = os.getenv("DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not DSN.startswith("postgres"), reason="needs DATABASE_URL pointing at Postgres"
)

IT_CONTRACT = "C-2024-001"      # it-dept, procurement
EDU_GROUP = "education"


@pytest.fixture()
def conn():
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(DSN) as connection:
        yield connection


def _rows(conn, groups: str | None) -> list[tuple]:
    """Raw SQL with no application filter at all -- only RLS is in play."""
    with conn.cursor() as cur:
        if groups is not None:
            cur.execute("SELECT set_config('app.user_groups', %s, false)", (groups,))
        cur.execute("SELECT contract_id, allowed_groups FROM chunks")
        return cur.fetchall()


def test_the_serving_role_cannot_bypass_row_security(conn):
    """The bug that made every other test in this file meaningless.

    `FORCE ROW LEVEL SECURITY` removes the table *owner's* exemption. It does
    nothing about superusers or BYPASSRLS, so an application connecting as the
    role `POSTGRES_USER` creates is never subject to any policy, however
    carefully the policy is written.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
        )
        is_superuser, bypasses_rls = cur.fetchone()

    assert not is_superuser, "the serving role is a superuser; RLS is not enforced"
    assert not bypasses_rls, "the serving role has BYPASSRLS; RLS is not enforced"


def test_a_session_with_no_groups_set_sees_nothing(conn):
    """Deny by default, enforced by the database rather than by the application.

    `current_setting('app.user_groups', true)` is NULL when unset, and `&&`
    against NULL is NULL rather than true. A caller who forgets to bind the
    variable gets an empty result, not the whole table.
    """
    assert _rows(conn, None) == []


def test_groups_bound_in_the_session_bound_the_rows(conn):
    """And the filter is the real one, not merely 'returns nothing always'."""
    edu = _rows(conn, EDU_GROUP)
    assert edu, "a user in education should be able to read something"
    assert all(EDU_GROUP in groups for _, groups in edu)
    assert all(contract != IT_CONTRACT for contract, _ in edu)


def test_deleting_the_application_filter_would_still_be_safe(conn):
    """The claim layer 2 exists to support, stated as a test.

    This query is what a catastrophic bug in `app/store/pgvector_store.py`
    would produce: the group predicate gone from the WHERE clause entirely.
    The database still refuses to return the forbidden rows.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('app.user_groups', %s, false)", (EDU_GROUP,))
        cur.execute("SELECT count(*) FROM chunks WHERE contract_id = %s", (IT_CONTRACT,))
        assert cur.fetchone()[0] == 0


def test_health_can_count_rows_it_is_not_allowed_to_read(conn):
    """Aggregate exposure without row exposure.

    /health reports how many clauses are indexed. Under RLS the serving role
    sees no rows without a principal, so `COUNT(*)` reports 0 and a healthy
    service looks broken. A SECURITY DEFINER function returns the number and
    nothing else.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        visible = cur.fetchone()[0]
        cur.execute("SELECT chunks_indexed()")
        indexed = cur.fetchone()[0]

    assert visible == 0, "no principal is bound, so no rows should be visible"
    assert indexed > 0, "but the health count still works"
