"""ai schema is Class-B (private/backend) — C2.2.

anon/authenticated must be denied on every ai table (schema USAGE + table
GRANTs revoked, all 93 authenticated policies dropped); service_role (the
project-service / PostgREST-as-admin path) must be unaffected. See
docs/database-roles-and-scopes.md §7 for the full decision.

Unlike the rest of this suite, these tests do NOT use the `app` fixture
from conftest.py — that fixture builds tables via `db.metadata.create_all()`,
which never applies GRANTs, RLS, or policies (those are Alembic-migration-only
side effects, see migrations/versions/0025_lock_ai_schema_class_b.py). This
file instead opens a raw connection to whatever `DATABASE_URL` points at and
requires migrations to already be at head (``flask db upgrade``) — e.g. the
isolated verification DB used for C2.2:

    DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:55432/oss_plan_b_test \\
        .venv/bin/python -m pytest tests/test_ai_classb.py -v

`anon` / `authenticated` / `service_role` are NOLOGIN group roles in the real
stack (PostgREST connects as `authenticator` and `SET ROLE`s into them per
docs/database-roles-and-scopes.md §2) — mirrored here by connecting as
whatever login role `DATABASE_URL` specifies and `SET ROLE`ing within the
session, exactly like PostgREST does. That makes this a direct test of the
real enforcement mechanism (Postgres GRANT + RLS), not a proxy for it.
"""

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

# A spread of ai tables: ordinary RLS-enabled tables plus the two that
# predate the blanket RLS-enable pass (docs/database-roles-and-scopes.md §7)
# and so need their own explicit ENABLE ROW LEVEL SECURITY.
_SAMPLE_TABLES = [
    "ai.sources",
    "ai.agents",
    "ai.message_citations",
    "ai.ai_provider_keys",
]


@pytest.fixture(scope="module")
def engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip(
            "DATABASE_URL not set — point it at a migrated ai-schema DB to "
            "run this test, e.g. the isolated C2.2 verification container "
            "(postgresql+psycopg://postgres:postgres@localhost:55432/oss_plan_b_test)"
        )
    eng = create_engine(url, future=True)
    with eng.connect() as conn:
        roles = {
            row[0]
            for row in conn.execute(
                text(
                    "SELECT rolname FROM pg_roles "
                    "WHERE rolname IN ('anon', 'authenticated', 'service_role')"
                )
            )
        }
    missing = {"anon", "authenticated", "service_role"} - roles
    if missing:
        pytest.skip(
            f"target DB is missing role(s) {sorted(missing)} — not a "
            f"Supabase-shaped stack, cannot exercise PostgREST role behavior"
        )
    try:
        yield eng
    finally:
        eng.dispose()


def _select_as(engine, role, table):
    """SET ROLE <role>; SELECT ... LIMIT 0 — a real query, not a catalog probe.

    Mirrors packages/agentic-control-plane/tests/integration/
    test_billing_service_role_fixture.py's pattern: exercise the actual
    grant/RLS set rather than introspecting pg_catalog (which would pass
    regardless of the connected role).
    """
    with engine.connect() as conn:
        conn.execute(text(f"SET ROLE {role}"))
        conn.execute(text(f"SELECT * FROM {table} LIMIT 0"))


@pytest.mark.parametrize("table", _SAMPLE_TABLES)
@pytest.mark.parametrize("role", ["anon", "authenticated"])
def test_anon_and_authenticated_denied_on_ai_tables(engine, role, table):
    """anon/authenticated get permission-denied on every ai table (Class-B).

    Covers both an ordinary RLS-enabled table (ai.sources, ai.agents) and
    the two tables that predate the blanket RLS-enable pass
    (ai.message_citations, ai.ai_provider_keys) — those two are NOT covered
    by policy-dropping alone (they never had anon/authenticated policies to
    drop) and need their own ENABLE ROW LEVEL SECURITY.
    """
    with pytest.raises(ProgrammingError) as exc_info:
        _select_as(engine, role, table)
    sqlstate = getattr(exc_info.value.orig, "sqlstate", None) or getattr(
        exc_info.value.orig, "pgcode", None
    )
    assert sqlstate == "42501", (
        f"expected insufficient_privilege (42501) for {role} on {table}, "
        f"got SQLSTATE {sqlstate!r}: {exc_info.value}"
    )


@pytest.mark.parametrize("table", _SAMPLE_TABLES)
def test_service_role_still_has_full_access(engine, table):
    """service_role (project-service's path) is unaffected by the lockdown.

    service_role BYPASSRLS and keeps its table GRANTs — this is the path
    project-service's own SQLAlchemy connection and PostgREST-as-service_role
    (the platform's `/platform/rest` CP proxy) both use.
    """
    _select_as(engine, "service_role", table)
