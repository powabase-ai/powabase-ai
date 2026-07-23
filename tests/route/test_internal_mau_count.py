"""Tests for GET /api/internal/mau-count (B4 / C1).

An hourly MAU-drain job calls this per-project endpoint with the project's
service_role_key to read the trailing-30d MAU / SSO-MAU / third-party-MAU
counts, which it bills against the compute-tier bundle.

These tests exercise:
  - service-role-only auth (a non-service-role JWT -> 403)
  - regular_mau counts only users active in the trailing 30 days
  - sso_mau counts only active is_sso_user=true users
  - third_party_mau counts only active non-SSO users with a non-email/phone
    OAuth identity
  - the empty case returns all zeros

The PS test DB has no `auth` schema (it is created by GoTrue in real projects),
so the module-scoped `auth_schema` fixture below creates a MINIMAL auth schema
(only the columns the queries read) and cleans its rows between tests.
"""

import uuid

import pytest
from sqlalchemy import text

from agentic_project_service.db import db


# ---------------------------------------------------------------------------
# Minimal auth-schema harness (see module docstring)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def auth_schema(app):
    """Create a minimal auth.users / auth.identities for the MAU queries.

    Real GoTrue tables have many more columns; we create only the columns the
    endpoint's three queries read: users(id, last_sign_in_at, is_sso_user) and
    identities(id, user_id, provider).
    """
    with app.app_context():
        db.session.execute(text("CREATE SCHEMA IF NOT EXISTS auth"))
        db.session.execute(
            text(
                "CREATE TABLE IF NOT EXISTS auth.users ("
                "  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),"
                "  last_sign_in_at timestamptz,"
                "  is_sso_user boolean NOT NULL DEFAULT false"
                ")"
            )
        )
        db.session.execute(
            text(
                "CREATE TABLE IF NOT EXISTS auth.identities ("
                "  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),"
                "  user_id uuid NOT NULL,"
                "  provider text NOT NULL"
                ")"
            )
        )
        db.session.commit()
    yield


@pytest.fixture(autouse=True)
def _clean_auth_rows(app, auth_schema):
    """Truncate auth.* rows between tests so per-test seeds don't bleed."""
    yield
    with app.app_context():
        db.session.execute(text("TRUNCATE auth.identities, auth.users CASCADE"))
        db.session.commit()


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _seed_user(app, *, days_ago: float | None, is_sso_user: bool = False) -> str:
    """Insert an auth.users row. `days_ago=None` means last_sign_in_at IS NULL
    (never signed in). Returns the user id."""
    user_id = str(uuid.uuid4())
    if days_ago is None:
        last_sign_in = None
    else:
        last_sign_in = text(f"now() - INTERVAL '{days_ago} days'")
    with app.app_context():
        if last_sign_in is None:
            db.session.execute(
                text(
                    "INSERT INTO auth.users (id, last_sign_in_at, is_sso_user) "
                    "VALUES (:id, NULL, :sso)"
                ),
                {"id": user_id, "sso": is_sso_user},
            )
        else:
            db.session.execute(
                text(
                    "INSERT INTO auth.users (id, last_sign_in_at, is_sso_user) "
                    f"VALUES (:id, now() - INTERVAL '{days_ago} days', :sso)"
                ),
                {"id": user_id, "sso": is_sso_user},
            )
        db.session.commit()
    return user_id


def _seed_identity(app, *, user_id: str, provider: str) -> None:
    with app.app_context():
        db.session.execute(
            text("INSERT INTO auth.identities (user_id, provider) VALUES (:uid, :prov)"),
            {"uid": user_id, "prov": provider},
        )
        db.session.commit()


def _service_role(mocker):
    """Patch decode_jwt to authenticate the request as service-role."""
    mocker.patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={
            "sub": "service",
            "role": "service_role",
            "is_service_role": True,
        },
    )


_SR_HEADERS = {"Authorization": "Bearer fake-service-role-key"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mau_count_requires_service_role(client, app, mocker):
    """A non-service-role (regular user) JWT is rejected with 403."""
    mocker.patch(
        "agentic_project_service.auth.decode_jwt",
        return_value={"sub": str(uuid.uuid4()), "role": "authenticated"},
    )
    resp = client.get("/api/internal/mau-count", headers=_SR_HEADERS)
    assert resp.status_code == 403


def test_mau_count_counts_active_users(client, app, mocker):
    """regular_mau counts only users active within the trailing 30 days."""
    _seed_user(app, days_ago=1)  # active
    _seed_user(app, days_ago=29)  # active (just inside)
    _seed_user(app, days_ago=45)  # inactive (outside window)
    _seed_user(app, days_ago=None)  # never signed in

    _service_role(mocker)
    resp = client.get("/api/internal/mau-count", headers=_SR_HEADERS)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body == {"regular_mau": 2, "sso_mau": 0, "third_party_mau": 0}


def test_mau_count_sso(client, app, mocker):
    """sso_mau counts only active is_sso_user=true users."""
    _seed_user(app, days_ago=2, is_sso_user=True)  # active SSO -> counts
    _seed_user(app, days_ago=5, is_sso_user=True)  # active SSO -> counts
    _seed_user(app, days_ago=3, is_sso_user=False)  # active non-SSO -> not SSO
    _seed_user(app, days_ago=60, is_sso_user=True)  # inactive SSO -> excluded

    _service_role(mocker)
    resp = client.get("/api/internal/mau-count", headers=_SR_HEADERS)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["sso_mau"] == 2
    # regular_mau counts ALL active users regardless of sso flag (2 sso + 1 non-sso)
    assert body["regular_mau"] == 3


def test_mau_count_third_party(client, app, mocker):
    """third_party_mau counts only active non-SSO users with a non-email/phone
    OAuth identity. Excludes: email-only users, and SSO users (even with an
    oauth identity)."""
    # active non-SSO user with a google identity -> COUNTS
    u_google = _seed_user(app, days_ago=4, is_sso_user=False)
    _seed_identity(app, user_id=u_google, provider="google")

    # active non-SSO user with only an email identity -> excluded
    u_email = _seed_user(app, days_ago=6, is_sso_user=False)
    _seed_identity(app, user_id=u_email, provider="email")

    # active SSO user with an oauth identity -> excluded (is_sso_user=true)
    u_sso = _seed_user(app, days_ago=7, is_sso_user=True)
    _seed_identity(app, user_id=u_sso, provider="google")

    # inactive non-SSO user with a github identity -> excluded (outside window)
    u_old = _seed_user(app, days_ago=40, is_sso_user=False)
    _seed_identity(app, user_id=u_old, provider="github")

    _service_role(mocker)
    resp = client.get("/api/internal/mau-count", headers=_SR_HEADERS)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["third_party_mau"] == 1


def test_mau_count_third_party_distinct_per_user(client, app, mocker):
    """A user with two third-party identities counts once (count DISTINCT)."""
    u = _seed_user(app, days_ago=3, is_sso_user=False)
    _seed_identity(app, user_id=u, provider="google")
    _seed_identity(app, user_id=u, provider="github")

    _service_role(mocker)
    resp = client.get("/api/internal/mau-count", headers=_SR_HEADERS)
    assert resp.status_code == 200
    assert resp.get_json()["third_party_mau"] == 1


def test_mau_count_empty(client, app, mocker):
    """No active users -> all zeros."""
    # one inactive user only, to prove zeros aren't an artifact of an empty table
    _seed_user(app, days_ago=90)

    _service_role(mocker)
    resp = client.get("/api/internal/mau-count", headers=_SR_HEADERS)
    assert resp.status_code == 200
    assert resp.get_json() == {"regular_mau": 0, "sso_mau": 0, "third_party_mau": 0}
