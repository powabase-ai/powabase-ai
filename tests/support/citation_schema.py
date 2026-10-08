"""Build ai.message_citations the way a deployed database has it.

The test bootstrap creates tables from the ORM, and message_citations has no
ORM model. Deployments get it from the base schema plus migrations 0009 and
0034, so this replays those two migrations' SQL. Every statement is
idempotent (IF NOT EXISTS), so a reused test database is upgraded in place.
"""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import text

_VERSIONS = Path(__file__).resolve().parents[2] / "migrations" / "versions"


def _upgrade_sql(prefix: str) -> list[str]:
    [path] = _VERSIONS.glob(f"{prefix}_*.py")
    spec = importlib.util.spec_from_file_location(f"mig_{prefix}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured: list[str] = []
    with patch("alembic.op.execute", side_effect=lambda sql, *a, **k: captured.append(str(sql))):
        module.upgrade()
    return captured


def ensure_citation_schema(session) -> None:
    for prefix in ("0009", "0034"):
        for statement in _upgrade_sql(prefix):
            session.execute(text(statement))
    session.commit()
