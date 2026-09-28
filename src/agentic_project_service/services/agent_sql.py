"""Run agent database tools as the caller, never as the service's own login.

The service connects as a superuser, so SQL run on its connection ignores
every privilege and RLS policy. Agent tools therefore use their own logins,
which are neither superusers nor able to bypass RLS:

* ``powabase_agent_user`` — for runs an end user started. Each transaction
  becomes ``authenticated`` with that user's JWT claims, so the agent reads
  and writes exactly what the user could through the REST API with their own
  token: the project's grants plus its RLS policies.
* ``powabase_agent_backend`` — for runs started with the service role key.
  Each transaction becomes the agent's own role, which holds grants on exactly
  the tables configured on its database tools (and bypasses RLS on them, as
  the service role would).

The caller's identity lives in session settings that SQL can overwrite in the
same statement (``set_config('request.jwt.claims', ...)``,
``set_config('role', ...)``), so free-form SQL from ``database_query`` is also
parsed before it runs: see :func:`parse_select`.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from pglast import ast, parse_sql
from pglast.parser import ParseError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

from ..db import db
from .tool_caller import ToolCaller

logger = logging.getLogger(__name__)

USER_LOGIN = "powabase_agent_user"
BACKEND_LOGIN = "powabase_agent_backend"
_AGENT_ROLE_PREFIX = "powabase_agent_"

# Never granted to an agent role, whatever its tool configuration names.
_PROTECTED_SCHEMAS = frozenset(
    {
        "ai",
        "auth",
        "storage",
        "extensions",
        "graphql",
        "graphql_public",
        "realtime",
        "supabase_functions",
        "supabase_migrations",
        "vault",
        "net",
        "cron",
        "pgbouncer",
        "pgsodium",
        "pgsodium_masks",
        "information_schema",
    }
)


class AgentSqlRejected(ValueError):
    """The SQL is not something an agent's query tool may run."""


def agent_role_name(agent_id: str) -> str:
    """The Postgres role a service-role run of this agent's tools assumes."""
    return f"{_AGENT_ROLE_PREFIX}{uuid.UUID(str(agent_id)).hex}"


def _is_protected_schema(schema: str) -> bool:
    return schema in _PROTECTED_SCHEMAS or schema.startswith("pg_")


# Built-in functions an agent's SQL may not call, even though they live in
# pg_catalog: they change session settings (the caller's identity lives
# there), run a SQL string the parse gate cannot see into, read other
# sessions or the server's files, or hold the connection.
_DENIED_FUNCTIONS = frozenset(
    {
        "set_config",
        "query_to_xml",
        "query_to_xmlschema",
        "query_to_xml_and_xmlschema",
        "cursor_to_xml",
        "cursor_to_xmlschema",
        "table_to_xml",
        "table_to_xmlschema",
        "table_to_xml_and_xmlschema",
        "schema_to_xml",
        "schema_to_xmlschema",
        "schema_to_xml_and_xmlschema",
        "database_to_xml",
        "database_to_xmlschema",
        "database_to_xml_and_xmlschema",
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "pg_notify",
        "dblink",
    }
)
_DENIED_FUNCTION_PREFIXES = (
    "pg_advisory",
    "pg_try_advisory",
    "pg_stat_",
    "pg_read_",
    "pg_ls_",
    "pg_file",
    "lo_",
    "pg_terminate",
    "pg_cancel",
    "pg_reload",
    "pg_rotate",
    "pg_logical",
    "pg_replication",
    "pg_create",
    "pg_drop",
    "pg_switch",
    "pg_backup",
    "pg_promote",
    "pg_import",
)

# Statements that write, and clauses that write or lock, wherever they appear
# (a data-modifying CTE hides a DELETE inside a SELECT).
_FORBIDDEN_NODES = (
    ast.InsertStmt,
    ast.UpdateStmt,
    ast.DeleteStmt,
    ast.MergeStmt,
    ast.IntoClause,
    ast.LockingClause,
)


@dataclass
class ParsedSelect:
    """What a query reads: tables as (schema or None, name), function names."""

    relations: list[tuple[str | None, str]] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)


def _function_denied(name: str) -> bool:
    return name in _DENIED_FUNCTIONS or name.startswith(_DENIED_FUNCTION_PREFIXES)


def parse_select(sql: str) -> ParsedSelect:
    """Parse ``sql`` and return what it reads, or raise AgentSqlRejected.

    Accepts exactly one SELECT that writes and locks nothing. Every function
    must be unqualified or ``pg_catalog``-qualified and not denied; whether an
    unqualified name really is a built-in is checked against the database by
    the caller. Tables are reported, not judged: the allowlist check happens
    in the database, where names resolve exactly as the query's will.
    """
    try:
        statements = parse_sql(sql)
    except ParseError as e:
        raise AgentSqlRejected(f"Could not parse the query: {e}") from None
    if len(statements) != 1 or not isinstance(statements[0].stmt, ast.SelectStmt):
        raise AgentSqlRejected("Only a single SELECT statement is allowed")

    parsed = ParsedSelect()
    _walk(statements[0].stmt, (), parsed)
    return parsed


def _walk(node, visible_ctes: tuple[frozenset[str], ...], parsed: ParsedSelect) -> None:
    """Visit every node, tracking which CTE names are in scope.

    A CTE name hides a table only where Postgres would resolve it to the CTE:
    inside the statement that defines it, and — unless the WITH is RECURSIVE —
    not inside its own body or the bodies of the CTEs listed before it.
    """
    if isinstance(node, (list, tuple)):
        for item in node:
            _walk(item, visible_ctes, parsed)
        return
    if not isinstance(node, ast.Node):
        return

    if isinstance(node, _FORBIDDEN_NODES):
        raise AgentSqlRejected("The query may not write, lock, or create anything")

    if isinstance(node, ast.RangeVar):
        name = node.relname
        if node.schemaname is None and any(name in scope for scope in visible_ctes):
            return
        parsed.relations.append((node.schemaname, name))
        return

    if isinstance(node, ast.FuncCall):
        parts = [part.sval for part in node.funcname]
        name = parts[-1].lower()
        if len(parts) > 1 and parts[0] != "pg_catalog":
            raise AgentSqlRejected(
                f"Function {'.'.join(parts)} is not allowed: only built-in functions may be called"
            )
        if _function_denied(name):
            raise AgentSqlRejected(f"Function {name} is not allowed")
        parsed.functions.append(name)
        # Arguments, filters and window specs can hold subqueries too.

    if isinstance(node, ast.SelectStmt) and node.withClause is not None:
        with_clause = node.withClause
        names = [cte.ctename for cte in with_clause.ctes]
        for index, cte in enumerate(with_clause.ctes):
            in_scope = names if with_clause.recursive else names[:index]
            _walk(cte.ctequery, visible_ctes + (frozenset(in_scope),), parsed)
        inner = visible_ctes + (frozenset(names),)
        for attr in node:
            if attr != "withClause":
                _walk(getattr(node, attr), inner, parsed)
        return

    for attr in node:
        _walk(getattr(node, attr), visible_ctes, parsed)


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


def ensure_login_roles() -> None:
    """Create or refresh the two tool logins. Run at boot, as the service's login.

    Their password is the service's own, as every other service in a project
    stack logs in with its own role and the shared database password; setting
    it on every boot follows a rotation.
    """
    password = db.engine.url.password
    with db.engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext('powabase_agent_logins'))"))
        for login in (USER_LOGIN, BACKEND_LOGIN):
            if conn.execute(
                text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": login}
            ).first():
                conn.execute(text(f'ALTER ROLE "{login}" {_LOGIN_ATTRIBUTES}'))
            else:
                conn.execute(text(f'CREATE ROLE "{login}" {_LOGIN_ATTRIBUTES}'))
            if password is not None:
                statement = conn.execute(
                    text(
                        "SELECT format('ALTER ROLE %I PASSWORD %L', CAST(:r AS text), CAST(:p AS text))"
                    ),
                    {"r": login, "p": password},
                ).scalar_one()
                conn.execute(text(statement))
        if conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = 'authenticated'")).first():
            conn.execute(text(f'GRANT authenticated TO "{USER_LOGIN}"'))
        else:
            logger.warning(
                "No 'authenticated' role in this database; agent tools cannot run for end users"
            )


_LOGIN_ATTRIBUTES = "LOGIN NOSUPERUSER NOBYPASSRLS NOINHERIT NOCREATEDB NOCREATEROLE NOREPLICATION"


def sync_agent_role(
    agent_id: str,
    read_tables: dict[str, list[str]],
    write_tables: dict[str, list[str]],
) -> None:
    """Make the agent's role hold exactly the grants its database tools configure.

    ``read_tables`` get SELECT; ``write_tables`` get SELECT, INSERT, UPDATE and
    DELETE plus the sequences their columns draw from. Tables that do not
    exist yet and tables in protected schemas are skipped. Grants no longer
    configured are revoked. Cheap when nothing changed: two catalog reads.
    """
    role = agent_role_name(agent_id)
    desired: dict[tuple[str, str], set[str]] = {}
    for tables, privileges in (
        (read_tables, {"SELECT"}),
        (write_tables, {"SELECT", "INSERT", "UPDATE", "DELETE"}),
    ):
        for schema, names in (tables or {}).items():
            if _is_protected_schema(schema):
                continue
            for name in names or []:
                desired.setdefault((schema, name), set()).update(privileges)

    with db.engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(CAST(:r AS text)))"), {"r": role})
        if not conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": role}).first():
            conn.execute(text(f'CREATE ROLE "{role}" NOLOGIN NOINHERIT BYPASSRLS'))
        conn.execute(text(f'GRANT "{role}" TO "{BACKEND_LOGIN}"'))

        existing = {
            (row.schema, row.table): set(row.privileges)
            for row in conn.execute(
                text(
                    """
                    SELECT n.nspname AS schema, c.relname AS table,
                           array_agg(a.privilege_type) AS privileges
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    CROSS JOIN LATERAL aclexplode(c.relacl) a
                    WHERE a.grantee = (SELECT oid FROM pg_roles WHERE rolname = :r)
                      AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
                    GROUP BY 1, 2
                    """
                ),
                {"r": role},
            )
        }

        for (schema, name), privileges in existing.items():
            extra = privileges - desired.get((schema, name), set())
            if extra:
                conn.execute(
                    text(
                        f'REVOKE {", ".join(sorted(extra))} ON TABLE "{schema}"."{name}" FROM "{role}"'
                    )
                )

        for (schema, name), privileges in desired.items():
            if not conn.execute(
                text("SELECT to_regclass(format('%I.%I', CAST(:s AS text), CAST(:t AS text)))"),
                {"s": schema, "t": name},
            ).scalar():
                continue
            missing = privileges - existing.get((schema, name), set())
            if missing:
                conn.execute(text(f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"'))
                conn.execute(
                    text(
                        f'GRANT {", ".join(sorted(missing))} ON TABLE "{schema}"."{name}" TO "{role}"'
                    )
                )
            if "INSERT" in privileges:
                for (sequence,) in conn.execute(
                    text(
                        """
                        SELECT s.oid::regclass::text
                        FROM pg_depend d
                        JOIN pg_class s ON s.oid = d.objid AND s.relkind = 'S'
                        WHERE d.refobjid = to_regclass(format('%I.%I', CAST(:s AS text), CAST(:t AS text)))
                          AND d.deptype IN ('a', 'i')
                        """
                    ),
                    {"s": schema, "t": name},
                ):
                    conn.execute(text(f'GRANT USAGE ON SEQUENCE {sequence} TO "{role}"'))


def drop_agent_role(agent_id: str) -> None:
    """Remove the agent's role and every grant it holds."""
    role = agent_role_name(agent_id)
    with db.engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(CAST(:r AS text)))"), {"r": role})
        if conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": role}).first():
            conn.execute(text(f'DROP OWNED BY "{role}"'))
            conn.execute(text(f'DROP ROLE "{role}"'))


# ---------------------------------------------------------------------------
# Running tool SQL
# ---------------------------------------------------------------------------

_engines: dict[str, Engine] = {}
_engines_lock = threading.Lock()


def _engine(login: str) -> Engine:
    """A small pool logged in as ``login``, with the service's host and password."""
    with _engines_lock:
        engine = _engines.get(login)
        if engine is None:
            engine = create_engine(
                db.engine.url.set(username=login),
                pool_size=1,
                max_overflow=4,
                pool_pre_ping=True,
                pool_recycle=1800,
            )
            _engines[login] = engine
        return engine


_CLAIM_KEY = re.compile(r"^[a-z_][a-z0-9_]*$")


def _set_claims(conn: Connection, claims: dict) -> None:
    """Expose the caller's JWT claims to RLS, both ways ``auth.uid()`` reads them.

    Newer Supabase databases read the ``request.jwt.claims`` JSON; older ones
    read one ``request.jwt.claim.<name>`` setting per claim, as PostgREST used
    to set them. Both are set so policies work on either.
    """
    conn.execute(
        text("SELECT set_config('request.jwt.claims', :claims, true)"),
        {"claims": json.dumps(claims)},
    )
    for key, value in claims.items():
        if _CLAIM_KEY.match(key) and isinstance(value, (str, int, float, bool)):
            conn.execute(
                text("SELECT set_config(:name, :value, true)"),
                {"name": f"request.jwt.claim.{key}", "value": str(value)},
            )


@contextmanager
def agent_transaction(
    caller: ToolCaller, agent_id: str, schemas: list[str], *, read_only: bool
) -> Iterator[Connection]:
    """A transaction on a tool login, acting as ``caller``. Commits on success.

    An end user's transaction is ``authenticated`` with their JWT claims; a
    service-role transaction is the agent's own role (see
    :func:`sync_agent_role`). Either way the login is not a superuser and
    cannot bypass RLS itself.
    """
    login = USER_LOGIN if caller.is_end_user else BACKEND_LOGIN
    with _engine(login).connect() as conn, conn.begin():
        if read_only:
            conn.execute(text("SET TRANSACTION READ ONLY"))
        if caller.is_end_user:
            conn.execute(text("SET LOCAL ROLE authenticated"))
            _set_claims(conn, caller.claims)
        else:
            conn.execute(text(f'SET LOCAL ROLE "{agent_role_name(agent_id)}"'))
        search_path = ", ".join(f'"{s}"' for s in schemas) or '""'
        conn.execute(text(f"SET LOCAL search_path TO {search_path}"))
        yield conn


def _check_references(
    conn: Connection, parsed: ParsedSelect, allowed: dict[str, list[str]]
) -> None:
    """Resolve what the query names, as the query will, and hold it to the allowlist."""
    permitted = {(schema, name) for schema, names in allowed.items() for name in names or []}
    for schema, name in parsed.relations:
        row = conn.execute(
            text(
                """
                SELECT n.nspname, c.relname
                FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.oid = to_regclass(
                    CASE WHEN CAST(:s AS text) IS NULL THEN format('%I', CAST(:t AS text))
                         ELSE format('%I.%I', CAST(:s AS text), CAST(:t AS text)) END)
                """
            ),
            {"s": schema, "t": name},
        ).first()
        label = f"{schema}.{name}" if schema else name
        if row is None or (row[0], row[1]) not in permitted:
            raise AgentSqlRejected(
                f"Table {label} is not in this agent's configured tables: "
                f"{sorted(f'{s}.{t}' for s, t in permitted)}"
            )

    names = sorted(set(parsed.functions))
    if not names:
        return
    builtin = set(
        conn.execute(
            text(
                "SELECT DISTINCT proname FROM pg_proc "
                "WHERE pronamespace = 'pg_catalog'::regnamespace AND proname = ANY(:n)"
            ),
            {"n": names},
        ).scalars()
    )
    shadowed = set(
        conn.execute(
            text(
                """
                SELECT DISTINCT p.proname FROM pg_proc p
                JOIN pg_namespace n ON n.oid = p.pronamespace
                WHERE p.proname = ANY(:n) AND n.nspname = ANY(current_schemas(false))
                  AND n.nspname <> 'pg_catalog'
                """
            ),
            {"n": names},
        ).scalars()
    )
    for name in names:
        if name not in builtin or name in shadowed:
            raise AgentSqlRejected(
                f"Function {name} is not allowed: only built-in functions may be called"
            )


def run_query(
    caller: ToolCaller, agent_id: str, sql: str, schemas_config: dict[str, list[str]]
) -> list[dict]:
    """Run an agent's read-only query as ``caller``. Raises AgentSqlRejected."""
    parsed = parse_select(sql)
    with agent_transaction(caller, agent_id, list(schemas_config), read_only=True) as conn:
        _check_references(conn, parsed, schemas_config)
        return [dict(row._mapping) for row in conn.execute(text(sql))]
