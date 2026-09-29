"""Run agent database tools as the caller, never as the service's own login.

The service connects as a superuser, so SQL run on its connection ignores
every privilege and RLS policy. Agent tools therefore use logins of their own,
none of them a superuser:

* ``powabase_agent_user`` — for runs an end user started. It may only become
  ``authenticated``; each transaction does, with that user's JWT claims, so
  the project's grants and RLS policies for that user apply.
* ``powabase_agent_<agent id>`` — one login per agent, for runs started with
  the service role key. It holds grants on exactly the tables configured on
  the agent's database tools and bypasses RLS on those alone. It is a member
  of no other role, so it cannot take on another agent's grants.

The caller's identity lives in session settings that SQL could overwrite
(``set_config('request.jwt.claims', ...)``), so free-form SQL from
``database_query`` is parsed before it runs and may only call an allowlist of
built-in functions and operators: see :func:`parse_select`. An allowlist
rather than a denylist, because some built-ins run a SQL string they are
handed (``ts_stat``, ``query_to_xml``) and no denylist stays complete.

Known limit: Postgres also runs functions implicitly — a domain's CHECK when
a value is written, the equality operator of a column's own type when a query
groups or joins on it. Those come from objects a privileged role already
created in the schemas an agent is given; configure agents only with tables
whose types you trust.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from pglast import ast, parse_sql
from pglast.enums.parsenodes import A_Expr_Kind
from pglast.parser import ParseError
from pglast.stream import RawStream
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import NullPool

from ..db import AI_SCHEMA, db
from .tool_caller import ToolCaller

logger = logging.getLogger(__name__)

USER_LOGIN = "powabase_agent_user"
_AGENT_ROLE_PREFIX = "powabase_agent_"
_AGENT_ROLE_PATTERN = re.compile(r"^powabase_agent_[0-9a-f]{32}$")

# Every tool transaction gives up rather than hold a connection or a lock.
STATEMENT_TIMEOUT = "30s"
LOCK_TIMEOUT = "5s"

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
    """The SQL is not something an agent's query tool may run.

    The message is meant for the model: it says what to change, and carries
    no SQL text, parameters or server details.
    """


class AgentToolsUnavailable(AgentSqlRejected):
    """The database tools cannot run here (not set up, or the session expired)."""


def agent_role_name(agent_id: str) -> str:
    """The login a service-role run of this agent's database tools uses."""
    return f"{_AGENT_ROLE_PREFIX}{uuid.UUID(str(agent_id)).hex}"


def _is_protected_schema(schema: str) -> bool:
    return schema in _PROTECTED_SCHEMAS or schema.startswith("pg_")


def _ident(name: str) -> str:
    """Quote an identifier for SQL, as ``quote_ident`` does."""
    return '"' + name.replace('"', '""') + '"'


# ---------------------------------------------------------------------------
# The SQL gate
# ---------------------------------------------------------------------------

# The only functions an agent's query may call: built-ins that compute on
# their arguments. None of them changes a setting, reads one, or runs SQL.
# Each must also exist in pg_catalog and not be shadowed on the search path,
# which is checked in the database (see _check_references).
ALLOWED_FUNCTIONS = frozenset(
    {
        # aggregates
        "count", "sum", "avg", "min", "max", "string_agg", "array_agg", "bool_and",
        "bool_or", "every", "json_agg", "jsonb_agg", "json_object_agg",
        "jsonb_object_agg", "stddev", "stddev_pop", "stddev_samp", "variance",
        "var_pop", "var_samp", "percentile_cont", "percentile_disc", "mode", "corr",
        "covar_pop", "covar_samp", "regr_slope", "regr_intercept", "bit_and", "bit_or",
        # window
        "row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile",
        "lag", "lead", "first_value", "last_value", "nth_value",
        # strings
        "lower", "upper", "initcap", "length", "char_length", "character_length",
        "octet_length", "substr", "substring", "left", "right", "btrim", "ltrim",
        "rtrim", "replace", "concat", "concat_ws", "position", "strpos", "split_part",
        "regexp_replace", "regexp_matches", "regexp_match", "regexp_split_to_array",
        "regexp_split_to_table", "regexp_count", "regexp_instr", "regexp_like",
        "regexp_substr", "lpad", "rpad", "reverse", "repeat", "format", "md5",
        "sha256", "starts_with", "translate", "chr", "ascii", "to_hex", "encode",
        "decode", "overlay", "similar_to_escape", "normalize", "quote_literal",
        "quote_ident",
        # formatting
        "to_char", "to_number", "to_date", "to_timestamp",
        # numbers
        "abs", "ceil", "ceiling", "floor", "round", "trunc", "mod", "power", "pow",
        "sqrt", "cbrt", "exp", "ln", "log", "log10", "sign", "pi", "random", "degrees",
        "radians", "div", "gcd", "lcm", "width_bucket", "sin", "cos", "tan", "asin",
        "acos", "atan", "atan2",
        # dates and times
        "now", "date_trunc", "date_part", "extract", "age", "make_date", "make_time",
        "make_timestamp", "make_timestamptz", "make_interval", "justify_days",
        "justify_hours", "justify_interval", "timezone", "date_bin", "isfinite",
        "clock_timestamp", "statement_timestamp", "transaction_timestamp", "overlaps",
        # json
        "json_build_object", "jsonb_build_object", "json_build_array",
        "jsonb_build_array", "json_object", "jsonb_object", "json_array_length",
        "jsonb_array_length", "json_each", "jsonb_each", "json_each_text",
        "jsonb_each_text", "json_array_elements", "jsonb_array_elements",
        "json_array_elements_text", "jsonb_array_elements_text", "json_extract_path",
        "jsonb_extract_path", "json_extract_path_text", "jsonb_extract_path_text",
        "json_typeof", "jsonb_typeof", "to_json", "to_jsonb", "json_object_keys",
        "jsonb_object_keys", "jsonb_pretty", "jsonb_set", "jsonb_insert",
        "jsonb_strip_nulls", "json_strip_nulls", "jsonb_path_query",
        "jsonb_path_query_array", "jsonb_path_query_first", "jsonb_path_exists",
        "jsonb_path_match", "row_to_json", "array_to_json",
        # arrays and sets
        "array_length", "array_lower", "array_upper", "cardinality", "unnest",
        "array_to_string", "string_to_array", "array_position", "array_positions",
        "array_remove", "array_replace", "array_append", "array_prepend", "array_cat",
        "array_dims", "generate_series", "generate_subscripts",
        # text search (not ts_stat / ts_rewrite, which run a SQL string)
        "to_tsvector", "to_tsquery", "plainto_tsquery", "phraseto_tsquery",
        "websearch_to_tsquery", "ts_rank", "ts_rank_cd", "ts_headline", "setweight",
        # misc
        "gen_random_uuid", "num_nulls", "num_nonnulls",
        # function-style casts to built-in types
        "int2", "int4", "int8", "float4", "float8", "text", "date", "timestamptz",
        "bool",
    }
)  # fmt: skip

ALLOWED_OPERATORS = frozenset(
    {
        "=", "<>", "<", ">", "<=", ">=", "+", "-", "*", "/", "%", "^", "||", "|/",
        "@", "&", "|", "#", "~", "<<", ">>",
        "~~", "!~~", "~~*", "!~~*", "!~", "~*", "!~*",
        "->", "->>", "#>", "#>>", "@>", "<@", "?", "?|", "?&", "&&", "@?", "@@", "#-",
    }
)  # fmt: skip

# Casting text to these looks up catalog objects by name, which reads the
# catalogs rather than the configured tables.
_LOOKUP_TYPES = frozenset(
    {
        "regclass", "regproc", "regprocedure", "regoper", "regoperator", "regtype",
        "regrole", "regnamespace", "regconfig", "regdictionary", "regcollation",
    }
)  # fmt: skip

_BETWEEN_KINDS = frozenset(
    {
        A_Expr_Kind.AEXPR_BETWEEN,
        A_Expr_Kind.AEXPR_NOT_BETWEEN,
        A_Expr_Kind.AEXPR_BETWEEN_SYM,
        A_Expr_Kind.AEXPR_NOT_BETWEEN_SYM,
    }
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
    """What a query reads and runs.

    ``relations`` as (schema or None, name); ``functions`` and ``operators``
    by name; ``types`` as the names of the types it casts to.
    """

    relations: list[tuple[str | None, str]] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    operators: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)


def parse_select(sql: str) -> ParsedSelect:
    """Parse ``sql`` and return what it reads and runs, or raise AgentSqlRejected.

    Accepts exactly one SELECT that writes and locks nothing, calling only
    allowlisted functions and operators (unqualified or in ``pg_catalog``)
    and casting only to ``pg_catalog`` types. Tables are reported, not
    judged: the table allowlist, and whether each name really resolves to a
    built-in, are checked in the database, where names resolve exactly as the
    query's will.
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


def _names(nodes) -> list[str]:
    return [node.sval for node in nodes]


def _add_operator(parts: list[str], parsed: ParsedSelect) -> None:
    symbol = parts[-1]
    if (len(parts) > 1 and parts[0] != "pg_catalog") or symbol not in ALLOWED_OPERATORS:
        raise AgentSqlRejected(f"Operator {'.'.join(parts)} is not allowed")
    parsed.operators.append(symbol)


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
        parts = _names(node.funcname)
        name = parts[-1].lower()
        if (len(parts) > 1 and parts[0] != "pg_catalog") or name not in ALLOWED_FUNCTIONS:
            raise AgentSqlRejected(
                f"Function {'.'.join(parts)} is not allowed: only a fixed set of "
                "built-in functions may be called"
            )
        parsed.functions.append(name)

    elif isinstance(node, ast.A_Expr):
        if node.kind in _BETWEEN_KINDS:
            parsed.operators.extend([">=", "<="])
        else:
            _add_operator(_names(node.name), parsed)

    elif isinstance(node, ast.SubLink) and node.operName:
        _add_operator(_names(node.operName), parsed)

    elif isinstance(node, ast.SortBy) and node.useOp:
        _add_operator(_names(node.useOp), parsed)

    elif isinstance(node, ast.TypeName):
        parts = _names(node.names)
        if (len(parts) > 1 and parts[0] != "pg_catalog") or parts[-1] in _LOOKUP_TYPES:
            raise AgentSqlRejected(f"Casting to {'.'.join(parts)} is not allowed")
        parsed.types.append(RawStream()(node))
        return

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

    # Arguments, filters, window specs and operands can hold subqueries too.
    for attr in node:
        _walk(getattr(node, attr), visible_ctes, parsed)


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

_LOGIN_ATTRIBUTES = "LOGIN NOSUPERUSER NOBYPASSRLS NOINHERIT NOCREATEDB NOCREATEROLE NOREPLICATION"
_AGENT_ATTRIBUTES = "LOGIN NOSUPERUSER BYPASSRLS NOINHERIT NOCREATEDB NOCREATEROLE NOREPLICATION"

# Whether this process set up the end-user login. Until it has, end-user
# runs' database tools refuse rather than fail on a missing login.
_user_login_ready = False


def _set_password(conn: Connection, role: str) -> None:
    password = db.engine.url.password
    if password is None:
        return
    statement = conn.execute(
        text("SELECT format('ALTER ROLE %I PASSWORD %L', CAST(:r AS text), CAST(:p AS text))"),
        {"r": role, "p": password},
    ).scalar_one()
    conn.execute(text(statement))


def _role_attributes_match(conn: Connection, role: str, bypass_rls: bool) -> bool | None:
    """None when the role is missing; else whether its attributes are as set here."""
    row = conn.execute(
        text(
            "SELECT rolcanlogin, rolsuper, rolbypassrls, rolinherit, rolcreatedb, "
            "rolcreaterole, rolreplication FROM pg_roles WHERE rolname = :r"
        ),
        {"r": role},
    ).first()
    if row is None:
        return None
    return tuple(row) == (True, False, bypass_rls, False, False, False, False)


def _ensure_login(conn: Connection, role: str, attributes: str, bypass_rls: bool) -> None:
    """Create the login, or correct its attributes only when they differ."""
    matches = _role_attributes_match(conn, role, bypass_rls)
    if matches is None:
        conn.execute(text(f"CREATE ROLE {_ident(role)} {attributes}"))
    elif not matches:
        conn.execute(text(f"ALTER ROLE {_ident(role)} {attributes}"))
    _set_password(conn, role)


def ensure_login_roles() -> None:
    """Create or refresh the end-user tool login. Run at boot, as the service's login.

    Its password is the service's own, as every other service in a project
    stack logs in with its own role and the shared database password; setting
    it on every boot follows a rotation.
    """
    global _user_login_ready
    with db.engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext('powabase_agent_logins'))"))
        _ensure_login(conn, USER_LOGIN, _LOGIN_ATTRIBUTES, bypass_rls=False)
        if conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = 'authenticated'")).first():
            conn.execute(text(f"GRANT authenticated TO {_ident(USER_LOGIN)}"))
        else:
            logger.warning(
                "No 'authenticated' role in this database; agent tools cannot run for end users"
            )
            return
    _user_login_ready = True


def reconcile_agent_roles() -> None:
    """Drop agent logins whose agent is gone; refresh the rest's passwords. Run at boot."""
    with db.engine.connect() as conn:
        roles = [
            name
            for name in conn.execute(
                text("SELECT rolname FROM pg_roles WHERE rolname LIKE 'powabase\\_agent\\_%'")
            ).scalars()
            if _AGENT_ROLE_PATTERN.match(name)
        ]
        agents = {
            uuid.UUID(str(agent_id)).hex
            for agent_id in conn.execute(text(f'SELECT id FROM "{AI_SCHEMA}".agents')).scalars()
        }
    for role in roles:
        agent_id = str(uuid.UUID(role.removeprefix(_AGENT_ROLE_PREFIX)))
        try:
            if role.removeprefix(_AGENT_ROLE_PREFIX) in agents:
                with db.engine.begin() as conn:
                    _set_password(conn, role)
            else:
                drop_agent_role(agent_id)
        except Exception:
            logger.exception("Could not reconcile the database login %s", role)


def _desired_grants(read_tables, write_tables) -> dict[tuple[str, str], set[str]]:
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
    return desired


def _readable_relation(conn: Connection, schema: str, name: str) -> bool:
    """A plain or partitioned table, or a view that runs as its caller.

    Other views run as their owner, so their rows are not filtered by the
    caller's RLS policies; materialized views and foreign tables have none.
    """
    return bool(
        conn.execute(
            text(
                """
                SELECT c.relkind IN ('r', 'p')
                    OR (c.relkind = 'v' AND coalesce(c.reloptions, '{}')
                        && ARRAY['security_invoker=true', 'security_invoker=on',
                                 'security_invoker=1'])
                FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = :s AND c.relname = :t
                """
            ),
            {"s": schema, "t": name},
        ).scalar()
    )


def sync_agent_role(
    agent_id: str,
    read_tables: dict[str, list[str]],
    write_tables: dict[str, list[str]],
) -> None:
    """Make the agent's login hold exactly the grants its database tools configure.

    ``read_tables`` get SELECT; ``write_tables`` get SELECT, INSERT, UPDATE and
    DELETE plus the sequences their columns draw from. Relations that do not
    exist, are not tables or security-invoker views, or sit in protected
    schemas are skipped. Revocations commit first, in their own transaction,
    so a failure later can only leave the login with less than it had. If the
    agent no longer exists, its login is dropped instead. Raises on failure.
    """
    role = agent_role_name(agent_id)
    desired = _desired_grants(read_tables, write_tables)

    with db.engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(CAST(:r AS text)))"), {"r": role})
        agent_exists = conn.execute(
            text(f'SELECT 1 FROM "{AI_SCHEMA}".agents WHERE id = CAST(:id AS uuid)'),
            {"id": str(uuid.UUID(str(agent_id)))},
        ).first()
    if not agent_exists:
        drop_agent_role(agent_id)
        return

    with db.engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(CAST(:r AS text)))"), {"r": role})
        _ensure_login(conn, role, _AGENT_ATTRIBUTES, bypass_rls=True)
        oid = conn.execute(text("SELECT oid FROM pg_roles WHERE rolname = :r"), {"r": role}).scalar()
        granted = conn.execute(
            text(
                """
                SELECT n.nspname, c.relname, c.relkind, array_agg(a.privilege_type)
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                CROSS JOIN LATERAL aclexplode(c.relacl) a
                WHERE a.grantee = :oid
                GROUP BY 1, 2, 3
                """
            ),
            {"oid": oid},
        ).all()
        existing = {(s, t): set(p) for s, t, kind, p in granted if kind != "S"}
        for (schema, name), privileges in existing.items():
            extra = privileges - desired.get((schema, name), set())
            if extra:
                conn.execute(
                    text(
                        f"REVOKE {', '.join(sorted(extra))} ON TABLE "
                        f"{_ident(schema)}.{_ident(name)} FROM {_ident(role)}"
                    )
                )
        for schema, name, kind, _ in granted:
            if kind == "S":
                conn.execute(
                    text(
                        f"REVOKE ALL ON SEQUENCE {_ident(schema)}.{_ident(name)} FROM {_ident(role)}"
                    )
                )
        for (schema,) in conn.execute(
            text(
                "SELECT n.nspname FROM pg_namespace n CROSS JOIN LATERAL aclexplode(n.nspacl) a "
                "WHERE a.grantee = :oid"
            ),
            {"oid": oid},
        ):
            conn.execute(text(f"REVOKE USAGE ON SCHEMA {_ident(schema)} FROM {_ident(role)}"))

    with db.engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(CAST(:r AS text)))"), {"r": role})
        for (schema, name), privileges in desired.items():
            if not _readable_relation(conn, schema, name):
                continue
            conn.execute(text(f"GRANT USAGE ON SCHEMA {_ident(schema)} TO {_ident(role)}"))
            conn.execute(
                text(
                    f"GRANT {', '.join(sorted(privileges))} ON TABLE "
                    f"{_ident(schema)}.{_ident(name)} TO {_ident(role)}"
                )
            )
            if "INSERT" not in privileges:
                continue
            for (sequence,) in conn.execute(
                text(
                    """
                    SELECT s.oid::regclass::text
                    FROM pg_depend d
                    JOIN pg_class s ON s.oid = d.objid AND s.relkind = 'S'
                    JOIN pg_class t ON t.oid = d.refobjid
                    JOIN pg_namespace n ON n.oid = t.relnamespace
                    WHERE n.nspname = :s AND t.relname = :t AND d.deptype IN ('a', 'i')
                    """
                ),
                {"s": schema, "t": name},
            ):
                conn.execute(text(f"GRANT USAGE ON SEQUENCE {sequence} TO {_ident(role)}"))


def drop_agent_role(agent_id: str) -> None:
    """Remove the agent's login and every grant it holds."""
    role = agent_role_name(agent_id)
    with db.engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(CAST(:r AS text)))"), {"r": role})
        if conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": role}).first():
            conn.execute(text(f"DROP OWNED BY {_ident(role)}"))
            conn.execute(text(f"DROP ROLE {_ident(role)}"))
    with _engines_lock:
        engine = _engines.pop(role, None)
    if engine is not None:
        engine.dispose()


# ---------------------------------------------------------------------------
# Running tool SQL
# ---------------------------------------------------------------------------

_engines: dict[str, Engine] = {}
_engines_lock = threading.Lock()


def _engine(login: str) -> Engine:
    """An engine logged in as ``login``, with the service's host and password.

    The end-user login keeps a small pool. Agent logins connect per
    transaction: there is one per agent, and a pool each would add up.
    """
    with _engines_lock:
        engine = _engines.get(login)
        if engine is None:
            url = db.engine.url.set(username=login)
            if login == USER_LOGIN:
                engine = create_engine(
                    url, pool_size=1, max_overflow=4, pool_pre_ping=True, pool_recycle=1800
                )
            else:
                engine = create_engine(url, poolclass=NullPool)
            _engines[login] = engine
        return engine


_CLAIM_KEY = re.compile(r"^[a-z_][a-z0-9_]*$")


def _set_claims(conn: Connection, claims) -> None:
    """Expose the caller's JWT claims to RLS, both ways ``auth.uid()`` reads them.

    Newer Supabase databases read the ``request.jwt.claims`` JSON; older ones
    read one ``request.jwt.claim.<name>`` setting per claim, as PostgREST used
    to set them. Both are set so policies work on either.
    """
    claims = dict(claims)
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


def _check_caller(caller: ToolCaller) -> None:
    if not caller.is_end_user:
        return
    if not _user_login_ready:
        raise AgentToolsUnavailable("Database tools are not set up for end users on this server")
    expires = caller.claims.get("exp")
    if isinstance(expires, (int, float)) and expires < time.time():
        raise AgentToolsUnavailable("The user's session has expired; ask them to sign in again")


@contextmanager
def agent_transaction(
    caller: ToolCaller, agent_id: str, schemas: list[str], *, read_only: bool
) -> Iterator[Connection]:
    """A transaction on a tool login, acting as ``caller``. Commits on success.

    An end user's transaction is ``authenticated`` with their JWT claims; a
    service-role transaction is the agent's own login (see
    :func:`sync_agent_role`). Neither login is a superuser.
    """
    _check_caller(caller)
    login = USER_LOGIN if caller.is_end_user else agent_role_name(agent_id)
    with _engine(login).connect() as conn, conn.begin():
        if read_only:
            conn.execute(text("SET TRANSACTION READ ONLY"))
        conn.execute(text(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'"))
        conn.execute(text(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'"))
        if caller.is_end_user:
            conn.execute(text("SET LOCAL ROLE authenticated"))
            _set_claims(conn, caller.claims)
        search_path = ", ".join(_ident(s) for s in schemas) or '""'
        conn.execute(text(f"SET LOCAL search_path TO {search_path}"))
        yield conn


def _check_references(
    conn: Connection, parsed: ParsedSelect, allowed: dict[str, list[str]]
) -> None:
    """Resolve what the query names, as the query will, and hold it to the allowlists."""
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
        if (
            row is None
            or (row[0], row[1]) not in permitted
            or not _readable_relation(conn, row[0], row[1])
        ):
            raise AgentSqlRejected(
                f"Table {label} is not in this agent's configured tables: "
                f"{sorted(f'{s}.{t}' for s, t in permitted)}"
            )

    names = sorted(set(parsed.functions))
    if names:
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
                    f"Function {name} is not allowed: only a fixed set of built-in "
                    "functions may be called"
                )

    operators = sorted(set(parsed.operators))
    if operators:
        # An operator on the search path outside pg_catalog could win
        # resolution for some operand types; one implemented in SQL or PL/pgSQL
        # could run anything. Extension operators written in C are fine.
        unsafe = conn.execute(
            text(
                """
                SELECT DISTINCT o.oprname FROM pg_operator o
                JOIN pg_namespace n ON n.oid = o.oprnamespace
                JOIN pg_proc p ON p.oid = o.oprcode
                JOIN pg_language l ON l.oid = p.prolang
                WHERE o.oprname = ANY(:ops) AND n.nspname = ANY(current_schemas(false))
                  AND n.nspname <> 'pg_catalog' AND l.lanname NOT IN ('internal', 'c')
                """
            ),
            {"ops": operators},
        ).scalars().all()
        if unsafe:
            raise AgentSqlRejected(f"Operator {unsafe[0]} is not allowed in this schema")

    for type_name in sorted(set(parsed.types)):
        row = conn.execute(
            text(
                "SELECT t.typnamespace = 'pg_catalog'::regnamespace AND t.typtype <> 'd' "
                "FROM pg_type t WHERE t.oid = to_regtype(:t)"
            ),
            {"t": type_name},
        ).first()
        if row is None or not row[0]:
            raise AgentSqlRejected(f"Casting to {type_name} is not allowed")


def run_query(
    caller: ToolCaller, agent_id: str, sql: str, schemas_config: dict[str, list[str]]
) -> list[dict]:
    """Run an agent's read-only query as ``caller``. Raises AgentSqlRejected."""
    parsed = parse_select(sql)
    with agent_transaction(caller, agent_id, list(schemas_config), read_only=True) as conn:
        _check_references(conn, parsed, schemas_config)
        return [dict(row._mapping) for row in conn.execute(text(sql))]
