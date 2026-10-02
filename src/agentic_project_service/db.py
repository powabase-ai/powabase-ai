"""Database connection for the project service.

Connects to the project's dedicated Supabase Postgres instance.
All AI-related tables are in the 'ai' schema.
"""

import logging
import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event, text
from sqlalchemy.exc import DisconnectionError
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)


# The schema name where AI tables live
AI_SCHEMA = "ai"


def commit_scoped_write(sql: str, params: dict, *, what: str) -> int:
    """Run one scoped UPDATE, commit it, and return the rows it matched.

    Exists because the same defect keeps being written by hand. A status write
    placed inside a loop, or inside an ``except`` block, will strand every
    remaining item if it raises: the exception escapes the handler, escapes the
    loop, and the sibling rows never get processed. That is worst precisely
    where these writes cluster -- the recovery paths, whose whole job is to
    stop rows being stranded, and which run over every affected row at once.

    So this never propagates. A failed write rolls the session back (leaving it
    usable for the next iteration rather than poisoned) and returns 0.

    The rowcount is returned rather than discarded because these writes are all
    scoped -- on ownership, or on status -- and 0 rows is a real outcome that
    means "someone else moved this row on", not an error. A caller that wants
    to log the difference can; one that does not is at least not pretending the
    write landed.

    ``what`` names the write for the failure log, e.g. "watchdog terminal".
    """
    try:
        result = db.session.execute(text(sql), params)
        db.session.commit()
        return result.rowcount
    except Exception:
        # Roll back so the session is usable for whatever runs next. Without
        # this a single failure inside a loop poisons every later iteration
        # too, turning one lost row into all of them.
        try:
            db.session.rollback()
        except Exception:
            logger.error("Rollback failed after %s", what, exc_info=True)
        logger.error("Scoped write failed (%s); row left as-is", what, exc_info=True)
        return 0


def get_database_url() -> str:
    """Get the database URL for this project's Supabase instance."""
    url = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@db:5432/postgres")
    # Ensure we use psycopg (psycopg3) driver
    # SQLAlchemy requires "postgresql://" not "postgres://"
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


# How long a pooled connection may live, and how many times it may be handed
# out, before the pool replaces it. See ``engine_options``. 0 turns a bound off.
DB_POOL_RECYCLE_SECONDS = 300
DB_POOL_MAX_CHECKOUTS = 200

_CHECKOUTS_KEY = "agentic_checkouts"


def _int_env(name: str, default: int) -> int:
    """A non-negative integer from the environment, or ``default`` if unusable."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        logger.warning("%s=%r is not a non-negative integer; using %d", name, raw, default)
        return default
    return value


def engine_options() -> dict:
    """Engine options that keep a pooled connection's memory bounded.

    A PostgreSQL backend never evicts its relation cache: every table and index
    a connection has opened stays in that backend's memory until it closes. The
    item tables are partitioned by knowledge base, so a project has thousands of
    relations, and the pool's connections are long-lived. Two things follow.

    **No server-side prepared statements** (``prepare_threshold=None``). psycopg
    prepares a statement after 5 executions on a connection, and after 5 more
    PostgreSQL builds its generic plan to compare. A statement that names its
    knowledge base as a bound parameter -- ``knowledge_base_id = $1`` -- cannot
    be pruned in a generic plan, so building one opens every knowledge base's
    partition and all their indexes; the plan is thrown away, the relcache is
    not. Measured on 513 partitions: 7 MB to 76-162 MB per connection at its
    eleventh search. Unprepared, every execution is planned with its values and
    prunes to one partition. The cost is re-planning each execution, which
    these statements were doing anyway for their first ten.

    **A bounded connection lifetime** (``pool_recycle``, and the checkout limit
    in ``limit_connection_reuse``). Even pruned, a connection keeps about 85 kB
    for each distinct knowledge base it has ever served.

    ``pool_pre_ping`` is here for a different reason: after PostgreSQL restarts,
    every pooled connection is dead, and without the ping each one fails a
    request before the pool replaces it.
    """
    options: dict = {
        "pool_pre_ping": True,
        "connect_args": {"prepare_threshold": None},
    }
    recycle = _int_env("DB_POOL_RECYCLE_SECONDS", DB_POOL_RECYCLE_SECONDS)
    if recycle:
        options["pool_recycle"] = recycle
    return options


def limit_connection_reuse(engine) -> None:
    """Replace a pooled connection after ``DB_POOL_MAX_CHECKOUTS`` checkouts.

    ``pool_recycle`` bounds a connection's age, which bounds what it has cached
    only at a given request rate. This bounds it by use, whatever the rate.
    Raising ``DisconnectionError`` from a checkout handler is the pool's own
    protocol for "discard this connection and hand out a fresh one".
    """
    limit = _int_env("DB_POOL_MAX_CHECKOUTS", DB_POOL_MAX_CHECKOUTS)
    if not limit:
        return

    @event.listens_for(engine, "checkout")
    def _retire_after_limit(dbapi_connection, connection_record, connection_proxy):
        checkouts = connection_record.info.get(_CHECKOUTS_KEY, 0) + 1
        if checkouts > limit:
            raise DisconnectionError(f"pooled connection retired after {limit} checkouts")
        connection_record.info[_CHECKOUTS_KEY] = checkouts
