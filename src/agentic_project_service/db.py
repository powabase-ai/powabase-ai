"""Database connection for the project service.

Connects to the project's dedicated Supabase Postgres instance.
All AI-related tables are in the 'ai' schema.
"""

import logging
import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
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
