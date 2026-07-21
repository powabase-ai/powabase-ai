"""Database connection for the project service.

Connects to the project's dedicated Supabase Postgres instance.
All AI-related tables are in the 'ai' schema.
"""

import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)


# The schema name where AI tables live
AI_SCHEMA = "ai"


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
