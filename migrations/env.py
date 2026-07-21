import logging
from logging.config import fileConfig

from alembic import context
from alembic.script import ScriptDirectory
from flask import current_app

from agentic_project_service._alembic_self_heal import cleanup_orphan_versions

config = context.config
try:
    if config.config_file_name is not None:
        fileConfig(config.config_file_name)
except KeyError:
    pass
logger = logging.getLogger("alembic.env")

MANAGED_SCHEMAS = {"ai"}
EXCLUDED_SCHEMAS = {"auth", "extensions", "supabase_functions", "realtime", "public", "storage"}


def include_name(name, type_, parent_names):
    """Only manage objects in the 'ai' schema."""
    if type_ == "schema":
        return name in MANAGED_SCHEMAS
    # Exclude indexes — they are managed by ai_schema.sql
    if type_ == "index":
        return False
    return True


def render_item(type_, obj, autogen_context):
    """Custom rendering for pgvector Vector type."""
    if type_ == "type" and hasattr(obj, "__class__") and obj.__class__.__name__ == "Vector":
        autogen_context.imports.add("from pgvector.sqlalchemy import Vector")
        return f"Vector({obj.dim})"
    return False


def get_engine():
    try:
        return current_app.extensions["migrate"].db.get_engine()
    except (TypeError, AttributeError):
        return current_app.extensions["migrate"].db.engine


def get_engine_url():
    try:
        return get_engine().url.render_as_string(hide_password=False).replace("%", "%%")
    except AttributeError:
        return str(get_engine().url).replace("%", "%%")


def get_metadata():
    target_db = current_app.extensions["migrate"].db
    if hasattr(target_db, "metadatas"):
        return target_db.metadatas[None]
    return target_db.metadata


def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = get_engine_url()
    context.configure(
        url=url,
        target_metadata=get_metadata(),
        literal_binds=True,
        include_schemas=True,
        version_table_schema="ai",
        include_name=include_name,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = get_engine()

    with connectable.connect() as connection:
        # Self-heal: remove orphan rows in ai.alembic_version that point at
        # revision names no longer present on disk. See
        # cleanup_orphan_versions() docstring for rationale. Runs in its
        # OWN short transaction so the orphan cleanup commits independently
        # of the main migration block — that way a downstream migration
        # failure doesn't roll the cleanup back into a corrupted state.
        script_dir = ScriptDirectory.from_config(config)
        with connection.begin():
            cleanup_orphan_versions(connection, script_dir)

        context.configure(
            connection=connection,
            target_metadata=get_metadata(),
            include_schemas=True,
            version_table_schema="ai",
            include_name=include_name,
            render_item=render_item,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
