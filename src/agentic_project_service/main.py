"""Agentic Project Service - Flask Application

Entry point for the project service that handles AI features:
sources, knowledge bases, agents, and background tasks.

Run with: python -m agentic_project_service.main
"""

import logging
import os
import time

from alembic import command as alembic_command
from flask import Flask, g, jsonify, request
from sqlalchemy import text
from flask_cors import CORS
from sqlalchemy import inspect

from .celery import init_celery
from .db import db, get_database_url
from .migrate import migrate
from .routes.sources import sources_bp
from .routes.knowledge_bases import knowledge_bases_bp
from .routes.agents import agents_bp
from .routes.sessions import sessions_bp
from .routes.context_handlers import context_handlers_bp
from .routes.enrichment import enrichment_bp
from .routes.workflows import workflows_bp
from .routes.webhooks import webhooks_bp
from .routes.database import database_bp
from .routes.config import config_bp
from .routes.copilot import copilot_bp
from .routes.tools import tools_bp
from .routes.orchestrations import orchestrations_bp
from .routes.settings import settings_bp
from .routes.ai_provider_keys import ai_provider_keys_bp
from .routes.observability import observability_bp
from .routes.internal import internal_bp

# Import ORM models so they register with SQLAlchemy metadata
from .models import tenant  # noqa: F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(testing: bool = False):
    """Create and configure the Flask application.

    When ``testing=True``, skips DB initialization and Alembic migrations so
    unit tests can construct the app without a live Postgres. All Flask-level
    wiring (routes, blueprints, before/after_request hooks) is still applied.
    """
    app = Flask(__name__)

    # CORS configuration
    # In production, set CORS_ORIGINS to specific domains
    # For development, allow all origins
    cors_origins_env = os.getenv("CORS_ORIGINS", "")
    if cors_origins_env:
        cors_origins = cors_origins_env.split(",")
    else:
        # Development default: allow all origins
        cors_origins = "*"

    CORS(
        app,
        resources={
            r"/*": {
                "origins": cors_origins,
                "methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
                "expose_headers": ["Content-Range", "X-Content-Range"],
                "supports_credentials": True if cors_origins != "*" else False,
            }
        },
    )

    # Prometheus /metrics endpoint — per-project request histogram, in-flight
    # gauge, and counter. Labeled so operators can split by endpoint and
    # project (project_ref is emitted as an env-level label at deploy time;
    # see infra/helm/project-stack/templates/podmonitor.yaml).
    try:
        from prometheus_flask_exporter import PrometheusMetrics

        PrometheusMetrics(
            app,
            defaults_prefix="project_api",
            group_by="endpoint",
        )
    except ImportError:
        # prometheus-flask-exporter is a soft dep; skip /metrics if unavailable.
        pass

    # Per-request memory observability (issue #107).
    # Logs RSS at request entry and response build, plus delta and duration.
    # Note: for streaming responses, after_request fires when headers are
    # finalized — body completion is not yet captured here. The cluster-side
    # memory-sampler covers the longitudinal view.
    def _read_rss_mb():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) // 1024
        except Exception:
            return None

    @app.before_request
    def _obs_request_start():
        g._obs_rss_start = _read_rss_mb()
        g._obs_t0 = time.monotonic()

    # Billing is an OPTIONAL adapter. The cloud edition ships services/billing_cloud
    # and installs the real credit-metering adapter here; the OSS edition ships
    # without that package, so the import fails and the no-op adapter (default in
    # billing_port) stands. This guarded import is the ONLY billing seam the public
    # core is allowed to reference (composition-root carve-out — see
    # test_billing_isolation.py).
    try:
        from .services.billing_cloud import install_billing
    except ImportError:
        logger.info("billing_cloud not present; running with no-op billing (OSS build)")
    else:
        install_billing(app)

    # Boot-time guard: warn loudly if any model exposed in the copilot
    # picker is missing from the deployed litellm's cost map. A picker
    # entry without cost data means every charge for that model fails
    # the response_cost > 0 gate and we'd silently free-ride. Logs at
    # error so the standard alert pipeline catches it; does NOT crash
    # startup so an unrecognized preview model can't take down the pod.
    try:
        import litellm
        from .services.copilot_config import COPILOT_MODEL_OPTIONS

        for _label, model_id in COPILOT_MODEL_OPTIONS:
            try:
                info = litellm.get_model_info(model_id)
            except Exception:
                logger.error(
                    "boot_picker_model_unknown_to_litellm model=%s",
                    model_id,
                )
                continue
            if not info.get("input_cost_per_token"):
                logger.error(
                    "boot_picker_model_missing_cost model=%s — AI-on-us "
                    "billing will silently drop charges for this model",
                    model_id,
                )
    except Exception:  # noqa: BLE001 — guard MUST NOT crash startup
        logger.exception("boot_picker_cost_check_failed")

    @app.after_request
    def _obs_request_end(resp):
        rss_end = _read_rss_mb()
        rss_start = getattr(g, "_obs_rss_start", None)
        delta = (rss_end - rss_start) if (rss_end is not None and rss_start is not None) else None
        dur_ms = (time.monotonic() - getattr(g, "_obs_t0", time.monotonic())) * 1000
        logger.info(
            "[obs] %s %s status=%s dur_ms=%.0f rss_mb=%s d_rss_mb=%s",
            request.method,
            request.path,
            resp.status_code,
            dur_ms,
            rss_end,
            delta,
        )
        return resp

    # Health check endpoint
    @app.route("/health")
    def health():
        return jsonify({"status": "healthy", "service": "project-service"})

    @app.route("/")
    def index():
        return jsonify(
            {
                "service": "agentic-project-service",
                "version": "0.1.0",
                "endpoints": [
                    "/health",
                    "/api/sources",
                    "/api/knowledge-bases",
                    "/api/agents",
                    "/api/sessions",
                    "/api/context-handlers",
                    "/api/config",
                    "/api/database",
                    "/api/workflows",
                    "/api/webhooks",
                    "/api/copilot",
                    "/api/tools",
                    "/api/orchestrations",
                    "/api/settings",
                    "/api/ai-provider-keys",
                ],
            }
        )

    # Register blueprints
    app.register_blueprint(sources_bp)
    app.register_blueprint(knowledge_bases_bp)
    app.register_blueprint(agents_bp)
    app.register_blueprint(sessions_bp)
    app.register_blueprint(context_handlers_bp)
    app.register_blueprint(enrichment_bp)
    app.register_blueprint(workflows_bp)
    app.register_blueprint(webhooks_bp)
    app.register_blueprint(database_bp)
    app.register_blueprint(config_bp)
    app.register_blueprint(copilot_bp)
    app.register_blueprint(tools_bp)
    app.register_blueprint(orchestrations_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(ai_provider_keys_bp)
    app.register_blueprint(observability_bp)
    app.register_blueprint(internal_bp)

    # Bind Flask context to the existing Celery app (broker config
    # lives in celery.py — no second Celery instance created here)
    init_celery(app)

    # Database configuration - connect to project's Supabase Postgres
    app.config["SQLALCHEMY_DATABASE_URI"] = get_database_url()
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    migrate.init_app(app, db, directory="migrations")

    if testing:
        # Unit tests construct the app without a live Postgres. Skip startup
        # schema probes and Alembic migrations; the route/integration test
        # suites that need a real DB call create_app() without testing=True.
        app.config["JWT_SECRET"] = os.getenv("JWT_SECRET")
        return app

    # Migrate CHECK constraint for existing projects to allow 'completed_with_errors'
    with app.app_context():
        try:
            # Check if table exists before querying constraints
            table_exists = db.session.execute(
                text(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'ai' AND table_name = 'enrichment_configs'"
                )
            ).fetchone()
            if not table_exists:
                logger.info("ai.enrichment_configs not yet created, skipping constraint migration")
            else:
                row = db.session.execute(
                    text(
                        "SELECT pg_get_constraintdef(c.oid) "
                        "FROM pg_constraint c "
                        "JOIN pg_namespace n ON n.oid = c.connamespace "
                        "WHERE n.nspname = 'ai' "
                        "AND c.conrelid = '\"ai\".enrichment_configs'::regclass "
                        "AND c.conname = 'enrichment_configs_status_check'"
                    )
                ).fetchone()
                if row is None or "completed_with_errors" not in row[0]:
                    db.session.execute(
                        text(
                            "ALTER TABLE IF EXISTS ai.enrichment_configs "
                            "DROP CONSTRAINT IF EXISTS enrichment_configs_status_check"
                        )
                    )
                    db.session.execute(
                        text(
                            "ALTER TABLE IF EXISTS ai.enrichment_configs "
                            "ADD CONSTRAINT enrichment_configs_status_check "
                            "CHECK (status IN ('idle', 'enriching', 'completed', 'completed_with_errors', 'failed'))"
                        )
                    )
                    db.session.commit()
                    logger.info("Migrated enrichment_configs status CHECK constraint")
                else:
                    logger.info("enrichment_configs status CHECK constraint already up to date")
        except Exception as e:
            logger.error(f"Failed to migrate enrichment_configs constraint: {e}")
            db.session.rollback()

    # Ensure enrichment_configs has max_tokens and use_multimodal columns
    with app.app_context():
        try:
            db.session.execute(
                text(
                    "ALTER TABLE IF EXISTS ai.enrichment_configs "
                    "ADD COLUMN IF NOT EXISTS max_tokens INTEGER DEFAULT 2000"
                )
            )
            db.session.execute(
                text(
                    "ALTER TABLE IF EXISTS ai.enrichment_configs "
                    "ADD COLUMN IF NOT EXISTS use_multimodal BOOLEAN DEFAULT FALSE"
                )
            )
            db.session.commit()
            logger.info("Ensured enrichment_configs columns (max_tokens, use_multimodal)")
        except Exception as e:
            logger.error(f"Failed to ensure enrichment_configs columns: {e}")
            db.session.rollback()

    # Widen sources.file_type for long MIME types (e.g. Office documents)
    with app.app_context():
        try:
            db.session.execute(
                text("ALTER TABLE IF EXISTS ai.sources ALTER COLUMN file_type TYPE VARCHAR(255)")
            )
            db.session.commit()
            logger.info("Ensured sources.file_type is VARCHAR(255)")
        except Exception as e:
            logger.error(f"Failed to widen sources.file_type: {e}")
            db.session.rollback()

    # JWT Secret for auth token validation
    app.config["JWT_SECRET"] = os.getenv("JWT_SECRET")

    # Run database migrations on startup.
    #
    # Use pg_try_advisory_lock (NOT pg_advisory_lock) and commit immediately so
    # that workers that lose the race do NOT sit in `idle in transaction`.
    # The previous pattern self-deadlocks with CREATE INDEX CONCURRENTLY in
    # any migration the holder is running: CIC waits for older transactions
    # to commit, the lock-waiters ARE older transactions, they cannot commit
    # until the holder releases, the holder cannot release until Alembic
    # completes — and Postgres's deadlock detector does not break the cycle
    # (advisory locks + virtxid waits are not in the relation-lock graph).
    # Migration 0019 was the first to use CIC and triggered this in prod.
    # See issue #162 (this fix) and #163 (move migrations out of worker startup).
    with app.app_context():
        try:
            got_lock = db.session.execute(text("SELECT pg_try_advisory_lock(43)")).scalar()
            db.session.commit()  # release the implicit txn immediately

            if got_lock:
                try:
                    # Check if this is an existing DB without alembic_version
                    inspector = inspect(db.engine)
                    has_tables = "sources" in inspector.get_table_names(schema="ai")
                    has_alembic = "alembic_version" in inspector.get_table_names(schema="ai")

                    if has_tables and not has_alembic:
                        # Existing DB: stamp baseline so upgrade() skips it
                        logger.info("Existing DB detected without alembic_version — stamping 0001")
                        alembic_cfg = migrate.get_config()
                        alembic_command.stamp(alembic_cfg, "0001")

                    if has_tables or has_alembic:
                        # Apply any pending migrations
                        alembic_cfg = migrate.get_config()
                        alembic_command.upgrade(alembic_cfg, "head")
                        logger.info("Database migrations applied successfully")
                    else:
                        logger.info(
                            "No ai schema tables found — skipping migrations "
                            "(db-init Job will create them)"
                        )
                finally:
                    db.session.execute(text("SELECT pg_advisory_unlock(43)"))
                    db.session.commit()
            else:
                # Another worker is running migrations. Wait for them to
                # complete by polling alembic_version against the head this
                # code expects. If the schema doesn't exist yet, the holder
                # is on the no-tables branch (skipping migration) — match it.
                inspector = inspect(db.engine)
                has_alembic = "alembic_version" in inspector.get_table_names(schema="ai")
                if not has_alembic:
                    logger.info(
                        "Another worker is initializing the DB; deferring to it (no ai schema yet)"
                    )
                else:
                    from alembic.script import ScriptDirectory

                    alembic_cfg = migrate.get_config()
                    expected_head = ScriptDirectory.from_config(alembic_cfg).get_current_head()
                    logger.info(
                        "Another worker holds the migration lock; "
                        "polling alembic_version until it reaches %s (timeout 300s)",
                        expected_head,
                    )
                    deadline = time.monotonic() + 300
                    current = None
                    while time.monotonic() < deadline:
                        current = db.session.execute(
                            text("SELECT version_num FROM ai.alembic_version")
                        ).scalar()
                        db.session.commit()
                        if current == expected_head:
                            logger.info(
                                "Migration completed by another worker (head=%s)",
                                current,
                            )
                            break
                        time.sleep(2)
                    else:
                        logger.error(
                            "Timed out after 300s waiting for migration; "
                            "current head=%s, expected=%s. Failing startup so "
                            "Kubernetes restarts and we retry.",
                            current,
                            expected_head,
                        )
                        raise SystemExit(1)
        except SystemExit:
            raise
        except Exception as e:
            logger.error(f"Database migration failed: {e}")
            db.session.rollback()
            raise SystemExit(1)

    return app


def main():
    """Run the development server."""
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
