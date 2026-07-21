"""Promote usage/tool_calls JSONB to typed columns.

Why: the observability dashboard needs to aggregate tokens by
model / agent / type / time, and tool-call stats by tool / agent. JSONB
aggregation (SUM((usage->>'prompt_tokens')::int)) works but can't be
indexed efficiently and requires parsing on every query. Promote the
hot fields to typed INT/VARCHAR columns on the existing run tables and
move per-tool-call detail into a dedicated `ai.tool_call_events` table.

Writers cut over at the same time (see services/session.py etc.) so
there is exactly one source of truth per metric.

ai.agent_runs:           add model, agent_id, 5 token cols, 3 tool-call
                         summary cols; backfill from `usage` / `tool_calls`
                         JSONB + joins to agent_sessions/agents; drop
                         `usage` and `tool_calls` JSONB.
ai.orchestration_runs:   add model + 5 token cols + 3 tool-call summary
                         cols; backfill from `usage`; drop `usage` JSONB.
ai.workflow_block_logs:  add model + 3 token cols for block_type='agent'
                         rows; backfill from `output` JSONB; keep `output`
                         (carries the block's actual return value).
ai.tool_call_events:     new table — one row per tool invocation; backfill
                         by unnesting agent_runs.tool_calls JSONB before we
                         drop it.

================================================================
Production safety — REQUIRES MAINTENANCE WINDOW FOR LARGE TABLES
================================================================

DO NOT RUN THIS MIGRATION DURING NORMAL TRAFFIC ON ANY PROJECT WHERE
``ai.agent_runs`` HAS MORE THAN ~500k ROWS. The migration runs several
unbatched UPDATEs / INSERTs in single transactions; they hold per-row
write locks on ``ai.agent_runs`` (and on ``ai.orchestration_runs`` /
``ai.workflow_block_logs`` to a smaller degree) for the duration.
Approximate impact, dominated by the agent_runs row count:

  • ~100k rows  → typically <1s
  • ~1M rows    → several seconds of row-write blocking
  • ~10M rows   → tens of seconds; concurrent writers will queue

The tool_call_events INSERT scales with total fan-out (rows ×
calls/row), not just the row count — so a 500k-row table averaging
4 tool calls per run produces ~2M INSERTs in one shot. Multiply the
above estimates by avg tool_calls/run for that mutation.

Estimate row counts up front:

    SELECT count(*) FROM ai.agent_runs;
    SELECT count(*) FROM ai.orchestration_runs;
    SELECT avg(jsonb_array_length(tool_calls))::int
      FROM ai.agent_runs WHERE jsonb_typeof(tool_calls) = 'array';

If any count is approaching the threshold, schedule a low-traffic
window before applying. Mutations in execution order:

  1. ``ALTER TABLE … ADD COLUMN`` (agent_runs) — fast, metadata-only.
  2. UPDATE agent_runs SET prompt/completion/reasoning/cached/total_tokens
     FROM the existing usage JSONB — locks every row WHERE usage IS NOT NULL.
  3. UPDATE agent_runs SET agent_id, model FROM agent_sessions JOIN
     agents — single transaction, locks every joined row.
  4. UPDATE agent_runs SET tool_call_count/error_count/duration_ms_total
     using LATERAL jsonb_array_elements over the tool_calls JSONB —
     single transaction, locks every row with a non-empty tool_calls.
  5. UPDATE agent_runs SET tool_call_count=0, ... WHERE tool_call_count
     IS NULL — locks every row that didn't have a tool_calls array
     (potentially most of the table). Cheap per-row but row count = N.
  6. INSERT INTO tool_call_events SELECT … FROM agent_runs JOIN
     LATERAL jsonb_array_elements(tool_calls) — bulk INSERT scaled
     by the total fan-out of tool_call elements (rows × calls/row).
  7. ``ALTER TABLE`` + UPDATE on orchestration_runs and workflow_block_logs
     — same shape as steps 2 and 4 but on those tables.
  8. CREATE INDEX CONCURRENTLY (autocommit blocks below) — does NOT
     take ACCESS EXCLUSIVE; readers and writers stay live.

If a future project crosses ~10M rows, split the unbatched UPDATE / INSERT
mutations (steps 2-7 above) into chunks of ~100k rows by primary key and
commit per chunk:

    UPDATE ai.agent_runs ar
    SET … FROM ai.agent_sessions s JOIN ai.agents a …
    WHERE ar.session_id = s.id
      AND ar.agent_id IS NULL
      AND ar.id BETWEEN :lo AND :hi;
    COMMIT;

Walk :lo/:hi from min(id) to max(id) in 100k-row strides. Same pattern
applies to the tool_call_count UPDATE and the tool_call_events INSERT.

CREATE INDEX statements already use CONCURRENTLY in autocommit blocks
so they never block the table — a previous version of this file used
plain CREATE INDEX inside the migration transaction, which would have
taken ACCESS EXCLUSIVE on agent_runs while the index built.

Revision ID: 0019
Revises: 0018
Create Date: 2026-04-17

TODO(future): convert backfills to chunked transactions when a project
hits ~10M agent_run rows. Tracked in known-issues backlog.
"""

from alembic import op
from sqlalchemy import text as sa_text


# revision identifiers, used by Alembic.
revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade():
    # ------------------------------------------------------------------
    # ai.agent_runs — add typed columns
    # ------------------------------------------------------------------
    op.execute("""
        ALTER TABLE ai.agent_runs
            ADD COLUMN IF NOT EXISTS agent_id UUID,
            ADD COLUMN IF NOT EXISTS model VARCHAR(128),
            ADD COLUMN IF NOT EXISTS prompt_tokens INT,
            ADD COLUMN IF NOT EXISTS completion_tokens INT,
            ADD COLUMN IF NOT EXISTS reasoning_tokens INT,
            ADD COLUMN IF NOT EXISTS cached_tokens INT,
            ADD COLUMN IF NOT EXISTS total_tokens INT,
            ADD COLUMN IF NOT EXISTS tool_call_count INT,
            ADD COLUMN IF NOT EXISTS tool_call_error_count INT,
            ADD COLUMN IF NOT EXISTS tool_call_duration_ms_total INT
    """)

    # Backfill token cols from existing usage JSONB. Guarded on column
    # existence — new project provisioning runs ai_schema.sql first (which
    # no longer includes the JSONB cols), then this migration; the backfill
    # is a no-op there but the upgrade still has to run cleanly.
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'ai' AND table_name = 'agent_runs'
                  AND column_name = 'usage'
            ) THEN
                UPDATE ai.agent_runs
                SET prompt_tokens = NULLIF(usage->>'prompt_tokens', '')::int,
                    completion_tokens = NULLIF(usage->>'completion_tokens', '')::int,
                    reasoning_tokens = NULLIF(usage->>'reasoning_tokens', '')::int,
                    cached_tokens = NULLIF(usage->>'cached_tokens', '')::int,
                    total_tokens = NULLIF(usage->>'total_tokens', '')::int
                WHERE usage IS NOT NULL;
            END IF;
        END $$
    """)

    # Backfill denormalized agent_id + model via the session → agent chain.
    #
    # ⚠ UNBATCHED — single transaction, locks every joined agent_runs row
    # for the duration. See the module docstring "Production safety"
    # section before running on a project with > ~500k agent_runs.
    #
    # Note: orchestration delegate agent_runs have session_id NULL, so this
    # WHERE clause never matches them — those rows stay agent_id NULL.
    # NEW delegate runs get agent_id directly from the on_run_complete
    # payload (see routes/orchestrations.py:_persist_delegate_run); the
    # historical rows can't be recovered after the fact because
    # ai.agent_runs doesn't keep a reference to which orchestration entity
    # ran. The per-agent dashboard breakdown will under-count delegated
    # historical runs.
    op.execute("""
        UPDATE ai.agent_runs ar
        SET agent_id = s.agent_id,
            model = a.model
        FROM ai.agent_sessions s
        LEFT JOIN ai.agents a ON a.id = s.agent_id
        WHERE ar.session_id = s.id
          AND ar.agent_id IS NULL
    """)

    # Backfill tool-call summary cols from the tool_calls JSONB array.
    #
    # ⚠ UNBATCHED — single transaction, locks every agent_runs row whose
    # tool_calls is a non-empty array. LATERAL fan-out over the JSONB
    # array adds CPU cost on top of the row-lock count. See the module
    # docstring "Production safety" section before running on a project
    # with > ~500k agent_runs.
    #
    # Note: jsonb_array_elements errors on non-array input, so we filter in
    # a subquery first (WHERE on a JOIN doesn't reliably gate LATERAL eval).
    # Guarded on column existence (see token-backfill comment above).
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'ai' AND table_name = 'agent_runs'
                  AND column_name = 'tool_calls'
            ) THEN
                UPDATE ai.agent_runs ar
                SET tool_call_count = sub.cnt,
                    tool_call_error_count = sub.err,
                    tool_call_duration_ms_total = sub.dur
                FROM (
                    SELECT
                        r.id,
                        COUNT(*) AS cnt,
                        COUNT(*) FILTER (
                            WHERE LOWER(COALESCE(c->>'result', '')) LIKE 'error%'
                        ) AS err,
                        COALESCE(SUM(NULLIF(c->>'duration_ms', '')::int), 0) AS dur
                    FROM (
                        SELECT id, tool_calls FROM ai.agent_runs
                        WHERE tool_calls IS NOT NULL AND jsonb_typeof(tool_calls) = 'array'
                    ) r,
                    LATERAL jsonb_array_elements(r.tool_calls) AS c
                    GROUP BY r.id
                ) sub
                WHERE ar.id = sub.id;
            END IF;
        END $$
    """)

    # Zero out tool-call summary for rows that had no tool_calls array.
    #
    # ⚠ UNBATCHED — single transaction, locks every row whose tool_calls
    # was absent or non-array (potentially most of the table). The per-row
    # work is trivial but the row count is the full agent_runs population
    # minus the rows updated in the previous step. See the module
    # docstring "Production safety" section.
    op.execute("""
        UPDATE ai.agent_runs
        SET tool_call_count = 0,
            tool_call_error_count = 0,
            tool_call_duration_ms_total = 0
        WHERE tool_call_count IS NULL
    """)

    # Indexes for the dashboard query paths.
    # CONCURRENTLY so the index build doesn't take ACCESS EXCLUSIVE on
    # agent_runs (which would block readers/writers for the build duration
    # — minutes on multi-million-row tables). Each statement runs in its
    # own transaction via the autocommit_block.
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            "idx_agent_runs_model_created ON ai.agent_runs (model, created_at DESC)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            "idx_agent_runs_agent_created ON ai.agent_runs (agent_id, created_at DESC)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            "idx_agent_runs_status_created ON ai.agent_runs (status, created_at DESC)"
        )

    # ------------------------------------------------------------------
    # ai.tool_call_events — new per-invocation table
    # ------------------------------------------------------------------
    # Created BEFORE we drop agent_runs.tool_calls so we can backfill from it.
    # FK to ai.workflow_executions is added in a guarded follow-up step
    # because that table is defined in ai_schema.sql and may not be present
    # in every environment (same defensive pattern as 0016_fk).
    # `arguments` and `result` keep the full tool-call payload (JSONB) so
    # multimodal tool responses (image_ref blocks etc.) round-trip cleanly
    # through the API's tool_calls reader. `*_preview` fields are short text
    # truncations meant for grids and dashboard drill-downs.
    op.execute("""
        CREATE TABLE IF NOT EXISTS ai.tool_call_events (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_run_id UUID REFERENCES ai.agent_runs(id) ON DELETE CASCADE,
            orchestration_run_id UUID REFERENCES ai.orchestration_runs(id) ON DELETE CASCADE,
            workflow_execution_id UUID,
            agent_id UUID,
            model VARCHAR(128),
            tool_name VARCHAR(255) NOT NULL,
            status VARCHAR(16) NOT NULL CHECK (status IN ('success', 'error')),
            duration_ms INTEGER,
            arguments JSONB,
            result JSONB,
            arguments_preview TEXT,
            result_preview TEXT,
            error TEXT,
            step INTEGER,
            occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'ai' AND table_name = 'workflow_executions'
            ) AND NOT EXISTS (
                SELECT 1 FROM information_schema.table_constraints
                WHERE constraint_name = 'fk_tool_call_events_wf_exec'
            ) THEN
                ALTER TABLE ai.tool_call_events
                    ADD CONSTRAINT fk_tool_call_events_wf_exec
                    FOREIGN KEY (workflow_execution_id)
                    REFERENCES ai.workflow_executions(id) ON DELETE CASCADE;
            END IF;
        END $$
    """)
    # Index the new tool_call_events table. The table was just created
    # above so it's empty — CONCURRENTLY isn't strictly required, but
    # using it keeps the pattern consistent and is defensive against the
    # backfill INSERT below populating it before the index is built.
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            "idx_tool_call_events_tool_occurred "
            "ON ai.tool_call_events (tool_name, occurred_at DESC)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            "idx_tool_call_events_agent_occurred "
            "ON ai.tool_call_events (agent_id, occurred_at DESC)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            "idx_tool_call_events_run ON ai.tool_call_events (agent_run_id)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            "idx_tool_call_events_orch_run "
            "ON ai.tool_call_events (orchestration_run_id)"
        )

    # Backfill: one row per element of agent_runs.tool_calls JSONB.
    #
    # ⚠ UNBATCHED — single transaction; bulk INSERT scaled by the total
    # fan-out (rows × calls/row), not just the row count. A project with
    # 1M agent_runs averaging 4 tool calls each writes 4M tool_call_events
    # rows in one shot. See the module docstring "Production safety"
    # section before running on a project with > ~500k agent_runs.
    #
    # `arguments` and `result` keys may hold non-string JSON (objects/lists);
    # `(c->'key')::text` renders any JSONB to text including its JSON quoting.
    # Wrapped in parens because `::` binds tighter than `->`, so without
    # parens `c->'arguments'::text` would parse as `c->('arguments'::text)`.
    # Errors are heuristically detected by an 'Error...' prefix in the result
    # string (that's how the ReAct loop records them today). Guarded on
    # column existence — new project provisioning has no tool_calls JSONB.
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'ai' AND table_name = 'agent_runs'
                  AND column_name = 'tool_calls'
            ) THEN
                INSERT INTO ai.tool_call_events (
                    agent_run_id, agent_id, model, tool_name, status,
                    duration_ms, arguments, result, arguments_preview, result_preview,
                    error, step, occurred_at
                )
                SELECT
                    r.id AS agent_run_id,
                    r.agent_id,
                    r.model,
                    COALESCE(c->>'tool_name', 'unknown') AS tool_name,
                    CASE
                        WHEN LOWER(COALESCE(c->>'result', '')) LIKE 'error%' THEN 'error'
                        ELSE 'success'
                    END AS status,
                    NULLIF(c->>'duration_ms', '')::int AS duration_ms,
                    c->'arguments' AS arguments,
                    c->'result' AS result,
                    LEFT((c->'arguments')::text, 500) AS arguments_preview,
                    LEFT((c->'result')::text, 500) AS result_preview,
                    CASE
                        WHEN LOWER(COALESCE(c->>'result', '')) LIKE 'error%'
                            THEN LEFT(c->>'result', 1000)
                        ELSE NULL
                    END AS error,
                    NULLIF(c->>'step', '')::int AS step,
                    COALESCE(r.completed_at, r.started_at, r.created_at) AS occurred_at
                FROM (
                    SELECT id, agent_id, model, tool_calls,
                           completed_at, started_at, created_at
                    FROM ai.agent_runs
                    WHERE tool_calls IS NOT NULL AND jsonb_typeof(tool_calls) = 'array'
                ) r,
                LATERAL jsonb_array_elements(r.tool_calls) AS c;
            END IF;
        END $$
    """)

    # ------------------------------------------------------------------
    # ai.orchestration_runs — typed token + tool-call summary cols
    # ------------------------------------------------------------------
    op.execute("""
        ALTER TABLE ai.orchestration_runs
            ADD COLUMN IF NOT EXISTS model VARCHAR(128),
            ADD COLUMN IF NOT EXISTS prompt_tokens INT,
            ADD COLUMN IF NOT EXISTS completion_tokens INT,
            ADD COLUMN IF NOT EXISTS reasoning_tokens INT,
            ADD COLUMN IF NOT EXISTS cached_tokens INT,
            ADD COLUMN IF NOT EXISTS total_tokens INT,
            ADD COLUMN IF NOT EXISTS tool_call_count INT,
            ADD COLUMN IF NOT EXISTS tool_call_error_count INT,
            ADD COLUMN IF NOT EXISTS tool_call_duration_ms_total INT
    """)

    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'ai' AND table_name = 'orchestration_runs'
                  AND column_name = 'usage'
            ) THEN
                UPDATE ai.orchestration_runs
                SET prompt_tokens = NULLIF(usage->>'prompt_tokens', '')::int,
                    completion_tokens = NULLIF(usage->>'completion_tokens', '')::int,
                    reasoning_tokens = NULLIF(usage->>'reasoning_tokens', '')::int,
                    cached_tokens = NULLIF(usage->>'cached_tokens', '')::int,
                    total_tokens = NULLIF(usage->>'total_tokens', '')::int,
                    model = usage->>'model'
                WHERE usage IS NOT NULL;
            END IF;
        END $$
    """)

    # Tool-call summary for orchestration_runs is computed by rolling up the
    # tool_call_events rows emitted by its child agent_runs. We intentionally
    # leave orchestration_runs.tool_call_* at NULL for pre-existing runs —
    # the dashboard derives per-orchestration tool stats by joining through
    # agent_runs.parent_orchestration_run_id. Future writers (orchestration.py)
    # will populate these summary cols directly.

    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            "idx_orch_runs_model_created "
            "ON ai.orchestration_runs (model, created_at DESC)"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            "idx_orch_runs_status_created "
            "ON ai.orchestration_runs (status, created_at DESC)"
        )

    # ------------------------------------------------------------------
    # ai.workflow_block_logs — typed model + token cols
    # ------------------------------------------------------------------
    # Guarded — workflow_block_logs is defined in ai_schema.sql and may not be
    # present in older environments (matches 0016_fk's defensive pattern).
    # ALTER + UPDATE inside a DO block (the table-existence guard is
    # required because workflow_block_logs lives in ai_schema.sql and may
    # not be present in older environments). Index creation is split out
    # below so we can use CONCURRENTLY.
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'ai' AND table_name = 'workflow_block_logs'
            ) THEN
                ALTER TABLE ai.workflow_block_logs
                    ADD COLUMN IF NOT EXISTS model VARCHAR(128),
                    ADD COLUMN IF NOT EXISTS prompt_tokens INT,
                    ADD COLUMN IF NOT EXISTS completion_tokens INT,
                    ADD COLUMN IF NOT EXISTS reasoning_tokens INT,
                    ADD COLUMN IF NOT EXISTS total_tokens INT;

                -- Only agent blocks emit LLM usage.
                UPDATE ai.workflow_block_logs
                SET model = output->>'model',
                    prompt_tokens = NULLIF(output->'usage'->>'prompt_tokens', '')::int,
                    completion_tokens = NULLIF(output->'usage'->>'completion_tokens', '')::int,
                    reasoning_tokens = NULLIF(output->'usage'->>'reasoning_tokens', '')::int,
                    total_tokens = NULLIF(output->'usage'->>'total_tokens', '')::int
                WHERE block_type = 'agent'
                  AND output IS NOT NULL;
            END IF;
        END $$
    """)

    # Index workflow_block_logs CONCURRENTLY (autocommit) so the build
    # doesn't block the table. CREATE INDEX CONCURRENTLY can't live
    # inside a DO block, so the CONCURRENTLY versions are no-ops if the
    # table is absent — Postgres raises "relation does not exist", which
    # would abort the migration. Guard with a separate existence check.
    bind = op.get_bind()
    has_wfb = bind.execute(
        sa_text(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'ai' AND table_name = 'workflow_block_logs'"
        )
    ).first()
    if has_wfb:
        with op.get_context().autocommit_block():
            op.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_wf_block_logs_type_created "
                "ON ai.workflow_block_logs (block_type, created_at DESC)"
            )
            op.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_wf_block_logs_model_created "
                "ON ai.workflow_block_logs (model, created_at DESC)"
            )

    # ------------------------------------------------------------------
    # Drop legacy JSONB columns now that typed cols are authoritative.
    # ------------------------------------------------------------------
    op.execute("ALTER TABLE ai.agent_runs DROP COLUMN IF EXISTS usage")
    op.execute("ALTER TABLE ai.agent_runs DROP COLUMN IF EXISTS tool_calls")
    op.execute("ALTER TABLE ai.orchestration_runs DROP COLUMN IF EXISTS usage")

    # ------------------------------------------------------------------
    # ai.observability_agent_run_buckets — server-side rollup for the
    # control-plane observability dashboard.
    #
    # Replaces the previous "fetch up to 20k row-dicts via PostgREST and
    # aggregate in Python" path. Callers go through PostgREST RPC at
    # /rest/v1/rpc/observability_agent_run_buckets with body
    # {"since": "<iso>", "bucket_trunc": "minute"|"hour"|"day"}.
    #
    # Returns one row per non-empty (bucket-truncated) created_at window.
    # The returned column set covers everything _rollup_project needs:
    # bucket key, run count, failed count, token sum, last activity in
    # the bucket. Total run count, total tokens, and overall last activity
    # are computed in Python by summing/maxing across the buckets — the
    # row count is bounded by ``range / bucket_size`` (e.g. 30d/day = 30
    # rows max), so memory stays flat regardless of activity volume.
    #
    # SECURITY DEFINER + restricted search_path so the function runs with
    # ai-schema permissions when called by anon/authenticated roles via
    # PostgREST. STABLE so PostgREST allows it to be called as GET when
    # convenient.
    # ------------------------------------------------------------------
    op.execute("""
        CREATE OR REPLACE FUNCTION ai.observability_agent_run_buckets(
            since timestamptz,
            bucket_trunc text
        )
        RETURNS TABLE (
            bucket timestamptz,
            total_runs bigint,
            failed_runs bigint,
            total_tokens bigint,
            last_activity_at timestamptz
        )
        LANGUAGE sql
        STABLE
        SECURITY DEFINER
        SET search_path = ai, pg_temp
        AS $$
            -- Reject unknown bucket_trunc up-front. date_trunc would raise
            -- a confusing error otherwise; an empty result is friendlier.
            SELECT
                date_trunc(bucket_trunc, ar.created_at) AS bucket,
                COUNT(*)::bigint AS total_runs,
                COUNT(*) FILTER (WHERE ar.status = 'failed')::bigint AS failed_runs,
                COALESCE(SUM(ar.total_tokens), 0)::bigint AS total_tokens,
                MAX(ar.created_at) AS last_activity_at
            FROM ai.agent_runs ar
            WHERE ar.created_at >= since
              AND bucket_trunc IN ('minute', 'hour', 'day')
            GROUP BY date_trunc(bucket_trunc, ar.created_at)
            ORDER BY bucket;
        $$
    """)
    # Grant execute to authenticated callers only. The control-plane proxy
    # always presents service_role; tests / direct PostgREST calls may use
    # authenticated. Anon is intentionally excluded — even though the
    # function only returns aggregates, exposing run-volume telemetry to
    # unauthenticated callers leaks tenant activity by design.
    op.execute(
        "GRANT EXECUTE ON FUNCTION ai.observability_agent_run_buckets(timestamptz, text) "
        "TO authenticated, service_role"
    )


def downgrade():
    # Best-effort downgrade — restores JSONB columns but does NOT re-populate
    # them from typed cols (dashboards would be blind during the window
    # anyway; re-upgrading re-backfills). This is the conventional Alembic
    # pattern for a destructive forward migration.
    op.execute("DROP FUNCTION IF EXISTS ai.observability_agent_run_buckets(timestamptz, text)")
    op.execute("""
        ALTER TABLE ai.agent_runs
            ADD COLUMN IF NOT EXISTS usage JSONB,
            ADD COLUMN IF NOT EXISTS tool_calls JSONB
    """)
    op.execute("""
        ALTER TABLE ai.orchestration_runs
            ADD COLUMN IF NOT EXISTS usage JSONB
    """)

    # Drop the typed cols we added.
    for col in (
        "agent_id",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "total_tokens",
        "tool_call_count",
        "tool_call_error_count",
        "tool_call_duration_ms_total",
    ):
        op.execute(f"ALTER TABLE ai.agent_runs DROP COLUMN IF EXISTS {col}")
    for col in (
        "model",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "total_tokens",
        "tool_call_count",
        "tool_call_error_count",
        "tool_call_duration_ms_total",
    ):
        op.execute(f"ALTER TABLE ai.orchestration_runs DROP COLUMN IF EXISTS {col}")
    for col in (
        "model",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
    ):
        op.execute(f"ALTER TABLE ai.workflow_block_logs DROP COLUMN IF EXISTS {col}")

    op.execute("DROP INDEX IF EXISTS ai.idx_agent_runs_model_created")
    op.execute("DROP INDEX IF EXISTS ai.idx_agent_runs_agent_created")
    op.execute("DROP INDEX IF EXISTS ai.idx_agent_runs_status_created")
    op.execute("DROP INDEX IF EXISTS ai.idx_orch_runs_model_created")
    op.execute("DROP INDEX IF EXISTS ai.idx_orch_runs_status_created")
    op.execute("DROP INDEX IF EXISTS ai.idx_wf_block_logs_type_created")
    op.execute("DROP INDEX IF EXISTS ai.idx_wf_block_logs_model_created")

    op.execute("DROP TABLE IF EXISTS ai.tool_call_events CASCADE")
