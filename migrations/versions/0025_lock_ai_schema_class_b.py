"""Lock ai schema as Class-B (private/backend) — drop anon/authenticated access.

ai holds project-service's backend state (sources, knowledge bases, agents,
chunks, runs, chat, etc.). By the schema-classification rule in
docs/database-roles-and-scopes.md §4 it should be Class B (private/backend),
like `auth` — reachable only via service_role / project-service, never via a
client's own JWT. It was mistakenly set up as Class A (public/client-facing)
with USING(true)/WITH CHECK(true) policies and anon/authenticated GRANTs —
see docs/database-roles-and-scopes.md §7 for the full decision. This
migration reclassifies it for existing projects, mirroring the same change
already made to ai_schema.sql (new projects) so both converge on the same
final state:

  1. Drop every RLS policy in ai that targets authenticated or anon. This is
     a dynamic pg_policies scan rather than a hardcoded list of the 93 policy
     names in the pre-Class-B ai_schema.sql: a project built through this
     migration chain can have a slightly different policy set than one
     seeded fresh from ai_schema.sql (verified against the C2.2 isolated
     test DB — 82 authenticated-role policies there vs. 93 in a fresh
     ai_schema.sql install), so matching on `pg_policies.roles` is the only
     approach that's actually correct for every project, not just fresh ones.
  2. Revoke the anon/authenticated GRANTs (schema USAGE, table
     SELECT/INSERT/UPDATE/DELETE, sequence USAGE/SELECT, the two ai-schema
     RPC EXECUTE grants). REVOKE is a no-op when a privilege was never held,
     so this is safe regardless of which grants a given project actually has.
  3. Enable RLS (no policy — deny-by-default; service_role bypasses via
     BYPASSRLS) on the two tables that predate the blanket RLS-enable pass:
     ai.message_citations and ai.ai_provider_keys.

Dropping `ai` from `PGRST_DB_SCHEMAS` (removing it from the REST-exposed
schema list) is a per-stack env/Helm-values change, not a DB migration —
handled separately per stack (OSS .env.example, the Docker-mode project
env.template, the K8s project-stack Helm values).

Revision ID: 0025
Revises: 0024
Create Date: 2026-07-17
"""

from alembic import op

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None


def upgrade():
    # 1. Drop every authenticated/anon policy in ai — dynamic (see module
    #    docstring for why a hardcoded name list isn't correct here).
    op.execute("""
        DO $$
        DECLARE
            pol RECORD;
        BEGIN
            FOR pol IN
                SELECT schemaname, tablename, policyname
                FROM pg_policies
                WHERE schemaname = 'ai'
                  AND roles && ARRAY['authenticated', 'anon']::name[]
            LOOP
                EXECUTE format('DROP POLICY IF EXISTS %I ON %I.%I',
                                pol.policyname, pol.schemaname, pol.tablename);
            END LOOP;
        END $$;
    """)

    # 2. Revoke anon/authenticated GRANTs.
    op.execute("""
        REVOKE ALL ON ALL TABLES IN SCHEMA ai FROM authenticated, anon;
        REVOKE ALL ON ALL SEQUENCES IN SCHEMA ai FROM authenticated, anon;
        REVOKE USAGE ON SCHEMA ai FROM authenticated, anon;
        REVOKE EXECUTE ON FUNCTION ai.observability_agent_run_buckets(timestamptz, text)
            FROM authenticated;
        REVOKE EXECUTE ON FUNCTION ai.list_sources_excluding_kb(uuid, text, int, int)
            FROM authenticated;
    """)

    # 3. The two tables that predate the blanket RLS-enable pass. Deny-by-
    #    default (no policy) — service_role bypasses RLS (BYPASSRLS), so it
    #    keeps working unchanged. Idempotent: re-enabling RLS on a table
    #    that already has it enabled is a no-op, not an error.
    op.execute("""
        ALTER TABLE ai.message_citations ENABLE ROW LEVEL SECURITY;
        ALTER TABLE ai.ai_provider_keys ENABLE ROW LEVEL SECURITY;
    """)


def downgrade():
    # Restore the pre-Class-B grants + the blanket USING(true)/WITH CHECK(true)
    # policies verbatim from ai_schema.sql (pre-C2.2), so a `flask db upgrade`
    # right after this downgrade re-locks a fully consistent Class-B state
    # again — not the incremental-migration-chain's slightly different
    # policy set (see the upgrade()'s docstring note on that drift).
    op.execute("""
        ALTER TABLE ai.ai_provider_keys DISABLE ROW LEVEL SECURITY;
        ALTER TABLE ai.message_citations DISABLE ROW LEVEL SECURITY;
    """)

    op.execute("""
        GRANT USAGE ON SCHEMA ai TO authenticated, anon;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA ai TO authenticated;
        GRANT SELECT ON ALL TABLES IN SCHEMA ai TO anon;
        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA ai TO authenticated;
        GRANT EXECUTE ON FUNCTION ai.observability_agent_run_buckets(timestamptz, text)
            TO authenticated;
        GRANT EXECUTE ON FUNCTION ai.list_sources_excluding_kb(uuid, text, int, int)
            TO authenticated;
    """)

    op.execute("""
        CREATE POLICY auth_read_sources ON ai.sources
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_knowledge_bases ON ai.knowledge_bases
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_indexed_sources ON ai.indexed_sources
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_chunks ON ai.chunks
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_page_index_toc ON ai.page_index_toc
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_page_index_nodes ON ai.page_index_nodes
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_full_documents ON ai.full_documents
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_doc2json_documents ON ai.doc2json_documents
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_graph_index_toc ON ai.graph_index_toc
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_graph_index_nodes ON ai.graph_index_nodes
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_agents ON ai.agents
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_agent_sessions ON ai.agent_sessions
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_agent_runs ON ai.agent_runs
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_context_handlers ON ai.context_handlers
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_enrichment_configs ON ai.enrichment_configs
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_workflows ON ai.workflows
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_workflow_blocks ON ai.workflow_blocks
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_workflow_edges ON ai.workflow_edges
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_workflow_executions ON ai.workflow_executions
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_workflow_block_logs ON ai.workflow_block_logs
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_embeddings ON ai.embeddings
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_project_settings ON ai.project_settings
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_copilot_sessions ON ai.copilot_sessions
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_copilot_messages ON ai.copilot_messages
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_tool_call_events ON ai.tool_call_events
            FOR SELECT TO authenticated USING (true);

        CREATE POLICY auth_write_project_settings ON ai.project_settings
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_project_settings ON ai.project_settings
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
        CREATE POLICY auth_delete_project_settings ON ai.project_settings
            FOR DELETE TO authenticated USING (true);

        CREATE POLICY auth_write_copilot_sessions ON ai.copilot_sessions
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_copilot_sessions ON ai.copilot_sessions
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
        CREATE POLICY auth_delete_copilot_sessions ON ai.copilot_sessions
            FOR DELETE TO authenticated USING (true);

        CREATE POLICY auth_write_copilot_messages ON ai.copilot_messages
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_copilot_messages ON ai.copilot_messages
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
        CREATE POLICY auth_delete_copilot_messages ON ai.copilot_messages
            FOR DELETE TO authenticated USING (true);

        CREATE POLICY auth_read_tools ON ai.tools
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_agent_tools ON ai.agent_tools
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_agent_knowledge_bases ON ai.agent_knowledge_bases
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_agent_mcp_servers ON ai.agent_mcp_servers
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_hooks ON ai.hooks
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_orchestrations ON ai.orchestrations
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_orchestration_entities ON ai.orchestration_entities
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_orchestration_sessions ON ai.orchestration_sessions
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_read_orchestration_runs ON ai.orchestration_runs
            FOR SELECT TO authenticated USING (true);

        CREATE POLICY auth_write_sources ON ai.sources
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_sources ON ai.sources
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

        CREATE POLICY auth_write_knowledge_bases ON ai.knowledge_bases
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_knowledge_bases ON ai.knowledge_bases
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
        CREATE POLICY auth_delete_knowledge_bases ON ai.knowledge_bases
            FOR DELETE TO authenticated USING (true);

        CREATE POLICY auth_write_agents ON ai.agents
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_agents ON ai.agents
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
        CREATE POLICY auth_delete_agents ON ai.agents
            FOR DELETE TO authenticated USING (true);

        CREATE POLICY auth_write_workflows ON ai.workflows
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_workflows ON ai.workflows
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
        CREATE POLICY auth_delete_workflows ON ai.workflows
            FOR DELETE TO authenticated USING (true);

        CREATE POLICY auth_write_workflow_blocks ON ai.workflow_blocks
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_write_workflow_edges ON ai.workflow_edges
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_write_workflow_executions ON ai.workflow_executions
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_workflow_executions ON ai.workflow_executions
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

        CREATE POLICY auth_write_workflow_block_logs ON ai.workflow_block_logs
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_workflow_block_logs ON ai.workflow_block_logs
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

        CREATE POLICY auth_delete_indexed_sources ON ai.indexed_sources
            FOR DELETE TO authenticated USING (true);

        CREATE POLICY auth_read_own_sessions ON ai.agent_sessions
            FOR SELECT TO authenticated
            USING (user_id IS NULL OR user_id = auth.uid());

        CREATE POLICY auth_write_sessions ON ai.agent_sessions
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_own_sessions ON ai.agent_sessions
            FOR UPDATE TO authenticated
            USING (user_id IS NULL OR user_id = auth.uid()) WITH CHECK (true);

        CREATE POLICY auth_read_runs ON ai.agent_runs
            FOR SELECT TO authenticated USING (true);
        CREATE POLICY auth_write_runs ON ai.agent_runs
            FOR INSERT TO authenticated WITH CHECK (true);

        CREATE POLICY auth_write_tools ON ai.tools
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_tools ON ai.tools
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
        CREATE POLICY auth_delete_tools ON ai.tools
            FOR DELETE TO authenticated USING (true);

        CREATE POLICY auth_write_agent_tools ON ai.agent_tools
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_delete_agent_tools ON ai.agent_tools
            FOR DELETE TO authenticated USING (true);

        CREATE POLICY auth_write_agent_knowledge_bases ON ai.agent_knowledge_bases
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_delete_agent_knowledge_bases ON ai.agent_knowledge_bases
            FOR DELETE TO authenticated USING (true);

        CREATE POLICY auth_write_agent_mcp_servers ON ai.agent_mcp_servers
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_agent_mcp_servers ON ai.agent_mcp_servers
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
        CREATE POLICY auth_delete_agent_mcp_servers ON ai.agent_mcp_servers
            FOR DELETE TO authenticated USING (true);

        CREATE POLICY auth_write_context_handlers ON ai.context_handlers
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_context_handlers ON ai.context_handlers
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

        CREATE POLICY auth_write_enrichment_configs ON ai.enrichment_configs
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_enrichment_configs ON ai.enrichment_configs
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
        CREATE POLICY auth_delete_enrichment_configs ON ai.enrichment_configs
            FOR DELETE TO authenticated USING (true);

        CREATE POLICY auth_write_hooks ON ai.hooks
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_hooks ON ai.hooks
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
        CREATE POLICY auth_delete_hooks ON ai.hooks
            FOR DELETE TO authenticated USING (true);

        CREATE POLICY auth_write_orchestrations ON ai.orchestrations
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_orchestrations ON ai.orchestrations
            FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
        CREATE POLICY auth_delete_orchestrations ON ai.orchestrations
            FOR DELETE TO authenticated USING (true);

        CREATE POLICY auth_write_orchestration_entities ON ai.orchestration_entities
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_delete_orchestration_entities ON ai.orchestration_entities
            FOR DELETE TO authenticated USING (true);

        CREATE POLICY auth_read_own_orch_sessions ON ai.orchestration_sessions
            FOR SELECT TO authenticated
            USING (user_id IS NULL OR user_id = auth.uid());
        CREATE POLICY auth_write_orch_sessions ON ai.orchestration_sessions
            FOR INSERT TO authenticated WITH CHECK (true);
        CREATE POLICY auth_update_own_orch_sessions ON ai.orchestration_sessions
            FOR UPDATE TO authenticated
            USING (user_id IS NULL OR user_id = auth.uid()) WITH CHECK (true);

        CREATE POLICY auth_write_orchestration_runs ON ai.orchestration_runs
            FOR INSERT TO authenticated WITH CHECK (true);
    """)
