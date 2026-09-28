# powabase-ai

The AI backend service of the [Powabase](https://github.com/powabase-ai) OSS
edition — the per-project service for AI features: sources, knowledge bases,
agents, workflows, and background task processing. It is published as the
container image `ghcr.io/powabase-ai/powabase-ai` and builds on the
[`powabase-agentic`](https://pypi.org/project/powabase-agentic/) library
(import module `agentic`).

> **Running the self-hosted stack?** You don't deploy this service on its own.
> The [Powabase stack](https://github.com/powabase-ai/powabase) pulls this image
> automatically alongside Postgres, Auth, Storage, and Studio — see its
> [architecture overview](https://github.com/powabase-ai/powabase#architecture)
> for how the pieces fit. This repo is for **developing the backend service itself**.

## Overview

This service runs within a project's Supabase stack and handles:
- Source ingestion and management
- Knowledge base creation and indexing
- Vector embeddings and semantic search
- Agent configuration and execution
- Background task processing via Celery workers

## Components

### Flask API Server
Handles HTTP requests for sources, knowledge bases, and agents.

### Celery Worker
Processes background tasks:
- Source extraction (PDF, web scraping, etc.)
- Document chunking and embedding
- Knowledge base indexing

## Running Locally

```bash
# Install dependencies
pip install -e .

# Set environment variables
export DATABASE_URL=postgresql://...
export REDIS_URL=redis://localhost:6379/0
export OPENAI_API_KEY=sk-...
export JWT_SECRET=your-jwt-secret
export SERVICE_ROLE_KEY=your-service-role-key  # required: see Authentication

# Run API server
gunicorn -w 4 -b 0.0.0.0:5000 agentic_project_service.main:app

# Run Celery worker (in separate terminal)
celery -A agentic_project_service.celery worker --loglevel=info
```

## Keyword search with pg_search (optional)

When the project's Postgres provides the ParadeDB
[`pg_search`](https://github.com/paradedb/paradedb) extension (preloaded via
`shared_preload_libraries`, with `vector` available), the service creates it at
start-up and serves hybrid and full-text keyword search from a `USING bm25`
index per knowledge base. Without it, keyword search uses the bm25s file index
or the tsvector fallback.

**On Postgres 15 and 16, use a pg_search build that contains
[paradedb/paradedb#6211](https://github.com/paradedb/paradedb/pull/6211).**
The service builds these indexes with `CREATE INDEX CONCURRENTLY` while the
knowledge base keeps being written to, and without that fix pg_search fails
such a build -- stock 0.25.9 can crash the Postgres server doing it. No 0.25.x
release contains the fix; Postgres 17 and 18 are not affected.
`ci/pg_search/Dockerfile` builds 0.25.9 with the fix, and is what the
`tests/pg_search` suite runs against in CI:

```bash
docker build -t pg-search-ci:0.25.9-paradedb-6211 ci/pg_search  # compiles pg_search: minutes to tens of minutes
```

## Docker

The image is **published automatically to `ghcr.io/powabase-ai/powabase-ai`**
(multi-arch, by this repo's `.github/workflows/publish.yml`), and the Powabase
stack pulls it for you — so you normally **don't build or run this container
directly**. To build it locally while developing the service:

```bash
docker build -t powabase-ai:dev .

# Run API
docker run -p 5000:5000 --env-file .env powabase-ai:dev

# Run Worker
docker run --env-file .env powabase-ai:dev celery -A agentic_project_service.celery worker --loglevel=info
```

## Authentication

Apart from inbound webhooks and the internal docs search, which each verify
their own secret, every `/api` route takes a `Bearer` token. There are two kinds
of caller:

- **Service role** — the bearer is exactly `SERVICE_ROLE_KEY`. This is how the
  dashboard and your own backend call the API. Every route is available.
- **End user** — a JWT for a signed-in user of the project (audience
  `authenticated`, signed with `JWT_SECRET`). An end user may only hold
  conversations: create, list, continue and delete *their own* agent and
  orchestration sessions, run agents and orchestrations in them, read their own
  runs, and approve their own paused runs. Every other route answers `403`.

An end user's run uses the knowledge bases configured on the agent. The run-body
fields that point it at other stored data — `knowledge_bases`,
`runtime_knowledge_bases`, `context_handler_id` and by-reference `context_items`
— are service-role only. To let users search particular knowledge bases, run
the agent from your backend with the service role key, and pass `user_id` when
creating the session so the conversation still belongs to that user.

Without `SERVICE_ROLE_KEY` set, no caller is the service role and every
management route answers `403`.

### What an agent's tools can reach

An agent's `database_query`, `database_write`, `storage_read` and
`storage_write` tools act as whoever started the run, never as this service's
own database login:

- **An end user's run** queries Postgres as `authenticated` with that user's JWT
  claims, and calls Storage with their own token. The agent can read and write
  exactly what the user could through the REST API themselves: your grants and
  RLS policies decide.
- **A service-role run** queries Postgres as a role of its own that holds grants
  on exactly the tables configured on the agent's database tools, and bypasses
  RLS on those tables only.

`database_query` accepts one plain `SELECT` that reads only the configured
tables and calls only built-in functions; `set_config`, functions that run a SQL
string, and functions defined in your own schemas are rejected. The tools log in
as `powabase_agent_user` and `powabase_agent_backend`, which the service creates
at startup with its own database password. Neither is a superuser or can bypass
RLS itself.

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/sources` - List sources
- `POST /api/sources` - Create source
- `POST /api/sources/<id>/reextract` - Re-run extraction
- `GET /api/knowledge-bases` - List knowledge bases
- `POST /api/knowledge-bases` - Create knowledge base
- `POST /api/knowledge-bases/<id>/sources` - Attach a source to a KB (triggers indexing)
- `POST /api/knowledge-bases/<id>/search` - Semantic search
- `GET /api/agents` - List agents
- `POST /api/agents` - Create agent
- `POST /api/agents/<id>/run` - Execute agent
