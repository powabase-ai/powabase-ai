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

## Database role permissions

The service expects to own the `ai` schema. Beyond that, one privilege matters
when `DATABASE_URL` connects as a role that is **not a superuser**:

```sql
GRANT pg_read_all_stats TO <the service's role>;   -- or pg_monitor
```

Large knowledge bases get a vector index of their own, built with
`CREATE INDEX CONCURRENTLY` on the shared `ai.embeddings` table, one build at a
time per project. Before each build the service reads `pg_stat_activity` to see
who else holds that table, so it never queues a build behind a running one
(queued index DDL deadlocks a build at its very end). Without `pg_read_all_stats`,
Postgres hides other roles' sessions from that view, and the service cannot tell
a running build from autovacuum.

The consequence is concrete. A long autovacuum of `ai.embeddings` then reads as an
unknown holder of the table. The pending build is deferred through its counted
retries, about half an hour, and then gives up with `PerKbVectorIndexTableWaitExhausted`;
the next indexed source or restart tries again. With the grant, autovacuum is
recognised and ignored, and the build proceeds as it always has. The worker logs a
WARNING naming the grant the first time it meets a holder it cannot see.

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
  dashboard and your own backend call the API. Every route that takes a bearer
  is available.
- **End user** — a JWT for a signed-in user of the project (audience
  `authenticated`, signed with `JWT_SECRET`). It must carry the user's id, a
  UUID, as `sub`, and an expiry as `exp`; a token without either answers `401`.
  Tokens the project's Auth service issues always have both. An end user may
  only hold conversations, and only in *their own* sessions and runs. These 16
  routes are the whole list:

  | Method | Path | What an end user may do |
  |---|---|---|
  | `GET` | `/api/agents/<agent_id>/sessions` | list their own sessions with the agent |
  | `POST` | `/api/agents/<agent_id>/sessions` | create a session, owned by them |
  | `DELETE` | `/api/agents/<agent_id>/sessions/<session_id>` | delete their own session |
  | `POST` | `/api/agents/<agent_id>/run` | run the agent, in a new session or their own |
  | `POST` | `/api/agents/<agent_id>/run/stream` | the same, streamed |
  | `GET` | `/api/agents/runs/<run_id>` | read a run in their own session |
  | `POST` | `/api/agents/runs/<run_id>/approve` | approve or reject their own paused run |
  | `GET` | `/api/sessions/<session_id>` | read their own session |
  | `GET` | `/api/sessions/<session_id>/messages` | read its messages |
  | `GET` | `/api/sessions/<session_id>/runs` | read its runs |
  | `GET` | `/api/sessions/<session_id>/runs/<run_id>/retrieved-context` | read a run's retrieved context |
  | `DELETE` | `/api/sessions/<session_id>` | delete their own session |
  | `GET` | `/api/orchestrations/<orch_id>/sessions` | list their own orchestration sessions |
  | `GET` | `/api/orchestrations/<orch_id>/sessions/<session_id>/messages` | read their own orchestration session's messages |
  | `POST` | `/api/orchestrations/<orch_id>/run/stream` | run the orchestration, in a new session or their own session of it |
  | `GET` | `/api/orchestrations/runs/<run_id>` | read an orchestration run in their own session |

  There is no end-user route to create or delete an orchestration session: a
  run creates one. No route deletes orchestration sessions at all, for any
  caller; they stay in the database, also after their orchestration is deleted.
  A session, run or paused run that is someone else's, or that a backend
  created without a `user_id`, answers `404` as if it did not exist.

  A `session_id`, in a run body or a path, must be a string of at most 255
  characters; anything else answers `400`. An end user's run continues a session only if it
  already exists and is their own session of the same agent or orchestration;
  an end user can never name a new session. To start one, they omit
  `session_id` (the run creates a session owned by them and returns its id) or
  call `POST /api/agents/<agent_id>/sessions`. This matters when your backend
  chooses session ids someone could guess, such as one per phone number: if an
  end user could create a session under such an id first, your backend's later
  runs in it would be fed their planted history, and they could read what
  those runs said. The check is made again when the run binds to the session:
  a non-streamed run then answers `404`, and a streamed run ends with the
  event `{"event": "error", "error": "Session not found"}`.

  An end user may approve or reject only the paused runs they started
  themselves. A run your backend started with the service role key has no
  end-user owner, even when it runs in a session created with that user's
  `user_id`, so only the service role key can approve it: relay the user's
  decision from your backend.

Every other route that takes a bearer answers an end user with `403` and
`{"error": "This endpoint requires the project's service role key"}`.

**Never ship `SERVICE_ROLE_KEY` to a browser or a mobile app.** Anyone who holds
it can call every route above as an administrator. Browsers and apps call with
the signed-in user's own JWT.

An end user's run uses the knowledge bases configured on the agent. The run-body
fields that point it at other stored data — `knowledge_bases`,
`runtime_knowledge_bases`, `context_handler_id` and by-reference `context_items`
— are service-role only. An end user who sends one gets `403` with a body naming
the fields, for example:

```json
{"error": "knowledge_bases may only be set with the project's service role key; an end user's run uses the knowledge bases configured on the agent"}
```

To let users search particular knowledge bases, run the agent from your backend
with the service role key, and pass `user_id` when creating the session so the
conversation still belongs to that user.

> **When your backend runs an agent for a user, its data tools do not act as
> that user.** A run made with the service role key is a service-role run, even
> in a session created with `user_id`:
>
> - the agent's `database_query` and `database_write` tools act as the agent's
>   own database role, with grants on the tables configured on its tools and
>   RLS bypassed on those tables, not as the session's user;
> - its `storage_read` and `storage_write` tools call Storage with the service
>   role key, so they reach every bucket except the internal `sources` bucket,
>   whatever your Storage policies say about the session's user.
>
> Configure such an agent only with tables and buckets that every user it runs
> for may see in full, or run agents that carry data tools with the user's own
> JWT instead.

Without `SERVICE_ROLE_KEY` set, no caller is the service role. A request that
bears the service role key is then decoded as an end-user token and answers
`401` (it has no `authenticated` audience); the management routes are
unreachable.

### What an agent's tools can reach

An agent's `database_query`, `database_write`, `storage_read` and
`storage_write` tools act as whoever started the run, never as this service's
own database login:

- **An end user's run** queries Postgres as the `authenticated` role with that
  user's JWT claims, limited to the tables configured on the agent's database
  tools, so your grants and RLS policies for that user apply to those tables.
  It calls Storage with the user's own token.
- **A service-role run** logs in as the agent's own database login, which holds
  grants on exactly the tables configured on the agent's database tools and
  bypasses RLS on those tables only. It is a member of no other role, so one
  agent cannot take on another's grants. Its storage tools use the service role
  key (see the note above).

Only plain tables, partitioned tables and views created `WITH
(security_invoker = true)` may be configured; assigning or updating a database
tool whose configuration names a view without `security_invoker`, a
materialized view or a foreign table answers `400`. A table that does not exist
yet may be configured.

`database_query` accepts one `SELECT` over the configured tables that calls only
an allowlist of built-in functions and operators and casts only to built-in
types. Functions that change or read settings or run a SQL string, custom
functions, operators written in SQL or PL/pgSQL in your schemas, and casts to
domains are rejected. Function names are matched as Postgres resolves them:
a quoted `"Sum"` is not `sum`. The allowlists are fixed in the code and cannot
be configured. An operator an installed extension defines is accepted when its
symbol is on the list (pgvector's distances `<->`, `<=>`, `<#>` and `<+>`, and
symbols such as `&&`, `@>` or `->` that other extensions reuse) and the
extension lives in one of the agent's configured schemas; an extension
installed in a schema the agent is not configured with, such as Supabase's
default `extensions` schema, is not visible to its queries. Functions that
extensions add are rejected. A rejected query
returns an error naming what was refused, in one of these shapes:

```text
Function <name> is not allowed: only a fixed set of built-in functions may be called
Operator <operator> is not allowed
```

Every tool transaction has a 30-second statement timeout and a 5-second lock
timeout. That applies to service-role runs as well as end users' runs, whatever
timeout the `authenticated` role has elsewhere.

The end-user login is `powabase_agent_user`. Each agent that has a database
tool gets its own login, `powabase_agent_<agent id without dashes>`; an agent
without database tools gets none. The service creates them with its own
database password, the end-user login at startup and an agent's login when its
database tools are configured or it is first run with them; none is a
superuser. Deleting an agent drops its login, and startup removes any login
whose agent is gone.

Postgres also runs some functions implicitly, such as a domain's `CHECK` when a
value is written or the equality operator of a column's own type when a query
groups or joins on it. Configure agents only with tables whose column types you
trust.

The storage tools accept bucket names matching `^[A-Za-z0-9_-]+$`, which is
stricter than Storage itself (a bucket whose name has a `.` or a space cannot be
used from a tool). Each `/`-separated part of a path must be a name: not empty,
`.` or `..`, and without `%`, `\`, `?`, `#` or control characters. The internal
`sources` bucket, which holds the original files of ingested sources, is never
reachable from a tool.

## API Endpoints

- `GET /health` - Health check
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
