# powabase-ai

The AI backend service of the [Powabase](https://github.com/powabase-ai) OSS
edition — the per-project service for AI features: sources, knowledge bases,
agents, workflows, and background task processing. It is published as the
container image `ghcr.io/powabase-ai/powabase-ai` and builds on the
[`powabase-agentic`](https://pypi.org/project/powabase-agentic/) library
(import module `agentic`).

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

# Run API server
gunicorn -w 4 -b 0.0.0.0:5000 agentic_project_service.main:app

# Run Celery worker (in separate terminal)
celery -A agentic_project_service.celery worker --loglevel=info
```

## Docker

```bash
docker build -t agentic-project-service:latest .

# Run API
docker run -p 5000:5000 --env-file .env agentic-project-service:latest

# Run Worker
docker run --env-file .env agentic-project-service:latest celery -A agentic_project_service.celery worker --loglevel=info
```

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/sources` - List sources
- `POST /api/sources` - Create source
- `POST /api/sources/<id>/extract` - Trigger extraction
- `GET /api/knowledge-bases` - List knowledge bases
- `POST /api/knowledge-bases` - Create knowledge base
- `POST /api/knowledge-bases/<id>/index` - Trigger indexing
- `POST /api/knowledge-bases/<id>/search` - Semantic search
- `GET /api/agents` - List agents
- `POST /api/agents` - Create agent
- `POST /api/agents/<id>/run` - Execute agent
