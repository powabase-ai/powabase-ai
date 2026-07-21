"""
Tenant-level ORM models for per-project data.

These models represent tables in the 'ai' schema. They mirror the SQL
definitions in templates/supabase-project/volumes/db/ai_schema.sql and
enable Alembic to auto-detect schema changes.

Existing raw SQL queries (db.session.execute(text(...))) continue to work
unchanged alongside these ORM models.
"""

import uuid
from datetime import datetime
from enum import Enum

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, ForeignKey, Index, UniqueConstraint
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..db import db

from agentic.knowledge.model_config import (
    AGENT_DEFAULT_MODEL,
    CHUNK_EMBED_DEFAULT_CHUNK_SIZE,
    CHUNK_EMBED_DEFAULT_OVERLAP,
    DEFAULT_MAX_CONTEXT_TOKENS,
)


class ExtractionStatus(str, Enum):
    """Status of source extraction."""

    PENDING = "pending"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    ATTENTION_REQUIRED = "attention_required"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IndexStatus(str, Enum):
    """Status of source indexing into a knowledge base."""

    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRunStatus(str, Enum):
    """Status of an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ContextHandlerStatus(str, Enum):
    """Status of a context handler retrieval operation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EnrichmentStatus(str, Enum):
    """Status of metadata enrichment for a knowledge base."""

    IDLE = "idle"
    ENRICHING = "enriching"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


class Source(db.Model):
    """A source document uploaded to the platform.

    Table: ai.sources
    """

    __tablename__ = "sources"
    __table_args__ = (
        Index(
            "sources_content_hash_uniq",
            "content_hash",
            unique=True,
            postgresql_where=sa_text("content_hash IS NOT NULL"),
        ),
        {"schema": "ai"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    name: Mapped[str | None] = mapped_column(db.String(255), nullable=True)
    file_type: Mapped[str] = mapped_column(db.String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(db.String(1024), nullable=False)
    extraction_status: Mapped[str | None] = mapped_column(
        db.String(50), server_default=sa_text("'pending'")
    )
    derivatives: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", JSONB, server_default=sa_text("'{}'::jsonb")
    )
    auto_metadata: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    error_message: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(db.String(255), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(db.String(64), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class KnowledgeBase(db.Model):
    """A named collection of indexed sources with configuration.

    Table: ai.knowledge_bases
    """

    __tablename__ = "knowledge_bases"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    name: Mapped[str] = mapped_column(db.String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    indexing_config: Mapped[dict | None] = mapped_column(
        JSONB,
        server_default=sa_text(
            f"""'{{"strategy": "chunk_embed", "chunk_size": {CHUNK_EMBED_DEFAULT_CHUNK_SIZE}, "overlap": {CHUNK_EMBED_DEFAULT_OVERLAP}}}'::jsonb"""
        ),
    )
    retrieval_config: Mapped[dict | None] = mapped_column(
        JSONB,
        server_default=sa_text("""'{"method": "hybrid", "top_k": 5}'::jsonb"""),
    )
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class IndexedSource(db.Model):
    """Represents a Source indexed into a KnowledgeBase.

    Table: ai.indexed_sources
    """

    __tablename__ = "indexed_sources"
    __table_args__ = (
        UniqueConstraint(
            "knowledge_base_id", "source_id", name="indexed_sources_knowledge_base_id_source_id_key"
        ),
        {"schema": "ai"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    knowledge_base_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.knowledge_bases.id", ondelete="CASCADE"),
        nullable=True,
    )
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.sources.id", ondelete="CASCADE"),
        nullable=True,
    )
    index_status: Mapped[str | None] = mapped_column(
        db.String(50), server_default=sa_text("'pending'")
    )
    indexed_at: Mapped[datetime | None] = mapped_column(db.DateTime(timezone=True), nullable=True)
    stats: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    error_message: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    indexing_config_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(db.String(255), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )
    last_dispatched_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class Chunk(db.Model):
    """A text chunk produced by chunk_embed indexing.

    Table: ai.chunks
    Note: embedding column moved to ai.embeddings table.
    """

    __tablename__ = "chunks"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    indexed_source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.indexed_sources.id", ondelete="CASCADE"),
        nullable=True,
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    text: Mapped[str] = mapped_column(db.Text, nullable=False)
    chunk_index: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    start_char: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    end_char: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    tokens: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class PageIndexTOC(db.Model):
    """Document table of contents — one row per document, lightweight metadata.

    Table: ai.page_index_toc
    """

    __tablename__ = "page_index_toc"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    indexed_source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.indexed_sources.id", ondelete="CASCADE"),
        nullable=True,
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    doc_name: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    doc_description: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    structure: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class PageIndexNodes(db.Model):
    """One row per section/node with full text.

    Table: ai.page_index_nodes
    """

    __tablename__ = "page_index_nodes"
    __table_args__ = (
        UniqueConstraint("toc_id", "node_id", name="page_index_nodes_toc_id_node_id_key"),
        {"schema": "ai"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    toc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.page_index_toc.id", ondelete="CASCADE"),
        nullable=False,
    )
    indexed_source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.indexed_sources.id", ondelete="CASCADE"),
        nullable=True,
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    node_id: Mapped[str] = mapped_column(db.Text, nullable=False)
    title: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    depth: Mapped[int | None] = mapped_column(db.Integer, server_default=sa_text("0"))
    parent_node_id: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    text: Mapped[str] = mapped_column(db.Text, nullable=False)
    line_num: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class GraphIndexTOC(db.Model):
    """Graph-based document table of contents — one row per document.

    Table: ai.graph_index_toc
    """

    __tablename__ = "graph_index_toc"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    indexed_source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.indexed_sources.id", ondelete="CASCADE"),
        nullable=True,
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    doc_name: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    doc_description: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    structure: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class GraphIndexNodes(db.Model):
    """Graph index nodes — one row per section/node with enrichment support.

    Table: ai.graph_index_nodes
    """

    __tablename__ = "graph_index_nodes"
    __table_args__ = (
        UniqueConstraint("toc_id", "node_id", name="graph_index_nodes_toc_id_node_id_key"),
        {"schema": "ai"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    toc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.graph_index_toc.id", ondelete="CASCADE"),
        nullable=False,
    )
    indexed_source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.indexed_sources.id", ondelete="CASCADE"),
        nullable=True,
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    node_id: Mapped[str] = mapped_column(db.Text, nullable=False)
    title: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    depth: Mapped[int | None] = mapped_column(db.Integer, server_default=sa_text("0"))
    parent_node_id: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    text: Mapped[str] = mapped_column(db.Text, nullable=False)
    line_num: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    enrichment_error: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class Embedding(db.Model):
    """Polymorphic embedding storage — one row per item+model combination.

    Table: ai.embeddings
    """

    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint(
            "item_id", "embedding_model", name="embeddings_item_id_embedding_model_key"
        ),
        CheckConstraint(
            "item_table IN ('chunks', 'graph_index_nodes', 'full_documents', 'doc2json_documents')",
            name="embeddings_item_table_check",
        ),
        {"schema": "ai"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    item_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    item_table: Mapped[str] = mapped_column(db.String(50), nullable=False)
    indexed_source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.indexed_sources.id", ondelete="CASCADE"),
        nullable=True,
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    embedding_model: Mapped[str] = mapped_column(db.String(255), nullable=False)
    dims: Mapped[int] = mapped_column(db.SmallInteger, nullable=False)
    embedding = mapped_column(Vector(), nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class FullDocument(db.Model):
    """Document-level summary for the full_document strategy.

    Table: ai.full_documents
    Note: summary_embedding and embedding_model moved to ai.embeddings table.
    """

    __tablename__ = "full_documents"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    indexed_source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.indexed_sources.id", ondelete="CASCADE"),
        nullable=True,
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    summary: Mapped[str] = mapped_column(db.Text, nullable=False)
    full_text_path: Mapped[str] = mapped_column(db.String(1024), nullable=False)
    summary_model: Mapped[str | None] = mapped_column(db.String(255), nullable=True)
    summary_tokens: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    full_text_tokens: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class Doc2JSONDocument(db.Model):
    """Document-level JSON extraction for the doc2json strategy.

    Table: ai.doc2json_documents
    Stores structured JSON extraction results from sliding window processing.
    """

    __tablename__ = "doc2json_documents"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    indexed_source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.indexed_sources.id", ondelete="CASCADE"),
        nullable=True,
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    summary: Mapped[str] = mapped_column(db.Text, nullable=False)
    extracted_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    json_schema: Mapped[dict] = mapped_column(JSONB, nullable=False)
    window_summaries: Mapped[list | None] = mapped_column(
        JSONB, server_default=sa_text("'[]'::jsonb")
    )
    extraction_model: Mapped[str | None] = mapped_column(db.String(255), nullable=True)
    summary_tokens: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    window_size: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    window_overlap: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    window_count: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class Agent(db.Model):
    """An agent definition with configuration.

    Table: ai.agents
    """

    __tablename__ = "agents"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    name: Mapped[str] = mapped_column(db.String(255), nullable=False)
    model: Mapped[str] = mapped_column(
        db.String(255), nullable=False, server_default=sa_text(f"'{AGENT_DEFAULT_MODEL}'")
    )
    system_prompt: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    settings: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class Tool(db.Model):
    """A tool definition that agents can invoke.

    Table: ai.tools
    """

    __tablename__ = "tools"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    name: Mapped[str] = mapped_column(db.String(255), nullable=False)
    description: Mapped[str] = mapped_column(db.Text, nullable=False)
    type: Mapped[str] = mapped_column(db.String(50), nullable=False)
    input_schema: Mapped[dict] = mapped_column(JSONB, nullable=False)
    config: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class AgentTool(db.Model):
    """Assignment of a tool to an agent.

    Table: ai.agent_tools
    """

    __tablename__ = "agent_tools"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.agents.id", ondelete="CASCADE"),
        nullable=False,
    )
    tool_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.tools.id", ondelete="CASCADE"),
        nullable=True,
    )
    tool_type: Mapped[str] = mapped_column(db.String(50), nullable=False)
    tool_name: Mapped[str] = mapped_column(db.String(255), nullable=False)
    config_override: Mapped[dict | None] = mapped_column(
        JSONB, server_default=sa_text("'{}'::jsonb")
    )
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class AgentKnowledgeBase(db.Model):
    """Assignment of a knowledge base to an agent for dynamic search.

    Table: ai.agent_knowledge_bases
    """

    __tablename__ = "agent_knowledge_bases"
    __table_args__ = (
        db.UniqueConstraint("agent_id", "knowledge_base_id"),
        {"schema": "ai"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.agents.id", ondelete="CASCADE"),
        nullable=False,
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    config: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class AgentMcpServer(db.Model):
    """An MCP server configured for an agent.

    Table: ai.agent_mcp_servers
    """

    __tablename__ = "agent_mcp_servers"
    __table_args__ = (
        db.UniqueConstraint("agent_id", "name"),
        {"schema": "ai"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.agents.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(db.String(255), nullable=False)
    transport: Mapped[str] = mapped_column(
        db.String(50), nullable=False, server_default=sa_text("'http'")
    )
    url: Mapped[str] = mapped_column(db.Text, nullable=False)
    headers: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    config: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    enabled: Mapped[bool | None] = mapped_column(db.Boolean, server_default=sa_text("true"))
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class AgentSession(db.Model):
    """A persistent conversation session for an agent.

    Table: ai.agent_sessions
    """

    __tablename__ = "agent_sessions"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    session_id: Mapped[str] = mapped_column(db.String(255), unique=True, nullable=False)
    agent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.agents.id", ondelete="SET NULL"),
        nullable=True,
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    session_data: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", JSONB, server_default=sa_text("'{}'::jsonb")
    )
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class AgentRun(db.Model):
    """An individual agent run/invocation within a session.

    Table: ai.agent_runs
    """

    __tablename__ = "agent_runs"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.agent_sessions.id", ondelete="CASCADE"),
        nullable=True,
    )
    run_id: Mapped[str] = mapped_column(db.String(255), unique=True, nullable=False)
    context_handler_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.context_handlers.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str | None] = mapped_column(db.String(50), server_default=sa_text("'pending'"))
    input_messages: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    output_messages: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    content: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    retrieved_context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(db.DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(db.DateTime(timezone=True), nullable=True)
    parent_orchestration_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.orchestration_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    steps: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    events: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'[]'::jsonb"))
    reasoning_steps: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    parent_workflow_execution_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.workflow_executions.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Denormalized identity + typed usage cols (replaces `usage` / `tool_calls`
    # JSONB; see migration 0018). Populated by services/session.py.
    agent_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    model: Mapped[str | None] = mapped_column(db.String(128), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    reasoning_tokens: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    cached_tokens: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    tool_call_count: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    tool_call_error_count: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    tool_call_duration_ms_total: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )

    @property
    def usage(self) -> dict[str, int] | None:
        """Back-compat shim for callers that read the old `usage` JSONB col."""
        return _pack_usage_from_attrs(
            self.prompt_tokens,
            self.completion_tokens,
            self.reasoning_tokens,
            self.cached_tokens,
            self.total_tokens,
        )

    @property
    def tool_calls(self) -> list[dict]:
        """Back-compat shim: materialize tool_calls list from tool_call_events.

        Reads the full JSONB `arguments` / `result` columns so multimodal
        tool responses round-trip; falls back to text previews for legacy
        rows backfilled before those columns existed.
        """
        from sqlalchemy.orm import object_session

        sess = object_session(self)
        if sess is None:
            return []
        rows = sess.execute(
            sa_text(
                """
                SELECT step, tool_name, arguments, result,
                       arguments_preview, result_preview, duration_ms
                FROM ai.tool_call_events
                WHERE agent_run_id = :run_id
                ORDER BY COALESCE(step, 0), occurred_at
                """
            ),
            {"run_id": str(self.id)},
        ).fetchall()
        out: list[dict] = []
        for r in rows:
            args_full, result_full = r[2], r[3]
            args_preview, result_preview = r[4], r[5]
            out.append(
                {
                    "step": r[0],
                    "tool_name": r[1],
                    "arguments": args_full if args_full is not None else args_preview,
                    "result": result_full if result_full is not None else result_preview,
                    "duration_ms": r[6],
                }
            )
        return out


def _pack_usage_from_attrs(
    prompt: int | None,
    completion: int | None,
    reasoning: int | None,
    cached: int | None,
    total: int | None,
) -> dict[str, int] | None:
    """Assemble a `usage` dict from typed-col values, or None if all empty."""
    if all(v is None for v in (prompt, completion, reasoning, cached, total)):
        return None
    out: dict[str, int] = {}
    if prompt is not None:
        out["prompt_tokens"] = prompt
    if completion is not None:
        out["completion_tokens"] = completion
    if reasoning is not None:
        out["reasoning_tokens"] = reasoning
    if cached is not None:
        out["cached_tokens"] = cached
    if total is not None:
        out["total_tokens"] = total
    return out


class ContextHandler(db.Model):
    """Encapsulates a single retrieval operation over one or more knowledge bases.

    Table: ai.context_handlers
    """

    __tablename__ = "context_handlers"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    query: Mapped[str] = mapped_column(db.Text, nullable=False)
    status: Mapped[str | None] = mapped_column(db.String(50), server_default=sa_text("'pending'"))
    knowledge_base_configs: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=sa_text("'[]'::jsonb")
    )
    max_context_tokens: Mapped[int | None] = mapped_column(
        db.Integer, server_default=sa_text(str(DEFAULT_MAX_CONTEXT_TOKENS))
    )
    retrieved_context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", JSONB, server_default=sa_text("'{}'::jsonb")
    )
    errors: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'[]'::jsonb"))
    formatted_context: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )
    completed_at: Mapped[datetime | None] = mapped_column(db.DateTime(timezone=True), nullable=True)


class ProjectSettings(db.Model):
    """Key-value project settings.

    Table: ai.project_settings
    """

    __tablename__ = "project_settings"
    __table_args__ = {"schema": "ai"}

    key: Mapped[str] = mapped_column(db.String(255), primary_key=True)
    value: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class Workflow(db.Model):
    """Stub so SQLAlchemy can resolve the copilot_sessions.workflow_id FK.

    Michael added this stub — not sure what this class is for. The ai.workflows
    table is defined in ai_schema.sql and owned by the studio frontend (accessed
    via PostgREST). Nothing in the backend queries it. Without this class,
    db.metadata.create_all() in the test conftest fails because it can't
    resolve the FK target in ORM metadata.
    """

    __tablename__ = "workflows"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )


class WorkflowExecution(db.Model):
    """Stub so SQLAlchemy can resolve the agent_runs.parent_workflow_execution_id FK.

    The ai.workflow_executions table is defined in ai_schema.sql and accessed via raw SQL
    from routes/workflows.py. This class only exists so ORM metadata can resolve the FK
    target (same rationale as the Workflow stub above).
    """

    __tablename__ = "workflow_executions"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )


class CopilotSession(db.Model):
    """A copilot chat session for a workflow.

    Table: ai.copilot_sessions
    """

    __tablename__ = "copilot_sessions"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    workflow_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.workflows.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        db.DateTime(timezone=True), nullable=False, server_default=sa_text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        db.DateTime(timezone=True), nullable=False, server_default=sa_text("now()")
    )


class CopilotMessage(db.Model):
    """A message in a copilot chat session.

    Table: ai.copilot_messages
    """

    __tablename__ = "copilot_messages"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.copilot_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(db.String(20), nullable=False)
    content: Mapped[str] = mapped_column(db.Text, nullable=False)
    workflow_diff: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    pre_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        db.DateTime(timezone=True), nullable=False, server_default=sa_text("now()")
    )


class EnrichmentConfig(db.Model):
    """Metadata enrichment configuration for a knowledge base.

    Table: ai.enrichment_configs
    """

    __tablename__ = "enrichment_configs"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.knowledge_bases.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    fields: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=sa_text("'[]'::jsonb")
    )
    llm_model: Mapped[str | None] = mapped_column(db.String(255), nullable=True)
    max_tokens: Mapped[int | None] = mapped_column(db.Integer, server_default=sa_text("2000"))
    use_multimodal: Mapped[bool | None] = mapped_column(db.Boolean, server_default=sa_text("false"))
    metadata_table_name: Mapped[str] = mapped_column(db.String(255), nullable=False)
    status: Mapped[str | None] = mapped_column(db.String(50), server_default=sa_text("'idle'"))
    enriched_count: Mapped[int | None] = mapped_column(db.Integer, server_default=sa_text("0"))
    total_count: Mapped[int | None] = mapped_column(db.Integer, server_default=sa_text("0"))
    error_message: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(db.String(255), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class Hook(db.Model):
    """A hook that fires on agent or orchestration lifecycle events.

    Table: ai.hooks
    """

    __tablename__ = "hooks"
    __table_args__ = (
        CheckConstraint(
            "agent_id IS NOT NULL OR orchestration_id IS NOT NULL",
            name="hooks_agent_or_orchestration_check",
        ),
        {"schema": "ai"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    agent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.agents.id", ondelete="CASCADE"),
        nullable=True,
    )
    orchestration_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.orchestrations.id", ondelete="CASCADE"),
        nullable=True,
    )
    event: Mapped[str] = mapped_column(db.String(50), nullable=False)
    matcher: Mapped[str | None] = mapped_column(db.String(255), nullable=True)
    type: Mapped[str] = mapped_column(db.String(50), nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    enabled: Mapped[bool | None] = mapped_column(db.Boolean, server_default=sa_text("true"))
    position: Mapped[int | None] = mapped_column(db.Integer, server_default=sa_text("0"))
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class OrchestrationModel(db.Model):
    """An orchestration definition.

    Table: ai.orchestrations
    """

    __tablename__ = "orchestrations"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    name: Mapped[str] = mapped_column(db.String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    strategy: Mapped[str] = mapped_column(
        db.String(50), nullable=False, server_default=sa_text("'supervisor'")
    )
    orchestrator_config: Mapped[dict | None] = mapped_column(
        JSONB, server_default=sa_text("'{}'::jsonb")
    )
    settings: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class OrchestrationEntityModel(db.Model):
    """An entity within an orchestration.

    Table: ai.orchestration_entities
    """

    __tablename__ = "orchestration_entities"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    orchestration_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.orchestrations.id", ondelete="CASCADE"),
        nullable=False,
    )
    entity_type: Mapped[str] = mapped_column(db.String(50), nullable=False)
    entity_ref_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    role_description: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    config: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    position: Mapped[int | None] = mapped_column(db.Integer, server_default=sa_text("0"))
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class OrchestrationSessionModel(db.Model):
    """A session for an orchestration.

    Table: ai.orchestration_sessions
    """

    __tablename__ = "orchestration_sessions"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    session_id: Mapped[str] = mapped_column(db.String(255), unique=True, nullable=False)
    orchestration_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.orchestrations.id", ondelete="SET NULL"),
        nullable=True,
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    session_data: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'{}'::jsonb"))
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", JSONB, server_default=sa_text("'{}'::jsonb")
    )
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )


class AIProviderKey(db.Model):
    """Per-project LLM provider API key (encrypted at rest).

    Table: ai.ai_provider_keys
    """

    __tablename__ = "ai_provider_keys"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    provider: Mapped[str] = mapped_column(db.String(50), nullable=False, unique=True)
    api_key_encrypted: Mapped[str] = mapped_column(db.Text, nullable=False)
    is_valid: Mapped[bool] = mapped_column(
        db.Boolean, nullable=False, server_default=sa_text("true")
    )
    last_validated_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        db.DateTime(timezone=True), nullable=False, server_default=sa_text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        db.DateTime(timezone=True), nullable=False, server_default=sa_text("now()")
    )

    def to_dict(self) -> dict:
        from ..services.encryption import decrypt_api_key, mask_api_key

        try:
            plaintext = decrypt_api_key(self.api_key_encrypted)
            masked = mask_api_key(plaintext)
        except Exception:
            masked = "***"
        return {
            "id": str(self.id),
            "provider": self.provider,
            "masked_key": masked,
            "is_valid": self.is_valid,
            "last_validated_at": (
                self.last_validated_at.isoformat() if self.last_validated_at else None
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class OrchestrationRunModel(db.Model):
    """A run of an orchestration.

    Table: ai.orchestration_runs
    """

    __tablename__ = "orchestration_runs"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.orchestration_sessions.id", ondelete="CASCADE"),
        nullable=True,
    )
    run_id: Mapped[str] = mapped_column(db.String(255), unique=True, nullable=False)
    orchestration_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.orchestrations.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str | None] = mapped_column(db.String(50), server_default=sa_text("'running'"))
    input_messages: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    content: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    events: Mapped[dict | None] = mapped_column(JSONB, server_default=sa_text("'[]'::jsonb"))
    error: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    reasoning_requested: Mapped[bool | None] = mapped_column(db.Boolean, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(db.DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(db.DateTime(timezone=True), nullable=True)
    # Typed usage cols (replaces `usage` JSONB; see migration 0018).
    model: Mapped[str | None] = mapped_column(db.String(128), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    reasoning_tokens: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    cached_tokens: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    tool_call_count: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    tool_call_error_count: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    tool_call_duration_ms_total: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )

    @property
    def usage(self) -> dict[str, int] | None:
        """Back-compat shim for callers that read the old `usage` JSONB col."""
        return _pack_usage_from_attrs(
            self.prompt_tokens,
            self.completion_tokens,
            self.reasoning_tokens,
            self.cached_tokens,
            self.total_tokens,
        )


class ToolCallEvent(db.Model):
    """One row per tool invocation within an agent/orchestration/workflow run.

    Table: ai.tool_call_events

    Feeds the observability dashboards' per-tool charts (calls by tool, p95
    duration, error rate) and per-run drill-down. Denormalized agent_id/model
    let the dashboard filter without joining back to agent_runs.
    """

    __tablename__ = "tool_call_events"
    __table_args__ = {"schema": "ai"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()")
    )
    agent_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.agent_runs.id", ondelete="CASCADE"),
        nullable=True,
    )
    orchestration_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.orchestration_runs.id", ondelete="CASCADE"),
        nullable=True,
    )
    workflow_execution_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai.workflow_executions.id", ondelete="CASCADE"),
        nullable=True,
    )
    agent_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    model: Mapped[str | None] = mapped_column(db.String(128), nullable=True)
    tool_name: Mapped[str] = mapped_column(db.String(255), nullable=False)
    status: Mapped[str] = mapped_column(db.String(16), nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    arguments: Mapped[dict | list | None] = mapped_column(JSONB, nullable=True)
    result: Mapped[dict | list | None] = mapped_column(JSONB, nullable=True)
    arguments_preview: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    result_preview: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    error: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    step: Mapped[int | None] = mapped_column(db.Integer, nullable=True)
    occurred_at: Mapped[datetime | None] = mapped_column(
        db.DateTime(timezone=True), server_default=sa_text("now()")
    )
