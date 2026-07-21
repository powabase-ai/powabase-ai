"""
Metadata Enrichment Service.

Manages dynamic per-KB metadata tables and drives LLM-based field extraction
for every item (chunk, node, or full document) in a knowledge base.
"""

import ast
import asyncio
import base64
import json
import logging
from typing import Any, Callable, Literal

from sqlalchemy import text
from sqlalchemy.orm import Session

from agentic.knowledge.model_config import (
    METADATA_ENRICHMENT_BATCH_SIZE,
    METADATA_ENRICHMENT_DEFAULT_MAX_TOKENS,
    METADATA_ENRICHMENT_MAX_CONCURRENT,
    METADATA_ENRICHMENT_MAX_IMAGES,
    METADATA_ENRICHMENT_MAX_INPUT_CHARS,
    METADATA_ENRICHMENT_MAX_INPUT_CHARS_MULTIMODAL,
    METADATA_ENRICHMENT_MAX_RETRIES,
)

from ..db import AI_SCHEMA
from .llm_call import cached_byok_resolver, with_llm_key
from .run_context import run_scope
from .sql_utils import validate_sql_identifier
from .storage import get_storage

logger = logging.getLogger(__name__)


class EnrichmentParseError(Exception):
    """Raised when LLM returns unparseable JSON after all retries."""


# Mapping from enrichment field types to PostgreSQL column types
_PG_TYPE_MAP = {
    "text": "TEXT",
    "boolean": "BOOLEAN",
    "number": "DOUBLE PRECISION",
    "enum": "VARCHAR(255)",
}

# After this many consecutive trippable errors from the same provider in a
# single enrichment run, short-circuit remaining items.
_CIRCUIT_BREAKER_THRESHOLD = 5

# Exceptions that count as "trippable" for the per-provider circuit breaker.
# Imported lazily (inside functions) to avoid top-level litellm import cost;
# this tuple is evaluated once per _enrich_one call via the except clause.
# Important 1 from PR #440 review: broadened from (RateLimitError,
# AuthenticationError) to also include infrastructure/service errors so a
# multi-hour 5xx outage triggers the breaker rather than running all items.
_TRIPPABLE_EXCEPTIONS_NAMES = (
    "RateLimitError",
    "AuthenticationError",
    "ServiceUnavailableError",
    "InternalServerError",
    "BadGatewayError",
    "APIConnectionError",
    "Timeout",
)


def _provider_for_model(model: str) -> str:
    """Return the LiteLLM provider name for a model string."""
    try:
        import litellm

        _, provider, _, _ = litellm.get_llm_provider(model)
        return provider
    except Exception:
        return model.split("/")[0] if "/" in model else "unknown"


def _build_metadata_json_schema(fields: list[dict]) -> dict:
    """Build a JSON Schema for the metadata extraction response."""
    properties = {}
    for f in fields:
        fname = f["name"]
        ftype = f["type"]
        if ftype == "boolean":
            properties[fname] = {"type": ["boolean", "null"]}
        elif ftype == "number":
            properties[fname] = {"type": ["number", "null"]}
        elif ftype == "enum":
            properties[fname] = {
                "anyOf": [
                    {"type": "string", "enum": f["enum_values"]},
                    {"type": "null"},
                ],
            }
        else:  # text
            properties[fname] = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": properties,
        "required": [f["name"] for f in fields],
        "additionalProperties": False,
    }


class MetadataEnricher:
    """Service for creating per-KB metadata tables and running LLM enrichment."""

    def __init__(self, db_session: Session, knowledge_base_id: str):
        self.db_session = db_session
        self.kb_id = knowledge_base_id

    # ------------------------------------------------------------------
    # Dynamic DDL
    # ------------------------------------------------------------------

    @staticmethod
    def table_name_for_kb(kb_id: str) -> str:
        """Generate table name: 'kb_metadata_' + UUID hex (no hyphens)."""
        return "kb_metadata_" + kb_id.replace("-", "")

    def create_metadata_table(self, fields: list[dict]) -> str:
        """CREATE TABLE with typed columns based on field definitions.

        Also applies RLS policies, grants, and indexes.
        Returns the table name.
        """
        table_name = self.table_name_for_kb(self.kb_id)
        qualified = f'"{AI_SCHEMA}"."{table_name}"'

        # Build column definitions from fields
        all_col_defs = [
            "id UUID PRIMARY KEY DEFAULT gen_random_uuid()",
            "item_id UUID NOT NULL UNIQUE",
            "item_type VARCHAR(50) NOT NULL CHECK (item_type IN ('chunk', 'node', 'full_document'))",
        ]
        for f in fields:
            pg_type = _PG_TYPE_MAP.get(f["type"])
            if not pg_type:
                raise ValueError(f"Unknown field type: {f['type']}")
            validate_sql_identifier(f["name"])
            all_col_defs.append(f'"{f["name"]}" {pg_type}')
        all_col_defs.append('"_enrichment_error" TEXT')
        all_col_defs.append("enriched_at TIMESTAMPTZ DEFAULT NOW()")

        create_sql = (
            f"CREATE TABLE IF NOT EXISTS {qualified} (\n" + ",\n".join(all_col_defs) + "\n)"
        )
        self.db_session.execute(text(create_sql))

        # RLS
        self.db_session.execute(text(f"ALTER TABLE {qualified} ENABLE ROW LEVEL SECURITY"))

        rls_policies = [
            (
                "service_role_all",
                f"CREATE POLICY service_role_all ON {qualified} FOR ALL TO service_role USING (true) WITH CHECK (true)",
            ),
            (
                "auth_read",
                f"CREATE POLICY auth_read ON {qualified} FOR SELECT TO authenticated USING (true)",
            ),
            (
                "auth_write",
                f"CREATE POLICY auth_write ON {qualified} FOR INSERT TO authenticated WITH CHECK (true)",
            ),
            (
                "auth_update",
                f"CREATE POLICY auth_update ON {qualified} FOR UPDATE TO authenticated USING (true) WITH CHECK (true)",
            ),
        ]
        for policy_name, create_stmt in rls_policies:
            self.db_session.execute(text(f"DROP POLICY IF EXISTS {policy_name} ON {qualified}"))
            self.db_session.execute(text(create_stmt))

        self.db_session.execute(
            text(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON {qualified} TO authenticated, service_role"
            )
        )
        self.db_session.execute(text(f"GRANT SELECT ON {qualified} TO anon"))

        # Indexes
        self.db_session.execute(
            text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_item_id ON {qualified} (item_id)")
        )
        # Additional indexes on enum/boolean columns for filtering
        for f in fields:
            if f["type"] in ("enum", "boolean"):
                self.db_session.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{f['name']} "
                        f'ON {qualified} ("{f["name"]}")'
                    )
                )

        # Ensure the cleanup trigger exists (idempotent — auto-migrates existing projects)
        self.db_session.execute(
            text("""
            CREATE OR REPLACE FUNCTION ai.cleanup_enrichment_metadata()
            RETURNS TRIGGER AS $fn$
            DECLARE
                _meta_table TEXT;
            BEGIN
                SELECT metadata_table_name INTO _meta_table
                FROM ai.enrichment_configs
                WHERE knowledge_base_id = OLD.knowledge_base_id;
                IF _meta_table IS NULL OR LEFT(_meta_table, 12) != 'kb_metadata_' THEN
                    RETURN OLD;
                END IF;
                BEGIN
                    EXECUTE format(
                        'DELETE FROM ai.%I WHERE item_id IN ('
                        '  SELECT id FROM ai.chunks WHERE indexed_source_id = $1'
                        '  UNION ALL'
                        '  SELECT id FROM ai.page_index_nodes WHERE indexed_source_id = $1'
                        '  UNION ALL'
                        '  SELECT id FROM ai.full_documents WHERE indexed_source_id = $1'
                        '  UNION ALL'
                        '  SELECT id FROM ai.graph_index_nodes WHERE indexed_source_id = $1'
                        ')', _meta_table
                    ) USING OLD.id;
                EXCEPTION WHEN undefined_table THEN NULL;
                END;
                RETURN OLD;
            END;
            $fn$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = ai
        """)
        )
        self.db_session.execute(
            text(
                "DROP TRIGGER IF EXISTS cleanup_enrichment_metadata_on_indexed_source_delete "
                "ON ai.indexed_sources"
            )
        )
        self.db_session.execute(
            text(
                "CREATE TRIGGER cleanup_enrichment_metadata_on_indexed_source_delete "
                "BEFORE DELETE ON ai.indexed_sources "
                "FOR EACH ROW EXECUTE FUNCTION ai.cleanup_enrichment_metadata()"
            )
        )

        logger.info(f"Created metadata table {qualified} with {len(fields)} columns")
        return table_name

    def drop_metadata_table(self, table_name: str) -> None:
        """DROP TABLE IF EXISTS ai.{table_name}."""
        if not table_name.startswith("kb_metadata_"):
            raise ValueError(f"Refusing to drop non-metadata table: {table_name}")
        self.db_session.execute(text(f'DROP TABLE IF EXISTS "{AI_SCHEMA}"."{table_name}" CASCADE'))
        logger.info(f"Dropped metadata table {AI_SCHEMA}.{table_name}")

    def _ensure_error_column(self, table_name: str) -> None:
        """Add _enrichment_error column if it doesn't exist (migrates pre-existing tables)."""
        qualified = f'"{AI_SCHEMA}"."{table_name}"'
        self.db_session.execute(
            text(f'ALTER TABLE {qualified} ADD COLUMN IF NOT EXISTS "_enrichment_error" TEXT')
        )
        self.db_session.commit()

    # ------------------------------------------------------------------
    # Item loading
    # ------------------------------------------------------------------

    def get_enrichable_items(
        self,
        strategy: str,
        include_source_info: bool = False,
    ) -> list[dict]:
        """Fetch all items from the KB based on strategy.

        Returns [{"id": ..., "text": ..., "item_type": ...}, ...]
        When include_source_info=True, also includes "source_id" and "meta".
        """
        if strategy == "page_index":
            if include_source_info:
                result = self.db_session.execute(
                    text(
                        f"SELECT id, text, meta, source_id "
                        f'FROM "{AI_SCHEMA}".page_index_nodes '
                        f"WHERE knowledge_base_id = :kb_id"
                    ),
                    {"kb_id": self.kb_id},
                )
                return [
                    {
                        "id": str(row[0]),
                        "text": row[1],
                        "meta": row[2] or {},
                        "source_id": str(row[3]),
                        "item_type": "node",
                    }
                    for row in result
                ]
            result = self.db_session.execute(
                text(
                    f'SELECT id, text FROM "{AI_SCHEMA}".page_index_nodes '
                    f"WHERE knowledge_base_id = :kb_id"
                ),
                {"kb_id": self.kb_id},
            )
            return [{"id": str(row[0]), "text": row[1], "item_type": "node"} for row in result]
        elif strategy == "graph_index":
            if include_source_info:
                result = self.db_session.execute(
                    text(
                        f"SELECT id, text, meta, source_id "
                        f'FROM "{AI_SCHEMA}".graph_index_nodes '
                        f"WHERE knowledge_base_id = :kb_id"
                    ),
                    {"kb_id": self.kb_id},
                )
                return [
                    {
                        "id": str(row[0]),
                        "text": row[1],
                        "meta": row[2] or {},
                        "source_id": str(row[3]),
                        "item_type": "node",
                    }
                    for row in result
                ]
            result = self.db_session.execute(
                text(
                    f'SELECT id, text FROM "{AI_SCHEMA}".graph_index_nodes '
                    f"WHERE knowledge_base_id = :kb_id"
                ),
                {"kb_id": self.kb_id},
            )
            return [{"id": str(row[0]), "text": row[1], "item_type": "node"} for row in result]
        elif strategy == "full_document":
            if include_source_info:
                result = self.db_session.execute(
                    text(
                        f'SELECT id, summary, source_id FROM "{AI_SCHEMA}".full_documents '
                        f"WHERE knowledge_base_id = :kb_id"
                    ),
                    {"kb_id": self.kb_id},
                )
                return [
                    {
                        "id": str(row[0]),
                        "text": row[1],
                        "source_id": str(row[2]),
                        "item_type": "full_document",
                    }
                    for row in result
                ]
            result = self.db_session.execute(
                text(
                    f'SELECT id, summary FROM "{AI_SCHEMA}".full_documents '
                    f"WHERE knowledge_base_id = :kb_id"
                ),
                {"kb_id": self.kb_id},
            )
            return [
                {"id": str(row[0]), "text": row[1], "item_type": "full_document"} for row in result
            ]
        else:  # chunk_embed (default)
            if include_source_info:
                result = self.db_session.execute(
                    text(
                        f"SELECT id, text, meta, source_id "
                        f'FROM "{AI_SCHEMA}".chunks '
                        f"WHERE knowledge_base_id = :kb_id"
                    ),
                    {"kb_id": self.kb_id},
                )
                return [
                    {
                        "id": str(row[0]),
                        "text": row[1],
                        "meta": row[2] or {},
                        "source_id": str(row[3]),
                        "item_type": "chunk",
                    }
                    for row in result
                ]
            result = self.db_session.execute(
                text(f'SELECT id, text FROM "{AI_SCHEMA}".chunks WHERE knowledge_base_id = :kb_id'),
                {"kb_id": self.kb_id},
            )
            return [{"id": str(row[0]), "text": row[1], "item_type": "chunk"} for row in result]

    @staticmethod
    def _strategy_table_info(strategy: str) -> tuple[str, str, str]:
        """Return (source_table, text_col, item_type) for a strategy."""
        if strategy == "page_index":
            return f'"{AI_SCHEMA}".page_index_nodes', "text", "node"
        elif strategy == "graph_index":
            return f'"{AI_SCHEMA}".graph_index_nodes', "text", "node"
        elif strategy == "full_document":
            return f'"{AI_SCHEMA}".full_documents', "summary", "full_document"
        else:
            return f'"{AI_SCHEMA}".chunks', "text", "chunk"

    def get_unenriched_items(
        self,
        strategy: str,
        table_name: str,
        include_source_info: bool = False,
    ) -> list[dict]:
        """Fetch items that don't yet have rows in the metadata table."""
        qualified = f'"{AI_SCHEMA}"."{table_name}"'
        source_table, text_col, item_type = self._strategy_table_info(strategy)

        if include_source_info:
            if strategy == "full_document":
                # full_documents has source_id directly
                result = self.db_session.execute(
                    text(
                        f"SELECT s.id, s.{text_col}, s.source_id FROM {source_table} s "
                        f"LEFT JOIN {qualified} m ON s.id = m.item_id "
                        f"WHERE s.knowledge_base_id = :kb_id AND m.id IS NULL"
                    ),
                    {"kb_id": self.kb_id},
                )
                return [
                    {
                        "id": str(row[0]),
                        "text": row[1],
                        "source_id": str(row[2]),
                        "item_type": item_type,
                    }
                    for row in result
                ]
            else:
                # chunks / page_index_nodes have source_id directly
                result = self.db_session.execute(
                    text(
                        f"SELECT s.id, s.{text_col}, s.meta, s.source_id "
                        f"FROM {source_table} s "
                        f"LEFT JOIN {qualified} m ON s.id = m.item_id "
                        f"WHERE s.knowledge_base_id = :kb_id AND m.id IS NULL"
                    ),
                    {"kb_id": self.kb_id},
                )
                return [
                    {
                        "id": str(row[0]),
                        "text": row[1],
                        "meta": row[2] or {},
                        "source_id": str(row[3]),
                        "item_type": item_type,
                    }
                    for row in result
                ]

        result = self.db_session.execute(
            text(
                f"SELECT s.id, s.{text_col} FROM {source_table} s "
                f"LEFT JOIN {qualified} m ON s.id = m.item_id "
                f"WHERE s.knowledge_base_id = :kb_id AND m.id IS NULL"
            ),
            {"kb_id": self.kb_id},
        )
        return [{"id": str(row[0]), "text": row[1], "item_type": item_type} for row in result]

    def get_failed_items(
        self,
        strategy: str,
        table_name: str,
        include_source_info: bool = False,
    ) -> list[dict]:
        """Fetch items that have _enrichment_error set in the metadata table."""
        qualified = f'"{AI_SCHEMA}"."{table_name}"'
        source_table, text_col, item_type = self._strategy_table_info(strategy)

        if include_source_info:
            if strategy == "full_document":
                result = self.db_session.execute(
                    text(
                        f"SELECT s.id, s.{text_col}, s.source_id FROM {source_table} s "
                        f"INNER JOIN {qualified} m ON s.id = m.item_id "
                        f'WHERE s.knowledge_base_id = :kb_id AND m."_enrichment_error" IS NOT NULL'
                    ),
                    {"kb_id": self.kb_id},
                )
                return [
                    {
                        "id": str(row[0]),
                        "text": row[1],
                        "source_id": str(row[2]),
                        "item_type": item_type,
                    }
                    for row in result
                ]
            else:
                # chunks / page_index_nodes have source_id directly
                result = self.db_session.execute(
                    text(
                        f"SELECT s.id, s.{text_col}, s.meta, s.source_id "
                        f"FROM {source_table} s "
                        f"INNER JOIN {qualified} m ON s.id = m.item_id "
                        f'WHERE s.knowledge_base_id = :kb_id AND m."_enrichment_error" IS NOT NULL'
                    ),
                    {"kb_id": self.kb_id},
                )
                return [
                    {
                        "id": str(row[0]),
                        "text": row[1],
                        "meta": row[2] or {},
                        "source_id": str(row[3]),
                        "item_type": item_type,
                    }
                    for row in result
                ]

        result = self.db_session.execute(
            text(
                f"SELECT s.id, s.{text_col} FROM {source_table} s "
                f"INNER JOIN {qualified} m ON s.id = m.item_id "
                f'WHERE s.knowledge_base_id = :kb_id AND m."_enrichment_error" IS NOT NULL'
            ),
            {"kb_id": self.kb_id},
        )
        return [{"id": str(row[0]), "text": row[1], "item_type": item_type} for row in result]

    def count_by_status(self, table_name: str) -> tuple[int, int]:
        """Return (ok_count, failed_count) from the metadata table."""
        qualified = f'"{AI_SCHEMA}"."{table_name}"'
        row = self.db_session.execute(
            text(
                f"SELECT "
                f'COUNT(*) FILTER (WHERE "_enrichment_error" IS NULL) AS ok, '
                f'COUNT(*) FILTER (WHERE "_enrichment_error" IS NOT NULL) AS failed '
                f"FROM {qualified}"
            )
        ).fetchone()
        return (row[0], row[1]) if row else (0, 0)

    def count_total_items(self, strategy: str) -> int:
        """Return total number of enrichable items in source table for this KB."""
        source_table, _, _ = self._strategy_table_info(strategy)

        row = self.db_session.execute(
            text(f"SELECT COUNT(*) FROM {source_table} WHERE knowledge_base_id = :kb_id"),
            {"kb_id": self.kb_id},
        ).fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Image resolution for multimodal enrichment
    # ------------------------------------------------------------------

    def _resolve_images_for_sources(
        self,
        source_ids: list[str],
    ) -> dict[str, list[dict]]:
        """Batch-fetch and base64-encode page images from source derivatives.

        Returns {source_id: [{"page": N, "content": base64, "format": "png"}, ...]}
        """
        if not source_ids:
            return {}

        placeholders = ", ".join(f":sid_{i}" for i in range(len(source_ids)))
        params = {f"sid_{i}": sid for i, sid in enumerate(source_ids)}
        rows = self.db_session.execute(
            text(f'SELECT id, derivatives FROM "{AI_SCHEMA}".sources WHERE id IN ({placeholders})'),
            params,
        ).fetchall()

        # Build source_id -> image derivative records
        source_derivs: dict[str, list[dict]] = {}
        for row in rows:
            sid = str(row[0])
            derivs = row[1] or {}
            image_records = derivs.get("image", [])
            if image_records:
                source_derivs[sid] = image_records

        if not source_derivs:
            return {}

        try:
            storage = get_storage()
        except Exception as e:
            logger.warning("Failed to init storage for multimodal enrichment: %s", e)
            return {}

        result: dict[str, list[dict]] = {}
        for sid, image_records in source_derivs.items():
            resolved = []
            for img_rec in image_records:
                storage_path = img_rec.get("storage_path")
                page = img_rec.get("page")
                fmt = img_rec.get("format", "png")
                if not storage_path or not page:
                    continue
                try:
                    image_bytes = storage.download_from_path(storage_path)
                    b64_data = base64.b64encode(image_bytes).decode("ascii")
                    resolved.append({"page": page, "content": b64_data, "format": fmt})
                except Exception as e:
                    logger.warning(
                        "Failed to resolve image for source %s page %s: %s",
                        sid,
                        page,
                        e,
                    )
            if resolved:
                result[sid] = resolved

        return result

    @staticmethod
    def _get_item_pages(item: dict, strategy: str) -> list[int]:
        """Determine which page numbers an item covers."""
        meta = item.get("meta") or {}
        if strategy == "chunk_embed":
            return meta.get("pages", [])
        elif strategy in ("page_index", "graph_index"):
            start = meta.get("start_page")
            end = meta.get("end_page")
            if start is not None and end is not None:
                return list(range(int(start), int(end) + 1))
            return []
        return []  # full_document → empty means "all pages"

    @staticmethod
    def _match_images_for_item(
        item: dict,
        strategy: str,
        source_image_map: dict,
    ) -> list[dict]:
        """Get the base64 images matching an item's pages."""
        source_id = item.get("source_id")
        if not source_id or source_id not in source_image_map:
            return []
        all_images = source_image_map[source_id]
        pages = MetadataEnricher._get_item_pages(item, strategy)
        if not pages:
            # full_document or unknown → all images, capped
            return sorted(all_images, key=lambda x: x.get("page", 0))[
                :METADATA_ENRICHMENT_MAX_IMAGES
            ]
        matched = [img for img in all_images if img.get("page") in pages]
        return matched[:METADATA_ENRICHMENT_MAX_IMAGES]

    # ------------------------------------------------------------------
    # LLM enrichment
    # ------------------------------------------------------------------

    async def enrich_single_item(
        self,
        item_text: str,
        fields: list[dict],
        model: str,
        max_tokens: int = METADATA_ENRICHMENT_DEFAULT_MAX_TOKENS,
        images: list[dict] | None = None,
    ) -> dict:
        """Call LLM to extract metadata for ONE item.

        When images are provided, builds a multimodal message with
        chunk/node text as the primary scope and page images as supporting visual context.

        Returns {"field_name": value, ...}
        """
        import litellm
        from litellm import supports_response_schema

        # Build field description lines
        field_lines = []
        for i, f in enumerate(fields, 1):
            type_desc = f["type"]
            if f["type"] == "enum":
                type_desc = f"enum: {', '.join(f['enum_values'])}"
            field_lines.append(f'{i}. "{f["name"]}" ({type_desc}) — {f["description"]}')
        field_descriptions = "\n".join(field_lines)

        # Build example output
        example = {}
        for f in fields:
            if f["type"] == "enum":
                example[f["name"]] = f["enum_values"][0]
            elif f["type"] == "boolean":
                example[f["name"]] = True
            elif f["type"] == "number":
                example[f["name"]] = 0.5
            else:
                example[f["name"]] = "example text"

        if images:
            # Multimodal path: text is primary scope, images are supporting context
            if item_text:
                # Text is primary scope, images are supporting context
                _text = item_text
                if (
                    METADATA_ENRICHMENT_MAX_INPUT_CHARS_MULTIMODAL > 0
                    and len(item_text) > METADATA_ENRICHMENT_MAX_INPUT_CHARS_MULTIMODAL
                ):
                    _text = item_text[:METADATA_ENRICHMENT_MAX_INPUT_CHARS_MULTIMODAL]
                prompt_text = (
                    "You are a metadata extractor. You will be given a specific text "
                    "section from a document, along with page images for visual context.\n\n"
                    "IMPORTANT: Extract metadata ONLY for the text section below. "
                    "The page images may contain additional content beyond this section — "
                    "ignore anything not covered by the text.\n\n"
                    f"Text section to analyze:\n---\n{_text}\n---\n\n"
                    f"Fields to extract:\n{field_descriptions}\n\n"
                    "The following page images show the document pages containing this "
                    "section. Use them for visual context (tables, figures, formatting) "
                    "but restrict your extraction to the text section above.\n\n"
                )
            else:
                # No text available — fall back to image-primary extraction
                prompt_text = (
                    "You are a metadata extractor. Extract the requested fields from "
                    "the following document page images.\n\n"
                    f"Fields to extract:\n{field_descriptions}\n\n"
                )
            prompt_text += (
                "Return ONLY a valid JSON object with the field values. "
                "Use standard JSON with double quotes for all keys and string values. "
                "Keep values concise. "
                "Example:\n"
                + json.dumps(example)
                + "\n\nOutput your response in clean, standard JSON format. "
                "Do not output anything else."
            )

            content_blocks: list[dict] = [{"type": "text", "text": prompt_text}]
            for img in images:
                fmt = img.get("format", "png").lower()
                mime = f"image/{fmt}" if fmt not in ("jpg", "jpeg") else "image/jpeg"
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{img['content']}",
                        },
                    }
                )
            messages = [{"role": "user", "content": content_blocks}]
        else:
            # Text-only path (original behavior)
            _text = item_text or ""
            if (
                METADATA_ENRICHMENT_MAX_INPUT_CHARS > 0
                and len(_text) > METADATA_ENRICHMENT_MAX_INPUT_CHARS
            ):
                _text = _text[:METADATA_ENRICHMENT_MAX_INPUT_CHARS]
            prompt = (
                "You are a metadata extractor. Given a text from a knowledge base, "
                "extract the requested fields.\n\n"
                "Fields:\n" + field_descriptions + "\n\nText:\n---\n" + _text + "\n---\n\n"
                "Return ONLY a valid JSON object with the field values. "
                "Use standard JSON with double quotes for all keys and string values. "
                "Keep values concise — for text fields, provide a brief extraction "
                "rather than copying large portions of the source text. "
                "Example:\n"
                + json.dumps(example)
                + "\n\nOutput your response in clean, standard JSON format. "
                "Do not output anything else."
            )
            messages = [{"role": "user", "content": prompt}]

        # Use structured JSON schema output when the model supports it
        if supports_response_schema(model=model):
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "metadata_extraction",
                    "strict": True,
                    "schema": _build_metadata_json_schema(fields),
                },
            }
        else:
            response_format = {"type": "json_object"}

        last_raw = ""
        for attempt in range(1, METADATA_ENRICHMENT_MAX_RETRIES + 1):
            with with_llm_key(model) as api_key:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    drop_params=True,
                    num_retries=0,  # LiteLLM-level retry cap; outer JSON loop handles parse failures
                    max_retries=0,  # OpenAI SDK-level cap (separate from LiteLLM's num_retries)
                    timeout=60,  # bound in-flight memory if one call hangs
                    api_key=api_key,
                )

            # Detect output truncation
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if finish_reason == "length":
                logger.warning(
                    "LLM output truncated (finish_reason=length, max_tokens=%d) "
                    "on attempt %d/%d. Consider increasing max_tokens.",
                    max_tokens,
                    attempt,
                    METADATA_ENRICHMENT_MAX_RETRIES,
                )

            last_raw = (response.choices[0].message.content or "").strip()
            try:
                parsed = json.loads(last_raw)
            except json.JSONDecodeError:
                # Fallback: parse Python-style dict literals (single-quoted keys/values)
                try:
                    parsed = ast.literal_eval(last_raw)
                    if not isinstance(parsed, dict):
                        raise ValueError("Expected a dict")
                except (ValueError, SyntaxError):
                    logger.warning(
                        "LLM returned invalid JSON (attempt %d/%d): %s",
                        attempt,
                        METADATA_ENRICHMENT_MAX_RETRIES,
                        last_raw[:200],
                    )
                    continue

            # Validate and sanitize each field value
            result = {}
            for f in fields:
                val = parsed.get(f["name"])
                result[f["name"]] = self._validate_field_value(val, f)
            return result

        raise EnrichmentParseError(
            f"LLM returned unparseable JSON after {METADATA_ENRICHMENT_MAX_RETRIES} retries: {last_raw[:200]}"
        )

    @staticmethod
    def _validate_field_value(val: Any, field_def: dict) -> Any:
        """Validate a single field value against its definition."""
        if val is None:
            return None

        ftype = field_def["type"]
        if ftype == "boolean":
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                if val.lower() in ("true", "yes", "1"):
                    return True
                if val.lower() in ("false", "no", "0"):
                    return False
            return None
        elif ftype == "number":
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        elif ftype == "enum":
            allowed = field_def.get("enum_values", [])
            if str(val) in allowed:
                return str(val)
            # Case-insensitive fallback
            val_lower = str(val).lower()
            for ev in allowed:
                if ev.lower() == val_lower:
                    return ev
            return None
        else:  # text
            return str(val)

    async def run_enrichment(
        self,
        fields: list[dict],
        model: str,
        strategy: str,
        table_name: str,
        incremental: bool = False,
        retry_failed: bool = False,
        max_tokens: int = METADATA_ENRICHMENT_DEFAULT_MAX_TOKENS,
        use_multimodal: bool = False,
        on_batch_complete: Callable[[int, list[str]], Literal["continue", "abort"]] | None = None,
    ) -> dict:
        # Resolve BYOK keys once per enrichment run — without this the
        # per-item with_llm_key wrappers each hit the DB +
        # self-heal commit (one query per item × per retry attempt =
        # N+1 across the whole KB).
        with cached_byok_resolver():
            return await self._run_enrichment_inner(
                fields,
                model,
                strategy,
                table_name,
                incremental,
                retry_failed,
                max_tokens,
                use_multimodal,
                on_batch_complete,
            )

    async def _run_enrichment_inner(
        self,
        fields: list[dict],
        model: str,
        strategy: str,
        table_name: str,
        incremental: bool = False,
        retry_failed: bool = False,
        max_tokens: int = METADATA_ENRICHMENT_DEFAULT_MAX_TOKENS,
        use_multimodal: bool = False,
        on_batch_complete: Callable[[int, list[str]], Literal["continue", "abort"]] | None = None,
    ) -> dict:
        """Main enrichment loop.

        Process items in batches with concurrent LLM calls per batch.
        When use_multimodal=True, fetches source page images and includes them
        in LLM calls. Items without image derivatives fall back to text-only.
        Returns {"enriched_count": N, "total_count": M, "errors": [...]}
        """
        self._ensure_error_column(table_name)

        if retry_failed:
            items = self.get_failed_items(
                strategy,
                table_name,
                include_source_info=use_multimodal,
            )
        elif incremental:
            items = self.get_unenriched_items(
                strategy,
                table_name,
                include_source_info=use_multimodal,
            )
        else:
            self.clear_results(table_name)
            items = self.get_enrichable_items(
                strategy,
                include_source_info=use_multimodal,
            )

        # Resolve images for multimodal enrichment
        source_image_map: dict[str, list[dict]] = {}
        if use_multimodal:
            source_ids = list({item["source_id"] for item in items if item.get("source_id")})
            source_image_map = self._resolve_images_for_sources(source_ids)
            logger.info(
                "Multimodal enrichment: resolved images for %d/%d sources",
                len(source_image_map),
                len(source_ids),
            )

        total = len(items)
        enriched = 0
        errors: list[str] = []
        sem = asyncio.Semaphore(METADATA_ENRICHMENT_MAX_CONCURRENT)

        # Circuit-breaker state: tracks consecutive quota/auth errors per provider.
        provider = _provider_for_model(model)
        tripped_providers: set[str] = set()
        consecutive_errors: dict[str, int] = {}
        # IMP-NEW-3: track which exception class caused each provider trip so that
        # short-circuited items re-raise the true cause rather than always RateLimitError.
        tripped_provider_cause: dict[str, type] = {}

        logger.info(
            "Starting enrichment for KB %s: %d items (incremental=%s, retry_failed=%s, multimodal=%s)",
            self.kb_id,
            total,
            incremental,
            retry_failed,
            use_multimodal,
        )

        async def _enrich_one(item):
            import litellm.exceptions as llme

            async with sem:
                if provider in tripped_providers:
                    # IMP-NEW-3: re-raise the true cause so _classify_enrichment_failure
                    # (and callers) can distinguish quota from provider-outage errors.
                    cause_cls = tripped_provider_cause.get(provider, llme.RateLimitError)
                    msg = (
                        f"Circuit breaker tripped for {provider} after "
                        f"{_CIRCUIT_BREAKER_THRESHOLD} consecutive errors. "
                        f"Resolve the upstream issue and re-run."
                    )
                    try:
                        raise cause_cls(msg, model=model, llm_provider=provider)
                    except TypeError:
                        raise cause_cls(msg)  # noqa: B904
                item_images = None
                if use_multimodal and source_image_map:
                    item_images = (
                        self._match_images_for_item(item, strategy, source_image_map) or None
                    )
                try:
                    return await self.enrich_single_item(
                        item["text"],
                        fields,
                        model,
                        max_tokens=max_tokens,
                        images=item_images,
                    )
                except (
                    # Important 1 from PR #440 review: broadened from (RateLimitError,
                    # AuthenticationError) to include infrastructure/service errors.
                    llme.RateLimitError,
                    llme.AuthenticationError,
                    llme.ServiceUnavailableError,
                    llme.InternalServerError,
                    llme.BadGatewayError,
                    llme.APIConnectionError,
                    llme.Timeout,
                ) as _trip_exc:
                    # Update circuit-breaker counter while still holding the semaphore
                    # so that tasks waiting at `async with sem:` see the tripped state.
                    consecutive_errors[provider] = consecutive_errors.get(provider, 0) + 1
                    if (
                        consecutive_errors[provider] >= _CIRCUIT_BREAKER_THRESHOLD
                        and provider not in tripped_providers
                    ):
                        tripped_providers.add(provider)
                        # IMP-NEW-3: store the trip-causing exception class for re-raise
                        tripped_provider_cause[provider] = type(_trip_exc)
                        logger.error(
                            "Circuit breaker tripped for provider %s after %d "
                            "consecutive errors; remaining items will short-circuit. "
                            "KB %s, model %s",
                            provider,
                            consecutive_errors[provider],
                            self.kb_id,
                            model,
                        )
                        # IMP-NEW-1: split alert by root cause so on-call follows
                        # the right runbook (quota/billing vs provider outage).
                        # CloudWatch metric filter: alert tag value → SNS → PagerDuty.
                        _quota_causes = (llme.RateLimitError, llme.AuthenticationError)
                        if isinstance(_trip_exc, _quota_causes):
                            logger.error(
                                "PLATFORM_LLM_QUOTA_EXHAUSTED provider=%s model=%s kb=%s "
                                "consecutive_errors=%d — investigate quota / spend cap / "
                                "billing on the platform LLM account.",
                                provider,
                                model,
                                self.kb_id,
                                consecutive_errors[provider],
                                extra={"alert": "platform_llm_quota_exhausted"},
                            )
                        else:
                            logger.error(
                                "PLATFORM_LLM_PROVIDER_DEGRADED provider=%s model=%s kb=%s "
                                "consecutive_errors=%d — investigate upstream provider status "
                                "(likely outage or network issue).",
                                provider,
                                model,
                                self.kb_id,
                                consecutive_errors[provider],
                                extra={"alert": "platform_llm_provider_degraded"},
                            )
                    raise
                except Exception:
                    consecutive_errors[provider] = 0  # reset on non-quota error
                    raise

        for batch_start in range(0, total, METADATA_ENRICHMENT_BATCH_SIZE):
            batch = items[batch_start : batch_start + METADATA_ENRICHMENT_BATCH_SIZE]
            batch_ok_count = 0  # per-batch success counter for on_batch_complete

            # Wrap _enrich_one so the result carries its item identity.
            # as_completed yields futures in completion order, not submission order,
            # so we bundle (item, result, error) to avoid needing zip().
            async def _enrich_with_id(item):
                # Tag this task's llm_call charge with the kb_metadata.item_id
                # of the chunk / node / document being enriched. asyncio.create_task
                # snapshots context per-task, so the run_scope wrapper here
                # only affects this task and is reset before the task ends.
                try:
                    with run_scope(f"kb_metadata:{item['id']}"):
                        result = await _enrich_one(item)
                    return (item, result, None)
                except BaseException as e:  # noqa: BLE001 — intentional broad catch
                    return (item, None, e)

            # Stream results as each LLM call completes — O(1) peak retained-result
            # memory instead of O(batch_size) with gather().
            tasks = [asyncio.create_task(_enrich_with_id(item)) for item in batch]
            for coro in asyncio.as_completed(tasks):
                item, result, error = await coro
                if error is not None:
                    error_msg = str(error)[:500]
                    errors.append(f"Item {item['id']}: {error_msg}")
                    logger.warning("Enrichment failed for item %s: %s", item["id"], error)
                    # Store row with NULL values + error message
                    null_values = {f["name"]: None for f in fields}
                    self.store_result(
                        table_name=table_name,
                        item_id=item["id"],
                        item_type=item["item_type"],
                        values=null_values,
                        error=error_msg,
                    )
                    continue

                consecutive_errors[provider] = 0  # reset on success
                self.store_result(
                    table_name=table_name,
                    item_id=item["id"],
                    item_type=item["item_type"],
                    values=result,
                    error=None,
                )
                enriched += 1
                batch_ok_count += 1

            self.db_session.commit()

            processed = min(batch_start + len(batch), total)
            logger.info("Enrichment progress: %d/%d", processed, total)

            # Invoke callback after commit so already-completed work is durable.
            if on_batch_complete is not None:
                batch_item_ids = [item["id"] for item in batch]
                decision = on_batch_complete(batch_ok_count, batch_item_ids)
                if decision == "abort":
                    logger.warning(
                        "Enrichment aborted by on_batch_complete callback after batch %d (kb=%s)",
                        batch_start,
                        self.kb_id,
                    )
                    break

        logger.info(
            "Enrichment complete for KB %s: %d/%d items, %d errors",
            self.kb_id,
            enriched,
            total,
            len(errors),
        )
        return {"enriched_count": enriched, "total_count": total, "errors": errors}

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store_result(
        self,
        table_name: str,
        item_id: str,
        item_type: str,
        values: dict,
        error: str | None = None,
    ) -> None:
        """INSERT or UPDATE a metadata row in the per-KB table."""
        qualified = f'"{AI_SCHEMA}"."{table_name}"'

        cols = ['"item_id"', '"item_type"', '"_enrichment_error"']
        val_placeholders = [":item_id", ":item_type", ":_enrichment_error"]
        update_parts = []
        params: dict[str, Any] = {
            "item_id": item_id,
            "item_type": item_type,
            "_enrichment_error": error,
        }

        for col_name, col_val in values.items():
            validate_sql_identifier(col_name)
            cols.append(f'"{col_name}"')
            val_placeholders.append(f":{col_name}")
            update_parts.append(f'"{col_name}" = :{col_name}')
            params[col_name] = col_val

        update_parts.append('"_enrichment_error" = :_enrichment_error')
        update_parts.append("enriched_at = NOW()")
        cols_str = ", ".join(cols)
        vals_str = ", ".join(val_placeholders)
        update_str = ", ".join(update_parts)

        sql = (
            f"INSERT INTO {qualified} ({cols_str}) VALUES ({vals_str}) "
            f"ON CONFLICT (item_id) DO UPDATE SET {update_str}"
        )
        self.db_session.execute(text(sql), params)

    def clear_results(self, table_name: str) -> None:
        """TRUNCATE the per-KB metadata table."""
        qualified = f'"{AI_SCHEMA}"."{table_name}"'
        self.db_session.execute(text(f"TRUNCATE {qualified}"))
        self.db_session.commit()
