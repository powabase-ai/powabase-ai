"""Give sources stored under a short file type their MIME type.

import-from-storage stored ``pdf``, ``docx``, ``xlsx``, ``xls``, ``pptx``,
``markdown`` and ``text`` instead of MIME types, and the documentation
knowledge base stored ``md``. Extraction looks an extractor up by MIME type,
so every one of those sources was extracted by the plain-text fallback. For
Markdown and plain text that was the right extractor. For a PDF or an Office
file it made the file's raw bytes its "text".

Each such row gets its MIME type, and its old type is kept in
``auto_metadata.legacy_file_type`` so it can still be found. A PDF or Office
source whose extraction completed that way is set to ``attention_required``:
indexing refuses any source that is not ``extracted``, so its bytes cannot be
indexed (and billed as embeddings) until it is re-extracted, which now picks
the right extractor. Rows in any other state keep their status.

Only rows that still carry a short type are touched, so running it again
changes nothing. Downgrade does not restore the short types: they are what
made extraction wrong.

Revision ID: 0034
Revises: 0033
Create Date: 2026-09-30
"""

from alembic import op

revision = "0034"
down_revision = "0033"
branch_labels = None
depends_on = None

_MIME_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "markdown": "text/markdown",
    "text": "text/plain",
    "md": "text/markdown",
}
_BINARY = ("pdf", "docx", "xlsx", "xls", "pptx")

# What the text extractors record as extraction_method. None of the PDF or
# Office extractors records one of these.
_TEXT_METHODS = ("text", "txt-native", "markdown-native", "html")

_HELD_MESSAGE = (
    "This file was extracted as plain text because it was stored under a "
    "file type no extractor handles, so its text is the raw bytes of the file. "
    "Re-extract it before indexing."
)

# auto_metadata as an object: a JSON null would otherwise be concatenated into
# an array, which every reader of auto_metadata then fails on.
_METADATA = (
    "CASE WHEN jsonb_typeof(auto_metadata) = 'object' THEN auto_metadata ELSE '{}'::jsonb END"
)


def _quoted(values) -> str:
    return ", ".join(f"'{v}'" for v in values)


def upgrade():
    # 1. Short types, whatever the path: all of them get their MIME type, and a
    #    binary one that completed extraction is held -- it can only have been
    #    extracted by the text fallback.
    mime_case = " ".join(f"WHEN '{short}' THEN '{mime}'" for short, mime in _MIME_TYPES.items())
    held = f"file_type IN ({_quoted(_BINARY)}) AND extraction_status = 'extracted'"
    # Every right-hand side reads the row as it was before this UPDATE.
    op.execute(
        f"""
        UPDATE ai.sources
        SET file_type = CASE file_type {mime_case} END,
            extraction_status = CASE WHEN {held} THEN 'attention_required'
                                     ELSE extraction_status END,
            error_message = CASE WHEN {held} THEN '{_HELD_MESSAGE}'
                                 ELSE error_message END,
            auto_metadata = {_METADATA}
                            || jsonb_build_object('legacy_file_type', file_type)
                            || CASE WHEN {held} THEN '{{"reextract_hold": true}}'::jsonb
                                    ELSE '{{}}'::jsonb END,
            updated_at = NOW()
        WHERE file_type IN ({_quoted(_MIME_TYPES)})
        """
    )

    # 2. Files with a PDF or Office extension stored under any other type than
    #    their MIME type -- an upload that kept application/octet-stream, or a
    #    declared type no extractor is registered under -- get it, so that a
    #    re-extract picks the real extractor. Any such file whose extraction
    #    completed with a text extractor is held: its text is its bytes.
    extension = "lower(substring(storage_path from '\\.([A-Za-z0-9]+)$'))"
    expected = (
        f"CASE {extension} "
        + " ".join(f"WHEN '{ext}' THEN '{_MIME_TYPES[ext]}'" for ext in _BINARY)
        + " END"
    )
    held = (
        "extraction_status = 'extracted' AND "
        f"auto_metadata->>'extraction_method' IN ({_quoted(_TEXT_METHODS)})"
    )
    retyped = f"file_type IS DISTINCT FROM {expected}"
    op.execute(
        f"""
        UPDATE ai.sources
        SET file_type = {expected},
            extraction_status = CASE WHEN {held} THEN 'attention_required'
                                     ELSE extraction_status END,
            error_message = CASE WHEN {held} THEN '{_HELD_MESSAGE}'
                                 ELSE error_message END,
            auto_metadata = {_METADATA}
                            || CASE WHEN {retyped}
                                    THEN jsonb_build_object('legacy_file_type', file_type)
                                    ELSE '{{}}'::jsonb END
                            || CASE WHEN {held} THEN '{{"reextract_hold": true}}'::jsonb
                                    ELSE '{{}}'::jsonb END,
            updated_at = NOW()
        WHERE {extension} IN ({_quoted(_BINARY)})
          AND ({retyped} OR ({held}))
        """
    )


def downgrade():
    pass
