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

_HELD_MESSAGE = (
    "This file was extracted as plain text because it was stored under a short "
    "file type, so its text is the raw bytes of the file. Re-extract it before indexing."
)


def _quoted(values) -> str:
    return ", ".join(f"'{v}'" for v in values)


def upgrade():
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
            auto_metadata = COALESCE(auto_metadata, '{{}}'::jsonb)
                            || jsonb_build_object('legacy_file_type', file_type),
            updated_at = NOW()
        WHERE file_type IN ({_quoted(_MIME_TYPES)})
        """
    )


def downgrade():
    pass
