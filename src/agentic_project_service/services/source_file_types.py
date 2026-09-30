"""The ``file_type`` a source is stored under, derived from its file name.

Extraction picks its extractor by looking ``file_type`` up as a MIME type
(``agentic.ingest.ExtractorRegistry``), so every route that creates a file
source must store a MIME type the registry knows -- and the same one for the
same file, whichever route it arrived by. This module is that one place.

The extension decides, not a client-declared Content-Type: the extension is
what the routes validate against their allowlists, and a declared type is
often useless (``application/octet-stream`` for a PDF or a Word file sends
it to the plain-text extractor, which "extracts" the raw bytes).
"""

from __future__ import annotations

import os

# Documents: accepted by both /upload and /import-from-storage.
DOCUMENT_MIME_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

# Images: accepted by /upload only.
IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".tiff": "image/tiff",
}

UPLOAD_MIME_TYPES: dict[str, str] = {**DOCUMENT_MIME_TYPES, **IMAGE_MIME_TYPES}
IMPORT_MIME_TYPES: dict[str, str] = DOCUMENT_MIME_TYPES

# The short names stored before every route used MIME types: import-from-storage
# wrote all but "md", which the documentation knowledge base wrote. Migration
# 0034 rewrites stored rows; extraction still maps any it meets.
LEGACY_SHORT_FILE_TYPES: dict[str, str] = {
    "pdf": DOCUMENT_MIME_TYPES[".pdf"],
    "text": DOCUMENT_MIME_TYPES[".txt"],
    "markdown": DOCUMENT_MIME_TYPES[".md"],
    "md": DOCUMENT_MIME_TYPES[".md"],
    "docx": DOCUMENT_MIME_TYPES[".docx"],
    "xlsx": DOCUMENT_MIME_TYPES[".xlsx"],
    "xls": DOCUMENT_MIME_TYPES[".xls"],
    "pptx": DOCUMENT_MIME_TYPES[".pptx"],
}


def file_extension(name: str) -> str:
    """The lower-cased extension of *name*, with its dot ("" when none)."""
    return os.path.splitext(name)[1].lower()


def source_mime_type(name: str, allowed: dict[str, str]) -> str | None:
    """The MIME type a file called *name* is stored under, or None when its
    extension is not one of *allowed* (a route's allowlist above)."""
    return allowed.get(file_extension(name))


def extraction_mime_type(file_type: str) -> str:
    """*file_type* as the MIME type extraction routes by: a legacy short name
    is mapped to its MIME type, anything else is returned unchanged."""
    return LEGACY_SHORT_FILE_TYPES.get(file_type, file_type)
