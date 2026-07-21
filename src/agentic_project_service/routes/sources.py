"""Source management routes for the project service."""

import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import uuid

import defusedxml.ElementTree as ET
from urllib.parse import urlparse, urlunparse

import httpx
from flask import Blueprint, Response, jsonify, request
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from ..auth import require_auth
from ..celery import celery_app
from ..db import db, AI_SCHEMA
from ..services.ai_provider_keys_resolver import get_all_user_provider_keys
from ..services import billing_port as billing
from ..services.settings_registry import EXTRACTION_METHOD_CHOICES, get_setting
from ..services.storage import (
    StorageError,
    get_source_storage_path,
    get_storage,
    SOURCES_BUCKET,
)
from ..tasks.extraction import extract_source
from ..tasks.url_extraction import extract_url_source

logger = logging.getLogger(__name__)


# Per-page estimate for pre-charge balance check at queue time. OCR is billed
# at ocr_pages × N_pages where N_pages is unknown until the file is decoded
# server-side; estimate conservative upper bound so we reject pre-flight when
# the org's balance can't cover the file. Worker post_charges the actual count.
#
# Conservative upper bound for upload-time pre-check. Bumped from 10 → 100
# because the page count isn't known until extraction runs; a small estimate
# silently leaks for any PDF > ~10 pages. Per-file probing (open the file at
# upload, count pages, refine estimate) is a deeper change tracked separately.
#
# UX tradeoff (F4 from calibration review): with _OCR_UNIT_CREDITS=3 (catalog
# in 0006_credit_ledger.py:204), the pre-check now requires 300 credits for
# any OCR upload. Free-tier customers with <300 credits will be 503'd for
# even a 1-page upload that actually costs 3 credits. Accepted tradeoff
# until per-file page-count probing lands.
# NOTE: these per-page unit-credit constants are a coarse mirror of the
# platform.credit_pricing catalog, used ONLY for the upload-time balance gate
# (pre-check), never for the actual debit. They intentionally do not hot-load
# the catalog (that would add a fetch per upload). If the catalog migration
# changes a price, bump the matching constant here too.
_EXTRACTION_ESTIMATED_PAGES: int = 100
_OCR_UNIT_CREDITS: int = 3  # ocr_pages cost per page in catalog
# advanced_ocr (LlamaParse) is ~18.75× ocr_pages (11250 vs 600 millicents/page
# in the catalog). Mirror _OCR_UNIT_CREDITS's pre-check denomination scaled by
# that ratio so the balance gate stays proportional to the real per-page cost.
_ADVANCED_OCR_UNIT_CREDITS: int = 57
_WEB_SCRAPE_UNIT_CREDITS: int = 5  # web_scrape cost per page in catalog


def _extraction_billing_action(extraction_model: str | None) -> str:
    """Map a chosen extraction model to its billing action.

    LlamaParse bills against the separate ``advanced_ocr`` catalog category;
    every other cloud-OCR method bills against ``ocr_pages``.

    This drives the upload-time pre-charge *balance gate*, which is keyed on the
    *requested* model and is therefore intentionally pessimistic: if LlamaParse
    is requested but later falls back to a cheaper engine (e.g. an image falling
    back to Mistral), the gate reserves at the advanced-OCR rate while the worker
    only ever debits the *actual* method's rate. The gate is a balance check, not
    a debit, so net billing is correct — a low-balance user may just be blocked
    from an upload they could afford under fallback.
    """
    return "advanced_ocr" if extraction_model == "llamaparse" else "ocr_pages"


def _estimated_extraction_cost(action: str) -> int:
    """Pre-charge cost estimate for an extraction/scrape action (a conservative
    upper bound: worst-case pages × the per-page unit-credit rate for *action*).

    Pure function of the action — it feeds the upload-time balance gate
    (``billing.check_balance``), which routes through the billing port and
    no-ops in the OSS build / when billing is unconfigured. The worker charges
    the actual page count separately; this is only the pre-flight reservation.
    Identity (org/project) and the idempotency key are no longer computed here —
    the worker recomputes the key from its own args via ``billing.charge``.
    """
    if action == "ocr_pages":
        per_unit = _OCR_UNIT_CREDITS
    elif action == "advanced_ocr":
        per_unit = _ADVANCED_OCR_UNIT_CREDITS
    else:
        per_unit = _WEB_SCRAPE_UNIT_CREDITS
    return _EXTRACTION_ESTIMATED_PAGES * per_unit


sources_bp = Blueprint("sources", __name__, url_prefix="/api/sources")


_CONTENT_HASH_UNIQUE_INDEX = "sources_content_hash_uniq"


def _safe_cleanup_storage(storage, bucket_id: str, path: str) -> None:
    """Best-effort delete of an uploaded storage object."""
    try:
        storage.delete(bucket_id, [path])
    except Exception:
        logger.exception("Failed to clean up storage object %s/%s", bucket_id, path)


def _safe_rollback() -> None:
    """Best-effort SQLAlchemy session rollback."""
    try:
        db.session.rollback()
    except Exception:
        logger.exception("db.session.rollback() failed during 500 handler")


def _safe_delete_source_row(source_id: str) -> None:
    """Best-effort delete of a committed source row whose downstream
    extract_source dispatch / celery_task_id UPDATE blew up.
    """
    try:
        db.session.execute(
            text(f'DELETE FROM "{AI_SCHEMA}".sources WHERE id = :id'),
            {"id": source_id},
        )
        db.session.commit()
    except Exception:
        logger.exception("Failed to delete phantom source row %s", source_id)
        try:
            db.session.rollback()
        except Exception:
            logger.exception("Rollback after _safe_delete_source_row failure also failed")


def _insert_source_with_dedup(
    *,
    storage,
    source_id: str,
    name: str | None,
    file_type: str,
    full_path: str,
    storage_path: str,
    metadata: dict,
    auto_metadata: dict,
    content_hash: str,
) -> bool:
    """INSERT a source row with race recovery on the content_hash unique
    index. Returns True on commit, False if a concurrent INSERT won the
    race (in which case the storage object is already cleaned up and the
    caller should return ``_duplicate_response(content_hash)``). Re-raises
    any non-dedup IntegrityError.
    """
    try:
        db.session.execute(
            text(f"""
                INSERT INTO "{AI_SCHEMA}".sources (
                    id, name, file_type, storage_path, extraction_status,
                    metadata, auto_metadata, content_hash
                ) VALUES (
                    :id, :name, :file_type, :storage_path, 'pending',
                    CAST(:metadata AS jsonb), CAST(:auto_metadata AS jsonb), :content_hash
                )
            """),
            {
                "id": source_id,
                "name": name,
                "file_type": file_type,
                "storage_path": full_path,
                "metadata": json.dumps(metadata),
                "auto_metadata": json.dumps(auto_metadata),
                "content_hash": content_hash,
            },
        )
        db.session.commit()
        return True
    except IntegrityError as e:
        db.session.rollback()
        constraint_name = getattr(getattr(e.orig, "diag", None), "constraint_name", None)
        if constraint_name != _CONTENT_HASH_UNIQUE_INDEX:
            raise
        _safe_cleanup_storage(storage, SOURCES_BUCKET, storage_path)
        return False


def _duplicate_response(content_hash: str):
    """Build the 409 body for a duplicate-content upload."""
    row = db.session.execute(
        text(f"""
            SELECT id, name, file_type, extraction_status, created_at
            FROM "{AI_SCHEMA}".sources
            WHERE content_hash = :h
            LIMIT 1
        """),
        {"h": content_hash},
    ).fetchone()
    logger.info("duplicate_source rejected upload (existing_id=%s)", row.id)
    return jsonify(
        {
            "error": "duplicate_source",
            "message": "A source with identical content already exists in this project.",
            "duplicate": {
                "id": str(row.id),
                "name": row.name,
                "file_type": row.file_type,
                "extraction_status": row.extraction_status,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            },
        }
    ), 409


@sources_bp.route("", methods=["GET"])
@require_auth
def list_sources():
    """List sources, paginated, with optional status filter and name search."""
    from ..services.list_params import parse_list_params, escape_like, ListParamsError

    status = request.args.get("status")

    try:
        limit, offset, q, sort, order = parse_list_params(
            request,
            sort_allowed={"created_at", "name"},
        )
    except ListParamsError as e:
        return jsonify({"error": str(e)}), e.status

    where_clauses = []
    params: dict = {"limit": limit, "offset": offset}
    if status:
        where_clauses.append("extraction_status = :status")
        params["status"] = status
    if q:
        where_clauses.append("name ILIKE :q_like")
        params["q_like"] = f"%{escape_like(q)}%"
    where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    order_by = f"{sort} {order.upper()}, id ASC"

    query = f"""
        SELECT id, name, file_type, storage_path, extraction_status,
               derivatives, metadata, auto_metadata, error_message,
               created_at, updated_at
        FROM "{AI_SCHEMA}".sources
        {where_clause}
        ORDER BY {order_by}
        LIMIT :limit OFFSET :offset
    """

    result = db.session.execute(text(query), params)

    sources = []
    for row in result:
        sources.append(
            {
                "id": str(row[0]),
                "name": row[1],
                "file_type": row[2],
                "storage_path": row[3],
                "extraction_status": row[4],
                "derivatives": row[5],
                "metadata": row[6],
                "auto_metadata": row[7],
                "error_message": row[8],
                "created_at": row[9].isoformat() if row[9] else None,
                "updated_at": row[10].isoformat() if row[10] else None,
            }
        )

    count_query = f'SELECT COUNT(*) FROM "{AI_SCHEMA}".sources {where_clause}'
    total = db.session.execute(text(count_query), params).scalar()

    return jsonify({"sources": sources, "total": total, "limit": limit, "offset": offset})


@sources_bp.route("/upload", methods=["POST"])
@require_auth
def upload_source():
    """Upload a new source file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No filename"}), 400

    name = request.form.get("name", file.filename)
    metadata = {}
    if request.form.get("metadata"):
        try:
            metadata = json.loads(request.form["metadata"])
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid metadata JSON"}), 400

    # Extraction model preference (PDF only — ignored for other types)
    VALID_EXTRACTION_MODELS = set(EXTRACTION_METHOD_CHOICES)
    extraction_model = request.form.get("extraction_model")
    if extraction_model and extraction_model not in VALID_EXTRACTION_MODELS:
        return jsonify(
            {
                "error": f"Invalid extraction_model '{extraction_model}'. "
                f"Valid options: {', '.join(sorted(VALID_EXTRACTION_MODELS))}"
            }
        ), 400
    # Validate file extension
    ALLOWED_EXTENSIONS = {
        ".pdf",
        ".txt",
        ".md",
        ".docx",
        ".xlsx",
        ".xls",
        ".pptx",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".tiff",
    }
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify(
            {
                "error": f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            }
        ), 400

    file_type = file.content_type or "application/octet-stream"

    # Browsers may send application/octet-stream for certain file types;
    # the extraction pipeline routes by MIME type, so we must correct these.
    _EXT_MIME_OVERRIDES = {
        ".md": "text/markdown",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".tiff": "image/tiff",
    }
    if file_type == "application/octet-stream" and ext in _EXT_MIME_OVERRIDES:
        file_type = _EXT_MIME_OVERRIDES[ext]

    source_id = str(uuid.uuid4())
    uploaded_storage_path: str | None = None
    source_row_committed = False
    storage = None

    # Pre-charge balance check (free-tier hard cap) BEFORE storage upload.
    # The billing port no-ops the check when billing is unconfigured.
    billing.check_balance(
        estimated_cost=_estimated_extraction_cost(_extraction_billing_action(extraction_model))
    )

    try:
        storage = get_storage()
        storage.ensure_bucket(SOURCES_BUCKET)

        storage_path = get_source_storage_path(source_id, file.filename)
        file_data = file.read()
        content_hash = hashlib.sha256(file_data).hexdigest()

        existing = db.session.execute(
            text(f'SELECT 1 FROM "{AI_SCHEMA}".sources WHERE content_hash = :h LIMIT 1'),
            {"h": content_hash},
        ).fetchone()
        if existing is not None:
            return _duplicate_response(content_hash)

        uploaded_storage_path = storage_path
        full_path = storage.upload(
            bucket_id=SOURCES_BUCKET,
            path=storage_path,
            file_data=file_data,
            content_type=file_type,
        )

        auto_metadata: dict = {"origin_path": file.filename}
        if extraction_model:
            auto_metadata["extraction_model"] = extraction_model
        committed = _insert_source_with_dedup(
            storage=storage,
            source_id=source_id,
            name=name,
            file_type=file_type,
            full_path=full_path,
            storage_path=storage_path,
            metadata=metadata,
            auto_metadata=auto_metadata,
            content_hash=content_hash,
        )
        if not committed:
            uploaded_storage_path = None
            return _duplicate_response(content_hash)
        source_row_committed = True

        task = extract_source.delay(
            source_id,
            SOURCES_BUCKET,
            extraction_model=extraction_model,
            provider_keys=get_all_user_provider_keys(),
        )

        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".sources
                SET celery_task_id = :task_id
                WHERE id = :id
            """),
            {"task_id": task.id, "id": source_id},
        )
        db.session.commit()

        return jsonify(
            {
                "id": source_id,
                "name": name,
                "file_type": file_type,
                "storage_path": full_path,
                "extraction_status": "pending",
                "task_id": task.id,
            }
        ), 201

    except StorageError as e:
        logger.error("Storage error during upload: source_id=%s err=%s", source_id, e)
        if uploaded_storage_path and storage is not None:
            _safe_cleanup_storage(storage, SOURCES_BUCKET, uploaded_storage_path)
        _safe_rollback()
        if source_row_committed:
            _safe_delete_source_row(source_id)
        return jsonify({"error": "Storage error"}), 500
    except Exception:
        logger.exception("Upload failed: source_id=%s", source_id)
        if uploaded_storage_path and storage is not None:
            _safe_cleanup_storage(storage, SOURCES_BUCKET, uploaded_storage_path)
        _safe_rollback()
        if source_row_committed:
            _safe_delete_source_row(source_id)
        return jsonify({"error": "Internal error"}), 500


def _normalize_url(url: str) -> str:
    """Normalize a URL for deduplication: strip fragments, trailing slashes."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, ""))


def _validate_url(url: str) -> bool:
    """Check that a string is a valid http/https URL and not an internal address."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return False
        # Resolve all addresses (IPv4 + IPv6) and reject if any are private
        addrinfos = socket.getaddrinfo(parsed.hostname, None)
        for family, _, _, _, sockaddr in addrinfos:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
    except socket.gaierror:
        # DNS resolution failed — reject the URL
        return False
    except Exception:
        return False
    return True


def _create_url_source(url: str) -> tuple[str, str]:
    """Create a source DB record for a URL and return (source_id, name)."""
    source_id = str(uuid.uuid4())
    name = url
    storage_path = f"{SOURCES_BUCKET}/{get_source_storage_path(source_id, 'page.html')}"
    metadata: dict = {}
    auto_metadata = {"origin_url": url, "source_type": "url"}

    db.session.execute(
        text(f"""
            INSERT INTO "{AI_SCHEMA}".sources (
                id, name, file_type, storage_path, extraction_status,
                metadata, auto_metadata
            ) VALUES (
                :id, :name, 'text/html', :storage_path, 'pending',
                CAST(:metadata AS jsonb), CAST(:auto_metadata AS jsonb)
            )
        """),
        {
            "id": source_id,
            "name": name,
            "storage_path": storage_path,
            "metadata": json.dumps(metadata),
            "auto_metadata": json.dumps(auto_metadata),
        },
    )
    return source_id, name


def _dispatch_url_extraction(
    source_id: str, url: str, provider_keys: dict[str, str] | None = None
) -> str:
    """Dispatch the URL extraction task and update the source with task ID."""
    task = extract_url_source.delay(
        source_id,
        SOURCES_BUCKET,
        url,
        provider_keys=provider_keys,
    )
    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".sources
            SET celery_task_id = :task_id
            WHERE id = :id
        """),
        {"task_id": task.id, "id": source_id},
    )
    return task.id


def _discover_urls_crawl(url: str, max_pages: int) -> list[str]:
    """Discover URLs via Firecrawl /v1/map, falling back to HTML link parsing.

    Assumes FIRECRAWL_API_KEY is set — the only public caller
    (``import_url``) pre-checks at the route boundary and returns 503 on
    missing env, so reaching this function with an empty key would require
    the env to disappear mid-request (functionally impossible). If the key
    is empty here, Firecrawl returns 401 and we fall back to HTML link
    extraction — graceful, but a sign the pre-check upstream needs review.
    """
    firecrawl_key = os.environ.get("FIRECRAWL_API_KEY", "")
    firecrawl_base = get_setting("FIRECRAWL_API_BASE").rstrip("/")

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{firecrawl_base}/map",
                headers={"Authorization": f"Bearer {firecrawl_key}"},
                json={"url": url, "limit": max_pages},
            )
            resp.raise_for_status()
            links = resp.json().get("links", [])
            if links:
                return [u for u in links if _validate_url(u)][:max_pages]
    except Exception:
        logger.warning(
            "Firecrawl /v1/map failed, falling back to HTML link extraction", exc_info=True
        )

    # Fallback: scrape page HTML and extract <a href>
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text
        # Extract href values
        href_re = re.compile(r'<a\s[^>]*href=["\']([^"\']+)["\']', re.IGNORECASE)
        base_parsed = urlparse(url)
        found: list[str] = []
        seen: set[str] = set()
        for href in href_re.findall(html):
            # Resolve relative URLs
            if href.startswith("/"):
                href = f"{base_parsed.scheme}://{base_parsed.netloc}{href}"
            if not _validate_url(href):
                continue
            norm = _normalize_url(href)
            if norm not in seen:
                seen.add(norm)
                found.append(href)
            if len(found) >= max_pages:
                break
        return found
    except Exception:
        logger.warning("HTML link fallback also failed", exc_info=True)
        return [url]  # At minimum, import the original URL


def _discover_urls_sitemap(url: str, max_pages: int) -> list[str]:
    """Parse a sitemap XML (or sitemap index) and return discovered URLs."""
    urls: list[str] = []
    seen: set[str] = set()

    def _parse_sitemap(sitemap_url: str, depth: int = 0) -> None:
        if len(urls) >= max_pages or depth > 5:
            return
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(sitemap_url, follow_redirects=True)
                resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception:
            logger.warning(f"Failed to fetch/parse sitemap: {sitemap_url}", exc_info=True)
            return

        ns = ""
        # Handle default XML namespace
        tag = root.tag
        if tag.startswith("{"):
            ns = tag.split("}")[0] + "}"

        # Check if sitemap index
        for sitemap_el in root.findall(f"{ns}sitemap"):
            loc_el = sitemap_el.find(f"{ns}loc")
            if loc_el is not None and loc_el.text:
                _parse_sitemap(loc_el.text.strip(), depth + 1)
                if len(urls) >= max_pages:
                    return

        # Regular sitemap — extract <loc> entries
        for url_el in root.findall(f"{ns}url"):
            loc_el = url_el.find(f"{ns}loc")
            if loc_el is not None and loc_el.text:
                page_url = loc_el.text.strip()
                norm = _normalize_url(page_url)
                if norm not in seen and _validate_url(page_url):
                    seen.add(norm)
                    urls.append(page_url)
                if len(urls) >= max_pages:
                    return

    _parse_sitemap(url)
    return urls


@sources_bp.route("/import-url", methods=["POST"])
@require_auth
def import_url():
    """Import sources from URLs (list, crawl, or sitemap)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    mode = data.get("mode")
    if mode not in ("urls", "crawl", "sitemap"):
        return jsonify({"error": "mode must be one of: urls, crawl, sitemap"}), 400

    # Firecrawl key is platform-injected; missing means operator misconfiguration.
    firecrawl_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not firecrawl_key:
        logger.error("FIRECRAWL_API_KEY missing from pod env — platform misconfiguration")
        return jsonify(
            {"error": "URL import is currently unavailable. Please try again later."}
        ), 503

    max_pages_setting = get_setting("URL_IMPORT_MAX_PAGES")
    try:
        max_pages = max(1, min(int(data.get("max_pages") or max_pages_setting), max_pages_setting))
    except (ValueError, TypeError):
        max_pages = max_pages_setting

    try:
        if mode == "urls":
            raw_urls = data.get("urls")
            if not raw_urls or not isinstance(raw_urls, list):
                return jsonify({"error": "urls must be a non-empty list"}), 400
            # Validate, deduplicate, cap
            seen: set[str] = set()
            urls: list[str] = []
            for u in raw_urls:
                if not isinstance(u, str) or not _validate_url(u):
                    continue
                norm = _normalize_url(u)
                if norm not in seen:
                    seen.add(norm)
                    urls.append(u)
                if len(urls) >= max_pages:
                    break
            if not urls:
                return jsonify({"error": "No valid URLs provided"}), 400

        elif mode == "crawl":
            url = data.get("url")
            if not url or not _validate_url(url):
                return jsonify({"error": "A valid url is required for crawl mode"}), 400
            urls = _discover_urls_crawl(url, max_pages)

        elif mode == "sitemap":
            url = data.get("url")
            if not url or not _validate_url(url):
                return jsonify({"error": "A valid sitemap url is required"}), 400
            urls = _discover_urls_sitemap(url, max_pages)
            if not urls:
                return jsonify({"error": "No URLs found in sitemap"}), 400

        # Pre-charge balance check covering ALL URLs about to be scraped.
        # Fail fast (402/503) BEFORE inserting any source rows so a failed
        # check doesn't leave orphan pending sources.
        billing.check_balance(estimated_cost=len(urls) * _WEB_SCRAPE_UNIT_CREDITS)

        # Create source records and dispatch extraction tasks
        sources_created = []
        for u in urls:
            source_id, name = _create_url_source(u)
            sources_created.append({"id": source_id, "name": name, "url": u})

        db.session.commit()

        # Dispatch tasks after commit so source records exist
        provider_keys = get_all_user_provider_keys()
        for src in sources_created:
            _dispatch_url_extraction(src["id"], src["url"], provider_keys)

        db.session.commit()

        return jsonify({"sources": sources_created, "count": len(sources_created)}), 201

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("URL import failed")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@sources_bp.route("/import-from-storage", methods=["POST"])
@require_auth
def import_from_storage():
    """Create a source record from a file already in storage (no re-upload)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    bucket = data.get("bucket")
    path = data.get("path")
    if not bucket or not path:
        return jsonify({"error": "bucket and path are required"}), 400

    # Determine file_type from extension
    EXTENSION_MAP = {
        ".pdf": "pdf",
        ".md": "markdown",
        ".txt": "text",
        ".docx": "docx",
        ".xlsx": "xlsx",
        ".xls": "xls",
        ".pptx": "pptx",
    }
    ext = os.path.splitext(path)[1].lower()
    if ext not in EXTENSION_MAP:
        return jsonify(
            {
                "error": f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(EXTENSION_MAP.keys()))}"
            }
        ), 400
    file_type = EXTENSION_MAP[ext]

    # Derive name from body or filename
    filename = os.path.basename(path)
    name = data.get("name") or os.path.splitext(filename)[0]

    # Extraction model preference
    VALID_EXTRACTION_MODELS = set(EXTRACTION_METHOD_CHOICES)
    extraction_model = data.get("extraction_model", "auto")
    if extraction_model not in VALID_EXTRACTION_MODELS:
        return jsonify(
            {
                "error": f"Invalid extraction_model '{extraction_model}'. "
                f"Valid options: {', '.join(sorted(VALID_EXTRACTION_MODELS))}"
            }
        ), 400

    source_id = str(uuid.uuid4())
    metadata: dict = {}
    uploaded_storage_path: str | None = None
    source_row_committed = False
    storage = None

    # MIME type mapping for content-type when re-uploading
    MIME_MAP = {
        "pdf": "application/pdf",
        "markdown": "text/markdown",
        "text": "text/plain",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls": "application/vnd.ms-excel",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }

    # Pre-charge balance check before any storage work. The billing port no-ops
    # the check when billing is unconfigured.
    billing.check_balance(
        estimated_cost=_estimated_extraction_cost(_extraction_billing_action(extraction_model))
    )

    try:
        storage = get_storage()
        storage.ensure_bucket(SOURCES_BUCKET)

        try:
            file_data = storage.download(bucket, path)
        except StorageError as e:
            logger.warning(
                "Import from storage: download failed bucket=%s path=%s err=%s",
                bucket,
                path,
                e,
            )
            return jsonify({"error": f"File not found in storage: {bucket}/{path}"}), 404

        content_hash = hashlib.sha256(file_data).hexdigest()

        existing = db.session.execute(
            text(f'SELECT 1 FROM "{AI_SCHEMA}".sources WHERE content_hash = :h LIMIT 1'),
            {"h": content_hash},
        ).fetchone()
        if existing is not None:
            return _duplicate_response(content_hash)

        # Re-upload to managed sources bucket (same structure as upload endpoint)
        storage_dest = get_source_storage_path(source_id, filename)
        content_type = MIME_MAP.get(file_type, "application/octet-stream")
        uploaded_storage_path = storage_dest
        full_path = storage.upload(
            bucket_id=SOURCES_BUCKET,
            path=storage_dest,
            file_data=file_data,
            content_type=content_type,
        )

        # Create DB record pointing to sources bucket (not user bucket)
        committed = _insert_source_with_dedup(
            storage=storage,
            source_id=source_id,
            name=name,
            file_type=file_type,
            full_path=full_path,
            storage_path=storage_dest,
            metadata=metadata,
            auto_metadata={
                "origin_path": path,
                "extraction_model": extraction_model,
                "imported_from_storage": True,
                "origin_bucket": bucket,
            },
            content_hash=content_hash,
        )
        if not committed:
            uploaded_storage_path = None
            return _duplicate_response(content_hash)
        source_row_committed = True

        # Trigger extraction from sources bucket (not user bucket)
        task = extract_source.delay(
            source_id,
            SOURCES_BUCKET,
            extraction_model=extraction_model,
            provider_keys=get_all_user_provider_keys(),
        )

        # Update with task ID
        db.session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".sources
                SET celery_task_id = :task_id
                WHERE id = :id
            """),
            {"task_id": task.id, "id": source_id},
        )
        db.session.commit()

        return jsonify(
            {
                "id": source_id,
                "name": name,
                "file_type": file_type,
                "storage_path": full_path,
                "extraction_status": "pending",
                "task_id": task.id,
            }
        ), 201

    except StorageError as e:
        logger.error("Storage error during import: source_id=%s err=%s", source_id, e)
        if uploaded_storage_path and storage is not None:
            _safe_cleanup_storage(storage, SOURCES_BUCKET, uploaded_storage_path)
        _safe_rollback()
        if source_row_committed:
            _safe_delete_source_row(source_id)
        return jsonify({"error": "Storage error"}), 500
    except Exception:
        logger.exception("Import from storage failed: source_id=%s", source_id)
        if uploaded_storage_path and storage is not None:
            _safe_cleanup_storage(storage, SOURCES_BUCKET, uploaded_storage_path)
        _safe_rollback()
        if source_row_committed:
            _safe_delete_source_row(source_id)
        return jsonify({"error": "Internal error"}), 500


@sources_bp.route("/<source_id>", methods=["GET"])
@require_auth
def get_source(source_id: str):
    """Get a specific source."""
    result = db.session.execute(
        text(f"""
            SELECT id, name, file_type, storage_path, extraction_status,
                   derivatives, metadata, auto_metadata, error_message,
                   celery_task_id, created_at, updated_at
            FROM "{AI_SCHEMA}".sources
            WHERE id = :id
        """),
        {"id": source_id},
    )

    row = result.fetchone()
    if not row:
        return jsonify({"error": "Source not found"}), 404

    return jsonify(
        {
            "id": str(row[0]),
            "name": row[1],
            "file_type": row[2],
            "storage_path": row[3],
            "extraction_status": row[4],
            "derivatives": row[5],
            "metadata": row[6],
            "auto_metadata": row[7],
            "error_message": row[8],
            "celery_task_id": row[9],
            "created_at": row[10].isoformat() if row[10] else None,
            "updated_at": row[11].isoformat() if row[11] else None,
        }
    )


@sources_bp.route("/<source_id>", methods=["PATCH"])
@require_auth
def update_source(source_id: str):
    """Update source metadata."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    updates = []
    params = {"id": source_id}

    if "name" in data:
        updates.append("name = :name")
        params["name"] = data["name"]
    if "metadata" in data:
        # source.metadata is user-owned tags; the three reserved system keys
        # (extraction_model, source_type, source_url) live in auto_metadata
        # as of migration 0015. Strip them on write so a well-meaning client
        # cannot re-pollute metadata by PATCHing {metadata: {extraction_model: ...}}.
        raw_meta = data["metadata"] or {}
        reserved = {"extraction_model", "source_type", "source_url"}
        clean_meta = {k: v for k, v in raw_meta.items() if k not in reserved}
        updates.append("metadata = CAST(:metadata AS jsonb)")
        params["metadata"] = json.dumps(clean_meta)

    if not updates:
        return jsonify({"error": "No valid fields to update"}), 400

    updates.append("updated_at = NOW()")

    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".sources
            SET {", ".join(updates)}
            WHERE id = :id
        """),
        params,
    )
    db.session.commit()

    return get_source(source_id)


@sources_bp.route("/<source_id>", methods=["DELETE"])
@require_auth
def delete_source(source_id: str):
    """Delete a source and its files."""
    # Get source info
    result = db.session.execute(
        text(f"""
            SELECT storage_path, derivatives FROM "{AI_SCHEMA}".sources WHERE id = :id
        """),
        {"id": source_id},
    )
    row = result.fetchone()
    if not row:
        return jsonify({"error": "Source not found"}), 404

    storage_path = row[0]
    derivatives = row[1] or {}

    # Delete from storage
    try:
        storage = get_storage()
        paths_to_delete = []

        if storage_path:
            parts = storage_path.split("/", 1)
            if len(parts) == 2:
                paths_to_delete.append(parts[1])

        for deriv_type, deriv_list in derivatives.items():
            for deriv in deriv_list:
                if deriv.get("storage_path"):
                    parts = deriv["storage_path"].split("/", 1)
                    if len(parts) == 2:
                        paths_to_delete.append(parts[1])

        if paths_to_delete:
            storage.delete(SOURCES_BUCKET, paths_to_delete)

    except StorageError as e:
        logger.warning(f"Failed to delete files: {e}")

    # Check for dependent indexed sources before deleting
    deps_result = db.session.execute(
        text(f"""
            SELECT kb.name FROM "{AI_SCHEMA}".indexed_sources idx
            JOIN "{AI_SCHEMA}".knowledge_bases kb ON idx.knowledge_base_id = kb.id
            WHERE idx.source_id = :source_id
        """),
        {"source_id": source_id},
    )
    dep_names = [row[0] for row in deps_result]

    # Delete from database
    db.session.execute(
        text(f'DELETE FROM "{AI_SCHEMA}".sources WHERE id = :id'),
        {"id": source_id},
    )
    db.session.commit()

    response = {"message": "Source deleted"}
    if dep_names:
        response["warning"] = (
            f"This source was indexed in {len(dep_names)} knowledge base(s): {', '.join(dep_names)}. Those indexes now have missing data and may need re-indexing."
        )
    return jsonify(response)


@sources_bp.route("/<source_id>/download", methods=["GET"])
@require_auth
def download_source(source_id: str):
    """Download the source file."""
    result = db.session.execute(
        text(f'SELECT name, file_type, storage_path FROM "{AI_SCHEMA}".sources WHERE id = :id'),
        {"id": source_id},
    )
    row = result.fetchone()
    if not row:
        return jsonify({"error": "Source not found"}), 404

    name, file_type, storage_path = row

    if not storage_path:
        return jsonify({"error": "No file available"}), 404

    try:
        storage = get_storage()
        safe_name = name.replace('"', "'").replace("\n", " ").replace("\r", " ")
        stream = storage.stream_download_from_path(storage_path)
        content_length = next(stream)  # eagerly connect + validate status
        resp_headers = {"Content-Disposition": f'attachment; filename="{safe_name}"'}
        if content_length:
            resp_headers["Content-Length"] = content_length
        return Response(
            stream,
            mimetype=file_type or "application/octet-stream",
            headers=resp_headers,
        )
    except StorageError as e:
        return jsonify({"error": str(e)}), 500
    except StopIteration:
        return jsonify({"error": "Empty file"}), 500


@sources_bp.route("/<source_id>/page-texts", methods=["GET"])
@require_auth
def get_page_texts(source_id: str):
    """Get per-page texts for a source.

    Supports an optional ``?page=N`` query parameter (1-indexed) for
    single-page retrieval.  When omitted, the full array is returned.
    """
    result = db.session.execute(
        text(f'SELECT derivatives FROM "{AI_SCHEMA}".sources WHERE id = :id'),
        {"id": source_id},
    )
    row = result.fetchone()
    if not row:
        return jsonify({"error": "Source not found"}), 404

    derivatives = row[0] or {}
    pt_derivs = derivatives.get("page_text", [])
    if not pt_derivs:
        page_param = request.args.get("page")
        if page_param is not None:
            return jsonify({"error": "No page texts available"}), 404
        return jsonify({"page_texts": [], "count": 0})

    pt_derivs = sorted(pt_derivs, key=lambda d: d.get("page", 0))

    # Parse optional ?page=N (1-indexed)
    page_param = request.args.get("page")
    requested_page: int | None = None
    if page_param is not None:
        try:
            requested_page = int(page_param)
            if requested_page < 1:
                return jsonify({"error": "page must be >= 1"}), 400
        except ValueError:
            return jsonify({"error": "page must be an integer"}), 400

    storage = get_storage()

    if requested_page is not None:
        target = None
        for d in pt_derivs:
            if d.get("page") == requested_page:
                target = d
                break
        if not target or not target.get("storage_path"):
            return jsonify({"error": f"page {requested_page} not found"}), 404
        parts = target["storage_path"].split("/", 1)
        if len(parts) != 2:
            return jsonify({"error": "Invalid storage path"}), 500
        try:
            raw = storage.download(parts[0], parts[1])
            text_content = raw.decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to download page_text for page {requested_page}: {e}")
            return jsonify({"error": f"Failed to download page {requested_page}"}), 500
        return jsonify({"text": text_content, "page": requested_page, "count": len(pt_derivs)})

    # Return all pages
    page_texts = []
    for d in pt_derivs:
        sp = d.get("storage_path")
        if not sp:
            continue
        parts = sp.split("/", 1)
        if len(parts) != 2:
            continue
        try:
            raw = storage.download(parts[0], parts[1])
            page_texts.append(raw.decode("utf-8"))
        except Exception as e:
            logger.warning(f"Failed to download page_text derivative: {e}")
            continue
    return jsonify({"page_texts": page_texts, "count": len(page_texts)})


@sources_bp.route("/<source_id>/cancel", methods=["POST"])
@require_auth
def cancel_extraction(source_id: str):
    """Cancel an in-progress extraction."""
    result = db.session.execute(
        text(f"""
            SELECT extraction_status, celery_task_id
            FROM "{AI_SCHEMA}".sources WHERE id = :id
        """),
        {"id": source_id},
    )
    row = result.fetchone()
    if not row:
        return jsonify({"error": "Source not found"}), 404

    status, celery_task_id = row
    if status not in ("pending", "extracting"):
        return jsonify({"error": f"Cannot cancel extraction with status '{status}'"}), 409

    if celery_task_id:
        try:
            celery_app.control.revoke(celery_task_id)
        except Exception as exc:
            logger.warning("Failed to revoke Celery task %s: %s", celery_task_id, exc)

    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".sources
            SET extraction_status = 'cancelled',
                error_message = 'Cancelled by user',
                updated_at = NOW()
            WHERE id = :id
        """),
        {"id": source_id},
    )
    db.session.commit()

    return jsonify({"message": "Extraction cancelled"})


@sources_bp.route("/<source_id>/reextract", methods=["POST"])
@require_auth
def reextract_source(source_id: str):
    """Re-trigger extraction for a source."""
    result = db.session.execute(
        text(f"""
            SELECT id, extraction_status, auto_metadata FROM "{AI_SCHEMA}".sources WHERE id = :id
        """),
        {"id": source_id},
    )
    row = result.fetchone()
    if not row:
        return jsonify({"error": "Source not found"}), 404

    source_auto_metadata = row[2] or {}

    # Allow overriding the extraction model; fall back to the originally stored one
    VALID_EXTRACTION_MODELS = set(EXTRACTION_METHOD_CHOICES)
    data = request.get_json(silent=True) or {}
    extraction_model = data.get("extraction_model")
    if extraction_model and extraction_model not in VALID_EXTRACTION_MODELS:
        return jsonify(
            {
                "error": f"Invalid extraction_model '{extraction_model}'. "
                f"Valid options: {', '.join(sorted(VALID_EXTRACTION_MODELS))}"
            }
        ), 400
    if not extraction_model:
        extraction_model = source_auto_metadata.get("extraction_model")

    # Persist the (possibly new) choice in auto_metadata
    if extraction_model:
        updated_auto_metadata = {**source_auto_metadata, "extraction_model": extraction_model}
    else:
        updated_auto_metadata = source_auto_metadata

    # URL sources use a different extraction task that re-scrapes from the original URL
    origin_url = (
        source_auto_metadata.get("origin_url")
        if source_auto_metadata.get("source_type") == "url"
        else None
    )
    # Re-extraction is a new chargeable op. Generate a fresh per-call seed so the
    # worker's charge produces a distinct ledger row per reextract (one per
    # call), rather than deduping against the original upload's stable per-source
    # key. Generated BEFORE dispatch so it's stable across THIS call's Celery
    # retries; threaded to the task as reextract_seed → the worker keys on
    # (source_id, seed). Both seed and estimated_cost are pure — no billing
    # identity is read here (the port owns identity).
    re_action = "web_scrape" if origin_url else _extraction_billing_action(extraction_model)
    per_call_seed = str(uuid.uuid4())
    # Pre-charge balance check BEFORE any state mutation. If the check raises
    # 402/503, the source's extraction_status stays untouched — otherwise we'd
    # leave it stuck at 'pending' with no Celery task to clear it. The billing
    # port no-ops the check when billing is unconfigured.
    billing.check_balance(estimated_cost=_estimated_extraction_cost(re_action))

    # Reset status and trigger extraction; metadata (user-owned) is not touched.
    # Only mutate state AFTER balance check has passed.
    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".sources
            SET extraction_status = 'pending', error_message = NULL,
                auto_metadata = CAST(:auto_metadata AS jsonb), updated_at = NOW()
            WHERE id = :id
        """),
        {"id": source_id, "auto_metadata": json.dumps(updated_auto_metadata)},
    )
    db.session.commit()

    provider_keys = get_all_user_provider_keys()

    if origin_url:
        task = extract_url_source.delay(
            source_id,
            SOURCES_BUCKET,
            origin_url,
            provider_keys=provider_keys,
            reextract_seed=per_call_seed,
        )
    else:
        task = extract_source.delay(
            source_id,
            SOURCES_BUCKET,
            extraction_model=extraction_model,
            provider_keys=provider_keys,
            reextract_seed=per_call_seed,
        )

    db.session.execute(
        text(f"""
            UPDATE "{AI_SCHEMA}".sources
            SET celery_task_id = :task_id
            WHERE id = :id
        """),
        {"task_id": task.id, "id": source_id},
    )
    db.session.commit()

    return jsonify(
        {
            "message": "Re-extraction started",
            "task_id": task.id,
        }
    )


@sources_bp.route("/<source_id>/derivatives/<deriv_type>/download", methods=["GET"])
@require_auth
def download_derivative(source_id: str, deriv_type: str):
    """Download derivative content directly."""
    try:
        index = int(request.args.get("index", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "index must be an integer"}), 400

    result = db.session.execute(
        text(f'SELECT derivatives FROM "{AI_SCHEMA}".sources WHERE id = :id'),
        {"id": source_id},
    )
    row = result.fetchone()
    if not row:
        return jsonify({"error": "Source not found"}), 404

    derivatives = row[0] or {}

    if deriv_type not in derivatives:
        return jsonify({"error": f"No {deriv_type} derivative found"}), 404

    deriv_list = derivatives[deriv_type]
    if not isinstance(deriv_list, list):
        deriv_list = [deriv_list]

    if index >= len(deriv_list):
        return jsonify({"error": f"Derivative index {index} out of range"}), 404

    deriv_info = deriv_list[index]
    storage_path = deriv_info.get("storage_path")
    if not storage_path:
        return jsonify({"error": "Derivative storage path not found"}), 500

    try:
        storage = get_storage()
        stream = storage.stream_download_from_path(storage_path)
        content_length = next(stream)
        mimetype = {
            "markdown": "text/markdown",
            "html": "text/html",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
        }.get(deriv_info.get("format", ""), "application/octet-stream")
        resp_headers = {}
        if content_length:
            resp_headers["Content-Length"] = content_length
        return Response(stream, mimetype=mimetype, headers=resp_headers)
    except StorageError as e:
        logger.error(f"Storage error downloading derivative: {e}")
        return jsonify({"error": "Failed to download derivative"}), 500
    except StopIteration:
        return jsonify({"error": "Empty derivative"}), 500
