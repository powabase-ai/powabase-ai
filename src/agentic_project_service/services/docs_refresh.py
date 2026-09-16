"""Docs RAG refresh — ingest Powabase documentation into the hidden docs KB.

Runs ONLY on the singleton "system docs" project (one per deployment). It keeps a
hidden, full_document-indexed knowledge base of the Powabase docs in sync with
three upstream sources:
  - https://powabase.ai/llms-full.txt   (single combined file)
  - https://github.com/powabase-ai/docs.git
  - https://github.com/powabase-ai/agent-skills.git

The per-project Project Copilot never holds docs data — it queries this KB through
the internal `/api/internal/docs/search` endpoint (routes/internal_docs.py).

Design notes for reviewers:
  - Pure helpers (`sha256_text`, `doc_title`, `discover_markdown_docs`,
    `should_reindex`, `bootstrap_docs_kb`) are unit-tested.
  - `_ingest_markdown_source` + `refresh_docs_kb` touch storage + the indexing
    Celery dispatch, so they need the live project stack (storage + worker +
    embedding keys) to validate end-to-end; the unit tests mock those.
"""

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db import AI_SCHEMA, db
from ..strategies.registry import get_strategy
from .storage import SOURCES_BUCKET, StorageError, get_derivative_storage_path, get_storage

logger = logging.getLogger(__name__)

# Deterministic id for the singleton docs KB so concurrent bootstraps converge on
# one row (no unique constraint on knowledge_bases.name to rely on).
_DOCS_KB_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-0000d0c5d0c5")
# Skip pathologically large files (binary blobs, generated dumps) when walking repos.
_MAX_DOC_BYTES = 2 * 1024 * 1024

# Well-known marker name for the hidden docs KB. Not user-facing (the docs project
# is hidden from non-members and this KB is only reached via the internal
# search endpoint), so the leading underscores just signal "system-managed".
DOCS_KB_NAME = "__powabase_docs__"
DOCS_KB_DESCRIPTION = "System-managed Powabase documentation index (do not edit)."

LLMS_FULL_URL = os.getenv("DOCS_LLMS_FULL_URL", "https://powabase.ai/llms-full.txt")


@dataclass(frozen=True)
class DocsRepo:
    url: str
    key: str  # stable key-prefix used in source names, e.g. "docs"


DOCS_REPOS: tuple[DocsRepo, ...] = (
    DocsRepo(os.getenv("DOCS_REPO_DOCS", "https://github.com/powabase-ai/docs.git"), "docs"),
    DocsRepo(
        os.getenv("DOCS_REPO_AGENT_SKILLS", "https://github.com/powabase-ai/agent-skills.git"),
        "agent-skills",
    ),
)


@dataclass(frozen=True)
class DocRecord:
    """One documentation article to (re)index."""

    key: str  # stable source name, e.g. "docs:guides/auth-connection.md"
    title: str
    content: str
    content_hash: str


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def doc_title(content: str, fallback: str) -> str:
    """First markdown H1 (`# Title`) or YAML-frontmatter `title:`, else fallback."""
    for line in content.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
        if s.lower().startswith("title:"):
            return s.split(":", 1)[1].strip().strip("\"'") or fallback
    return fallback


def discover_markdown_docs(root: Path, key_prefix: str) -> list[DocRecord]:
    """Walk a checkout for markdown files and build DocRecords (sorted by key).

    Skips empty/whitespace-only files. The source key is
    ``<key_prefix>:<relative posix path>`` so it's stable across refreshes.
    """
    docs: list[DocRecord] = []
    for path in sorted(root.rglob("*")):
        # Skip symlinks (avoid loops / escaping the checkout) and non-markdown.
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix.lower() not in (".md", ".mdx"):
            continue
        try:
            if path.stat().st_size > _MAX_DOC_BYTES:
                logger.warning("Skipping oversized doc %s", path)
                continue
        except OSError:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            continue
        rel = path.relative_to(root).as_posix()
        docs.append(
            DocRecord(
                key=f"{key_prefix}:{rel}",
                title=doc_title(content, fallback=rel),
                content=content,
                content_hash=sha256_text(content),
            )
        )
    return docs


def should_reindex(existing_hash: str | None, new_hash: str) -> bool:
    """A doc needs (re)indexing when it's new or its content changed."""
    return existing_hash != new_hash


# ---------------------------------------------------------------------------
# KB bootstrap (unit-tested against the test DB)
# ---------------------------------------------------------------------------


def bootstrap_docs_kb(session: Session) -> str:
    """Return the hidden docs KB id, creating it (full_document strategy) if absent.

    Idempotent AND concurrency-safe: the KB id is derived deterministically from
    the well-known name, and the insert is ``ON CONFLICT (id) DO NOTHING``, so two
    workers bootstrapping at once converge on a single row.
    """
    kb_id = str(uuid.uuid5(_DOCS_KB_NAMESPACE, DOCS_KB_NAME))
    strategy = get_strategy("full_document")
    # Dense (vector_search) retrieval instead of the strategy's default hybrid:
    # semantic search fits documentation Q&A well, and it avoids the BM25s sparse
    # path, whose incrementally-built index can fall out of sync with its item-id
    # map under a doc-at-a-time refresh (surfaces as an IndexError at query time).
    # "vector_search" is a compatible retriever for full_document (see
    # strategies/registry.py). Existing KBs keep their config (ON CONFLICT below).
    retrieval_config = {**strategy["default_retrieval_config"], "method": "vector_search"}
    session.execute(
        text(f"""
            INSERT INTO "{AI_SCHEMA}".knowledge_bases (id, name, description, indexing_config, retrieval_config)
            VALUES (:id, :name, :desc, CAST(:ic AS jsonb), CAST(:rc AS jsonb))
            ON CONFLICT (id) DO NOTHING
        """),
        {
            "id": kb_id,
            "name": DOCS_KB_NAME,
            "desc": DOCS_KB_DESCRIPTION,
            "ic": json.dumps(strategy["default_indexing_config"]),
            "rc": json.dumps(retrieval_config),
        },
    )
    session.commit()
    return kb_id


# ---------------------------------------------------------------------------
# Ingestion + orchestration (need the live stack to validate end-to-end)
# ---------------------------------------------------------------------------


def fetch_llms_full(url: str = LLMS_FULL_URL) -> DocRecord | None:
    """Fetch llms-full.txt as a single doc; returns None on failure."""
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None
    content = resp.text
    if not content.strip():
        return None
    return DocRecord(
        key="llms-full:llms-full.txt",
        title="Powabase llms-full",
        content=content,
        content_hash=sha256_text(content),
    )


def _git_clone(url: str, dest: Path) -> bool:
    """Shallow-clone a repo. Returns False (logged) on failure."""
    try:
        subprocess.run(
            # `--` stops option parsing so a URL starting with `-` can't be
            # interpreted as a git flag (argument injection).
            ["git", "clone", "--depth", "1", "--", url, str(dest)],
            check=True,
            capture_output=True,
            timeout=300,
            # A private/renamed repo would otherwise block on an interactive
            # credential prompt for the full 300s timeout; fail fast instead.
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        logger.warning("git clone failed for %s: %s", url, e)
        return False


def _ingest_markdown_source(session: Session, storage, kb_id: str, doc: DocRecord) -> str:
    """Create/update a source from markdown and dispatch indexing into the docs KB.

    Returns one of: "unchanged", "dispatched", "error:<msg>". Dedups on
    content_hash so unchanged docs are skipped (no re-embed). Mirrors the
    extraction pipeline's derivative shape so the existing full_document indexer
    can read it (``derivatives["markdown"][0]["storage_path"]``).
    """
    from ..routes.knowledge_bases import index_source_into_kb
    from ..tasks.extraction import update_source_extraction_result

    existing = session.execute(
        text(
            f'SELECT s.id, s.content_hash, s.extraction_status, '
            f"  EXISTS(SELECT 1 FROM \"{AI_SCHEMA}\".indexed_sources i "
            f"         WHERE i.source_id = s.id AND i.knowledge_base_id = :kb "
            f"           AND i.index_status = 'indexed') "
            f'FROM "{AI_SCHEMA}".sources s WHERE s.name = :n'
        ),
        {"n": doc.key, "kb": kb_id},
    ).fetchone()
    # Only skip when the content is unchanged, extraction completed, AND the doc
    # is *successfully* indexed into THIS KB (index_status = 'indexed', the
    # terminal-success value written by tasks/indexing.py). Checking mere row
    # existence is not enough: index_source_into_kb inserts the indexed_sources
    # row as 'pending' before dispatch, and a failed embed leaves it at 'failed'
    # (not deleted). Without the '= indexed' predicate a doc whose indexing failed
    # *after* extraction (a transient error, or the pre-fix billing 402) would
    # match the EXISTS forever — every later refresh skips it as "unchanged" and
    # never retries the index dispatch, silently dropping it.
    if (
        existing
        and not should_reindex(existing[1], doc.content_hash)
        and existing[2] == "extracted"
        and existing[3]  # already indexed into this KB
    ):
        return "unchanged"

    source_id = str(existing[0]) if existing else str(uuid.uuid4())
    filename = "content.md"
    data = doc.content.encode("utf-8")
    deriv_path = get_derivative_storage_path(source_id, "markdown", filename)
    # Deterministic stored path (what storage.upload will return) — computed up
    # front so we can persist the source row BEFORE writing the object, ensuring
    # a row INSERT that violates UNIQUE(content_hash) can't orphan an upload.
    stored_path = f"{SOURCES_BUCKET}/{deriv_path.lstrip('/')}"

    if existing:
        session.execute(
            text(f"""
                UPDATE "{AI_SCHEMA}".sources
                SET content_hash = :h, storage_path = :sp, updated_at = NOW()
                WHERE id = :id
            """),
            {"h": doc.content_hash, "sp": stored_path, "id": source_id},
        )
    else:
        session.execute(
            text(f"""
                INSERT INTO "{AI_SCHEMA}".sources
                    (id, name, file_type, storage_path, extraction_status, content_hash)
                VALUES (:id, :n, 'md', :sp, 'pending', :h)
            """),
            {"id": source_id, "n": doc.key, "sp": stored_path, "h": doc.content_hash},
        )
    session.commit()

    # Now write the derivative + mark extracted. If this fails the source is left
    # 'pending' (re-processed next refresh — the deterministic path overwrites).
    # (bucket existence is ensured once per refresh by the caller, not per-doc.)
    storage.upload(SOURCES_BUCKET, deriv_path, data, "text/markdown")

    update_source_extraction_result(
        source_id,
        derivatives={
            "markdown": [
                {
                    "storage_path": stored_path,
                    "filename": filename,
                    "content_type": "text/markdown",
                    "size": len(data),
                }
            ]
        },
        auto_metadata={"title": doc.title, "url": _doc_url(doc), "source": "powabase-docs"},
        status="extracted",
    )

    result = index_source_into_kb(kb_id, source_id)
    if result.get("error"):
        return f"error:{result['error']}"
    # index_source_into_kb DISPATCHES an async indexing task; actual embedding
    # completes later in the worker. "dispatched" reflects that honestly.
    return "dispatched"


def _doc_url(doc: DocRecord) -> str | None:
    """Best-effort public docs URL for citation, derived from the source key.

    Only the `docs:` repo maps cleanly to https://docs.powabase.ai/<path>;
    `index`/`readme` pages resolve to their directory URL. agent-skills and
    llms-full have no per-article public URL, so they return None.
    """
    if not doc.key.startswith("docs:"):
        return None
    rel = doc.key[len("docs:") :].rsplit(".", 1)[0]  # strip extension
    head, _, tail = rel.rpartition("/")
    if tail.lower() in ("index", "readme"):
        rel = head  # the directory page
    return f"https://docs.powabase.ai/{rel}".rstrip("/")


def _prune_stale_docs_sources(
    session: Session, storage, fresh_keys: set[str], reachable_prefixes: set[str]
) -> int:
    """Delete ai.sources rows for docs-namespace sources no longer upstream.

    Scoped to ``<prefix>:*`` names for each prefix in ``reachable_prefixes``
    (e.g. "llms-full", "docs", "agent-skills") — never touches non-docs
    sources, and never touches a prefix whose upstream fetch/clone failed, or
    gathered zero docs, *this* run (an outage — or a clone that "succeeds" but
    yields an empty tree — must not be read as "everything was deleted"; see
    the caller's ``prune_scope`` intersection with ``fresh_keys``' prefixes).

    This closes the rename-with-identical-content trap: without pruning, a
    renamed doc (a.md -> b.md, same content) collides with the stale a.md row
    on the global UNIQUE(content_hash) index (migration 0021) and every later
    refresh errors out on the insert. Deleting the stale row here frees the
    hash before ingestion runs. Deletion cascades to ai.indexed_sources /
    ai.chunks / ai.embeddings automatically (ON DELETE CASCADE, see
    ai_schema.sql) — mirrors routes/sources.py's delete_source, including its
    best-effort storage cleanup. One failed prune shouldn't abort the refresh:
    caught + logged per source.
    """
    if not reachable_prefixes:
        return 0
    prefixes = sorted(reachable_prefixes)
    like_clauses = " OR ".join(f"name LIKE :p{i}" for i in range(len(prefixes)))
    params = {f"p{i}": f"{p}:%" for i, p in enumerate(prefixes)}
    rows = session.execute(
        text(
            f'SELECT id, name, storage_path, derivatives FROM "{AI_SCHEMA}".sources WHERE {like_clauses}'
        ),
        params,
    ).fetchall()

    pruned = 0
    for source_id, name, storage_path, derivatives in rows:
        if name in fresh_keys:
            continue
        try:
            paths_to_delete = []
            if storage_path:
                parts = storage_path.split("/", 1)
                if len(parts) == 2:
                    paths_to_delete.append(parts[1])
            for deriv_list in (derivatives or {}).values():
                for deriv in deriv_list:
                    dpath = deriv.get("storage_path")
                    if dpath:
                        dparts = dpath.split("/", 1)
                        if len(dparts) == 2:
                            paths_to_delete.append(dparts[1])
            if paths_to_delete:
                try:
                    storage.delete(SOURCES_BUCKET, paths_to_delete)
                except StorageError as e:
                    logger.warning("Failed to delete storage for stale doc %s: %s", name, e)
            session.execute(
                text(f'DELETE FROM "{AI_SCHEMA}".sources WHERE id = :id'),
                {"id": source_id},
            )
            session.commit()
            pruned += 1
        except Exception as e:  # one bad prune shouldn't abort the whole refresh
            session.rollback()
            logger.warning("Failed to prune stale doc %s: %s", name, e)
    return pruned


def refresh_docs_kb(
    *,
    repos: tuple[DocsRepo, ...] = DOCS_REPOS,
    llms_full_url: str = LLMS_FULL_URL,
) -> dict:
    """Full refresh: bootstrap the KB, gather all docs, (re)index changed ones.

    Safe to run repeatedly — unchanged docs are skipped via content_hash.
    """
    session = db.session
    kb_id = bootstrap_docs_kb(session)
    storage = get_storage()
    storage.ensure_bucket(SOURCES_BUCKET)  # once per refresh, not once per doc

    docs: list[DocRecord] = []
    # Track upstream-source reachability separately from per-doc index results:
    # every source failing (GitHub outage, bad DOCS_REPO_* URL, no `git` in the
    # image) gathers 0 docs and would otherwise log INFO "complete" — an
    # indistinguishable healthy no-op while the index silently goes stale.
    # Also gates pruning below: a prefix that failed to fetch this run must not
    # have its (still-valid) sources read as "gone".
    sources_total = 1 + len(repos)  # llms-full + each repo
    sources_ok = 0
    reachable_prefixes: set[str] = set()
    llms = fetch_llms_full(llms_full_url)
    if llms:
        docs.append(llms)
        sources_ok += 1
        reachable_prefixes.add("llms-full")

    with tempfile.TemporaryDirectory(prefix="docs-refresh-") as tmp:
        for repo in repos:
            dest = Path(tmp) / repo.key
            if _git_clone(repo.url, dest):
                docs.extend(discover_markdown_docs(dest, key_prefix=repo.key))
                sources_ok += 1
                reachable_prefixes.add(repo.key)

    # Dedup by content_hash across all sources: identical content (e.g. a doc that
    # also appears verbatim in llms-full.txt) would otherwise trip the global
    # UNIQUE(content_hash) index on ai.sources and abort the ingest.
    seen_hashes: set[str] = set()
    deduped: list[DocRecord] = []
    for doc in docs:
        if doc.content_hash in seen_hashes:
            continue
        seen_hashes.add(doc.content_hash)
        deduped.append(doc)

    # Prune before ingesting: a rename (a.md -> b.md, same content) must free
    # a.md's content_hash before b.md's insert runs, or the insert hits the
    # UNIQUE(content_hash) constraint every cycle. See _prune_stale_docs_sources.
    fresh_keys = {doc.key for doc in deduped}
    # A clone/fetch can "succeed" (git exits 0 / HTTP 200) yet gather zero markdown
    # docs for that prefix — e.g. upstream force-pushed to empty, or renamed its
    # default branch out from under a shallow clone. `reachable_prefixes` alone
    # can't tell that apart from a healthy repo, so it's not safe to prune on: doing
    # so would treat the whole prefix as "gone upstream" and wipe every source under
    # it. Only prune a prefix this run actually gathered >=1 doc for; the rename
    # case still gathers >=1 doc for its prefix, so pruning still runs there.
    prefixes_with_docs = {key.split(":", 1)[0] for key in fresh_keys}
    prune_scope = reachable_prefixes & prefixes_with_docs
    pruned = _prune_stale_docs_sources(session, storage, fresh_keys, prune_scope)

    counts = {"dispatched": 0, "unchanged": 0, "error": 0}
    for doc in deduped:
        try:
            status = _ingest_markdown_source(session, storage, kb_id, doc)
        except Exception as e:  # one bad doc shouldn't abort the whole refresh
            # Roll back so a failed statement doesn't poison the session for the
            # remaining docs in the loop.
            session.rollback()
            logger.warning("Failed to ingest %s: %s", doc.key, e)
            status = "error"
        if status == "dispatched":
            counts["dispatched"] += 1
        elif status == "unchanged":
            counts["unchanged"] += 1
        else:
            # "error:<msg>" from a failed index dispatch (the exception path above
            # logs its own case). Surface the message with the doc key so a doc
            # that always fails to index isn't an anonymous "error: N" forever.
            if status.startswith("error:"):
                logger.warning("Index dispatch failed for %s: %s", doc.key, status[6:])
            counts["error"] += 1

    if sources_ok == 0:
        logger.error(
            "Docs refresh gathered 0 docs: all %d upstream sources failed "
            "(GitHub outage / bad DOCS_REPO_* URL / missing git). Index left stale.",
            sources_total,
        )
    logger.info(
        "Docs refresh complete: kb=%s docs=%d sources_ok=%d/%d pruned=%d %s",
        kb_id,
        len(deduped),
        sources_ok,
        sources_total,
        pruned,
        counts,
    )
    return {
        "kb_id": kb_id,
        "docs": len(deduped),
        "sources_ok": sources_ok,
        "sources_total": sources_total,
        "pruned": pruned,
        **counts,
    }
