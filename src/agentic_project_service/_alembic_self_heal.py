"""Self-heal helper for alembic's env.py (project-service per-project DB).

Kept in the importable package — not under ``migrations/`` — so unit tests
can import it directly without triggering alembic's env.py boot dispatch.

Imported by ``migrations/env.py`` immediately before alembic resolves
heads on each container start. Production / CI / fresh local dev all hit
no-op branches and pay nothing. Only pre-rename local environments see
the warning + heal — see ``cleanup_orphan_versions`` docstring for full
context.

Design intent (revised after the f7294e9b post-mortem)
======================================================

The naive version of this helper just DELETEd any row in
``ai.alembic_version`` whose ``version_num`` no longer matched a script
on disk. That looked safe for the April-29 PR #144 case (just a rename),
but turned out to silently corrupt schema state when the orphan row was
load-bearing: it signalled that the DB had followed a chain branch which
skipped a sibling migration. Specifically the pre-rename ``0017``
(reasoning_requested) DBs had never run ``0017_ai_provider_keys.py``
either; deleting the orphan claimed the chain was complete when in fact
``ai.ai_provider_keys`` table was missing, and the next agent run blew
up with ``relation "ai.ai_provider_keys" does not exist``.

The revised contract:

  * **No row is silently deleted.** Every orphan is matched against a
    registry of known-rename incidents (``KNOWN_ORPHANS``). For each
    matching orphan, we run the registered idempotent schema-coherence
    DDL BEFORE deleting the row. That DDL exists to bring the schema
    up to whatever state the chain implies, even if the actual
    migration that creates it was skipped.
  * **Unknown orphans halt startup with a loud error.** A row that
    points at no script and no registered handler is treated as a
    correctness emergency: the operator must investigate the schema by
    hand and decide whether to add a ``KNOWN_ORPHANS`` entry or delete
    the row themselves. Silent deletion is never the right move.

This means future migration renames also need to register their
``KNOWN_ORPHANS`` entry IN THE SAME PR. The PR review checklist should
catch this — adding an orphan-producing rename without a registry
entry will fail every dev environment that had the pre-rename revision
stamped.
"""

from __future__ import annotations

import logging
from typing import Iterable, Protocol

from sqlalchemy import text


logger = logging.getLogger("alembic.env")


# Registry of (version_num that no longer appears in any .py file) ->
# (human-readable rationale, idempotent DDL to make the schema coherent
# before the row is deleted). DDL MUST be CREATE-IF-NOT-EXISTS /
# ADD-COLUMN-IF-NOT-EXISTS style so reruns are no-ops.
#
# Add a new entry IN THE SAME PR as any future migration rename that
# could leave a dev DB stamped at the old name. The review checklist
# is: "this PR renames or deletes a migration revision -> does
# KNOWN_ORPHANS cover both the old name and the schema state that name
# implied?".
KNOWN_ORPHANS: dict[str, tuple[str, str]] = {
    "0017": (
        "PR #144 (April 29, 2026) renamed revision '0017' "
        "(add_reasoning_requested_to_orchestration_runs) -> '0018' to "
        "resolve a same-number collision with '0017_ai_provider_keys'. "
        "DBs stamped at the old '0017' followed the chain "
        "0016 -> old 0017 -> ... and SKIPPED '0017_ai_provider_keys' "
        "entirely. Before deleting the orphan row we apply that "
        "migration's CREATE TABLE so the schema becomes coherent with "
        "what the current chain implies. 'reasoning_requested' column "
        "is already present from the old 0017 path; no action needed "
        "for it.",
        # Exact body of 0017_ai_provider_keys.py's upgrade(), kept in
        # sync with the migration. If that migration changes, update
        # this string. The IF NOT EXISTS guard makes the rerun safe
        # for DBs that DO have the table.
        """
        CREATE TABLE IF NOT EXISTS ai.ai_provider_keys (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            provider VARCHAR(50) NOT NULL UNIQUE,
            api_key_encrypted TEXT NOT NULL,
            is_valid BOOLEAN NOT NULL DEFAULT true,
            last_validated_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
    ),
}


class _ScriptDirectoryLike(Protocol):
    """Minimal interface we use from alembic's ScriptDirectory."""

    def walk_revisions(self) -> Iterable: ...  # pragma: no cover - structural type


class UnknownOrphanVersion(RuntimeError):
    """Raised when ai.alembic_version has a row whose version_num matches
    no on-disk migration AND no entry in ``KNOWN_ORPHANS``.

    This is treated as a correctness emergency: silent deletion would
    risk leaving the schema in a state that doesn't match what the
    chain implies (the exact regression caused by the naive first
    version of this helper). Operator must investigate."""


def cleanup_orphan_versions(connection, script_dir: _ScriptDirectoryLike) -> None:
    """Resolve rows in ``ai.alembic_version`` that reference revision
    names no longer present in any migration script on disk.

    For each orphan row:
      * If the orphan is in ``KNOWN_ORPHANS``: run the registered
        schema-coherence DDL, then delete the row. Idempotent under
        re-run (DDL is IF-NOT-EXISTS).
      * If the orphan is NOT in ``KNOWN_ORPHANS``: raise
        ``UnknownOrphanVersion`` rather than silently delete. The
        operator must investigate the schema and either add a
        ``KNOWN_ORPHANS`` entry or delete the row by hand after
        verifying schema coherence.

    No-op when:
      * the version table doesn't exist yet (first-ever boot),
      * the version table is empty, or
      * every version_num corresponds to a known revision name on disk.

    Production / CI / new local dev envs all hit a no-op path. Only
    pre-rename local envs see the warning path.

    Args:
        connection: a live SQLAlchemy Connection bound to the project DB.
        script_dir: alembic ScriptDirectory, used to enumerate the set
            of valid revision names from the on-disk migration scripts.

    Raises:
        UnknownOrphanVersion: an orphan row exists with no on-disk
            migration script AND no entry in KNOWN_ORPHANS.
    """
    exists = connection.execute(
        text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='ai' AND table_name='alembic_version')"
        )
    ).scalar()
    if not exists:
        return

    db_revs: set[str] = set(
        connection.execute(text("SELECT version_num FROM ai.alembic_version")).scalars()
    )
    if not db_revs:
        return

    valid_revs = {rev.revision for rev in script_dir.walk_revisions()}
    orphans = db_revs - valid_revs
    if not orphans:
        return

    # Triage: split known from unknown. Fail loud on unknown BEFORE
    # touching anything — partial work is worse than no work.
    unknown = [o for o in orphans if o not in KNOWN_ORPHANS]
    if unknown:
        raise UnknownOrphanVersion(
            f"ai.alembic_version contains row(s) with no matching .py file "
            f"AND no entry in _alembic_self_heal.KNOWN_ORPHANS: "
            f"{sorted(unknown)}. This means a migration was renamed/deleted "
            f"without registering its schema-coherence handler. Refusing to "
            f"delete the row(s) silently because doing so might claim the "
            f"chain is complete while the schema is actually missing tables "
            f"or columns. Investigate by hand: (1) check what the orphan "
            f"revision originally did to the schema; (2) verify those "
            f"changes exist in the current DB; (3) either add a "
            f"KNOWN_ORPHANS entry with the idempotent DDL needed to make "
            f"the schema coherent, OR DELETE the row manually after "
            f"confirming the schema is fine. See the docstring of this "
            f"module for the f7294e9b post-mortem."
        )

    # All orphans are registered — process each one with its DDL.
    for orphan in sorted(orphans):
        rationale, ddl = KNOWN_ORPHANS[orphan]
        logger.warning(
            "alembic env self-heal: orphan version_num=%r detected. Rationale: %s",
            orphan,
            rationale,
        )
        connection.execute(text(ddl))
        connection.execute(
            text("DELETE FROM ai.alembic_version WHERE version_num = :v"),
            {"v": orphan},
        )
        logger.info(
            "alembic env self-heal: applied schema-coherence DDL and deleted orphan version_num=%r",
            orphan,
        )
