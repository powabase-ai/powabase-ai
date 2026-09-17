"""Every writer of ``ai.indexed_sources.index_status`` must be declared here.

WHY THIS FILE EXISTS.

The invariant the indexing pipeline maintains -- at most one durable result per
row, and every row reaches a terminal state -- is a statement about ALL writers
of that column taken together. There is no single gate. There are ~39 write
sites across four modules, and each carries its own hand-written ``WHERE``
clause chosen by reasoning about that one site.

That asymmetry is a defect generator, and it has fired repeatedly. Every one of
those bugs had the same shape: a predicate picked because it discriminates
*here*, described in a comment as though it discriminated *everywhere*. The
worst example took three attempts at ONE ``except`` block:

    celery_task_id       -- rejected, the reconciler owns no task id
    index_status='pending' -- wrong, ten writers produce 'pending'
    attempts             -- current

Each wrong choice was made the same way that produced the previous wrong one:
from memory, because the information needed to check it was not written down
anywhere. This file writes it down.

WHAT IT BUYS, AND WHAT IT DOES NOT.

It does not enforce correctness. A declared writer can still carry a wrong
predicate. What it removes is the *excuse*: "does this predicate identify only
my row?" becomes a question you answer against a list rather than against
recall. A new write site that nobody has thought about fails here until someone
declares its from-states and what authorizes it.

It is deliberately a test in ``tests/unit`` rather than a doc: a table nobody is
forced to update is a table that rots, and this tier is the one CI gates on.

MAINTAINING IT: if this fails after you add a write, add your site to
``DECLARED`` with the states it may transition FROM and the token that
authorizes it. If you cannot name the authorizing token, that is the finding --
an unconditional write to this column is how rows get silently overwritten.
"""

import collections
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "agentic_project_service"

# module -> target state -> (count, what authorizes the write)
#
# "token" is the thing that makes the write MINE rather than anyone's:
#   ownership  -- celery_task_id equals the writing task's own id
#   status     -- the row is in a state only my predecessor could have left it in
#   status+attempts -- status alone is ambiguous here; attempts disambiguates
#                      (the reconciler's reset preserves attempts, every reindex
#                       route zeroes it)
#   user-intent -- an explicit user action; deliberately unconditional
DECLARED: dict[tuple[str, str], tuple[int, str]] = {
    ("routes/knowledge_bases.py", "pending"):   (5, "user-intent (reindex: fresh intent, resets attempts)"),
    ("routes/knowledge_bases.py", "indexing"):  (3, "status ('indexed' only, for reenrich) + dispatch"),
    ("routes/knowledge_bases.py", "indexed"):   (3, "ownership (reenrich undo, fenced on our own tid + status)"),
    ("routes/knowledge_bases.py", "failed"):    (4, "user-intent / validation failure at the route"),
    ("routes/knowledge_bases.py", "cancelled"): (2, "user-intent (explicit cancel)"),
    ("routes/observability.py", "failed"):      (1, "operator action"),
    ("tasks/indexing.py", "pending"):           (3, "ownership (retry paths, fenced on the claiming task)"),
    ("tasks/indexing.py", "indexing"):          (4, "the atomic claim: pending -> indexing, attempts+1"),
    ("tasks/indexing.py", "indexed"):           (4, "ownership (terminal write fenced on celery_task_id)"),
    ("tasks/indexing.py", "failed"):            (2, "ownership (terminal write fenced on celery_task_id)"),
    ("tasks/watchdog.py", "indexing"):          (4, "read-side only: ORPHAN_QUERY predicate, not a write"),
    ("tasks/watchdog.py", "pending"):           (2, "status ('indexing' + task not alive) -- recovery reset"),
    ("tasks/watchdog.py", "failed"):            (2, "status+attempts -- see the module's own comment"),
}


def _actual() -> dict[tuple[str, str], int]:
    found: collections.Counter = collections.Counter()
    for f in SRC.rglob("*.py"):
        for line in f.read_text().splitlines():
            for m in re.finditer(r"index_status\s*=\s*'([a-z_]+)'", line):
                found[(str(f.relative_to(SRC)), m.group(1))] += 1
    return dict(found)


def test_every_index_status_writer_is_declared():
    actual = _actual()
    undeclared = sorted(set(actual) - set(DECLARED))
    assert not undeclared, (
        f"undeclared index_status write sites: {undeclared}. Add each to "
        "DECLARED with the states it may transition FROM and the token that "
        "authorizes it. If you cannot name the token, the write is "
        "unconditional -- which is how a row silently loses someone else's "
        "durable result."
    )


def test_no_declared_writer_has_disappeared():
    """A stale entry is as misleading as a missing one — it is read as coverage."""
    actual = _actual()
    gone = sorted(set(DECLARED) - set(actual))
    assert not gone, f"declared but no longer present: {gone}. Remove them."


def test_writer_counts_match():
    """Counts, not just presence: a second write slipped into an existing module
    is exactly the case that reads as already-covered."""
    actual = _actual()
    drift = {
        k: (DECLARED[k][0], actual[k])
        for k in DECLARED
        if k in actual and DECLARED[k][0] != actual[k]
    }
    assert not drift, (
        f"write-count drift (declared, actual): {drift}. A new write in a module "
        "that already appears here is the easiest one to miss — confirm its "
        "predicate identifies only its own row, then update the count."
    )
