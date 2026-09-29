"""One partial HNSW index per large knowledge base, on ``ai.embeddings``.

``ai.embeddings`` carries a single HNSW index per embedding dimension, over
every knowledge base in the project. A KB-scoped vector search therefore walks
a graph built over rows it will throw away, and stops as soon as the join has
produced ``top_k`` rows of the knowledge base it wanted -- which is both slow
(the graph is as large as the whole table) and approximate over the wrong
population (the true top-k *within* one knowledge base needs far more
candidates than ``hnsw.ef_search`` emits globally).

A partial index fixes both, for the knowledge bases big enough to be worth one --
counting only the one item population it covers, because ``ai.embeddings`` is
polymorphic (``PER_KB_INDEX_ITEM_TABLE``):

    CREATE INDEX CONCURRENTLY hnsw_kb_<hex>_<dims> ON ai.embeddings
      USING hnsw ((embedding::vector(<dims>)) vector_cosine_ops)
      WHERE knowledge_base_id = '<kb>' AND dims = <dims>
        AND item_table = 'chunks';

Measured on a 56,000-row fixture at 1536 dimensions whose largest knowledge
base held 12,000 rows (21% of the table), ``shared_buffers`` 128 MB,
``hnsw.ef_search`` 40, top-20: 2.3 ms warm on the shared index against 1.7 ms
on the partial one. On a larger fixture (600,000 rows, a 73,290-row knowledge
base) the same change moved warm p50 from 182 ms to 1.4 ms and cold p50 from
824 ms to 225 ms, and the partial index was 573 MB and built in 15-50 s
depending on ``maintenance_work_mem``.

The *quality* half of the argument is not yet established at the scale that
matters. The gain appears where one knowledge base is a small fraction of a
large table, because there the shared index's post-filter throws away most of
what it found; on the fixtures reachable from a test suite the indexed
knowledge base is most of the table and the two are equally approximate at
``hnsw.ef_search = 40`` (0.28 against 0.29 recall@20 -- a wash). What is pinned
by tests is what holds at every scale: the partial index is *complete*, so an
exhaustive search through it returns exactly an exact scan's top-k.

Three properties shape this module:

* **The planner only matches a partial index from a predicate on the indexed
  relation.** Today's ``vector_search`` filters ``knowledge_base_id`` on the
  *item* table and joins to ``ai.embeddings``; nothing constrains
  ``e.knowledge_base_id``, and the planner does not reason through the join to
  get there. So the index is useless without the matching predicate in
  ``base_vector_store`` -- verified by ``EXPLAIN``: with the partial index
  present and the old query shape, the plan still picks the shared index.
* **A generic plan can only match it if it can prove the whole predicate.**
  The predicate names a ``knowledge_base_id`` *and* a ``dims`` as literals, and
  an unknown ``LIMIT`` separately prices an ordered index scan out. So
  ``vector_search`` interpolates all three -- the KB id, ``dims`` and the limit
  -- rather than binding them; ``base_vector_store.kb_sql_literal`` carries the
  measured decomposition showing that any one of them left bound loses the
  index in a generic plan. This matters because psycopg prepares a statement
  after ``prepare_threshold`` executions and PostgreSQL then weighs its generic
  plan against the custom ones: measured with the id bound, the 11th execution
  onwards fell back to a bitmap scan plus an exact sort, 1.1 ms to 57 ms --
  correct but 50x slower, for the life of that connection.
* **``CREATE INDEX CONCURRENTLY`` cannot run inside a transaction** and can
  leave an ``INVALID`` index behind when it fails, so the build lives here, in
  an out-of-band task, with the same invalid-index repair the pg_search BM25
  path uses -- not in ``base_vector_store.ensure_embedding_index``, which runs
  inside the indexing transaction on purpose.

**No index gets built for a project whose knowledge bases are all below the
threshold; the query change still applies to all of them.** The embeddings-side
predicate applies to every KB-scoped vector search from the moment it deploys,
with or without a partial index, and for a knowledge base big enough that the
planner had been reaching the shared index it can flip the plan to an exact
bitmap scan and sort: measured 0.356 ms approximate to 3.540 ms exact at 9%
selectivity, 3.59 ms to 80 ms at 21%, and 129 ms at 30%. A knowledge base whose
searches already run an exact scan sees no flip at all, and neither do the small
ones, which measured *faster* with the predicate (9.96 ms to 4.76 ms at 1.4%
selectivity, 19.50 ms to 13.00 ms at 4.3%).

That window is a real cost and it is paid for something real: the plan it
replaces was fast and quietly wrong. Measured against an exact scan over the
same knowledge base, the old shape returned recall 0.65-0.70 -- it stopped as
soon as the join had produced ``top_k`` rows of the wanted knowledge base --
where the new one returns 1.0. So the trade in that window is lossy-fast for
exact-slow.

**Whether a given project has that window at all depends on its table shape, not
on its embedding width.** Both the flip and its absence are decided by a cost
race between an approximate index scan and an exact one, and that race is decided
by how large the knowledge base is relative to the table, how the rows are laid
out and what is cached -- the two knowledge bases where the regression was
measured held 12.6k and 18k rows, and a second fixture built independently
measured the new shape 3.6x *faster* at the same recall. Stating the window as a
property of the width would be the same mistake as stating the planner's choice
that way. The 50,000-row default deliberately leaves the measured window
unindexed, and a project finds out whether it has one by measuring its own.
The reason is the other side of the trade: an index can be built, maintained on
every write and never scanned, because in some storage layouts the planner
prefers the project-wide index even for a knowledge base that owns one. Until
that is settled, the default keeps almost nothing crossing it; lowering it is a
per-project setting change, made after confirming the index is really scanned.

(The partial index is itself approximate, at recall 0.90 on that fixture. The
exactness above is a property of the transitional plan, not of the destination.)

The start-up sweep is not free for a small project either: it runs one grouped
count over ``ai.embeddings`` on every boot, bounded by ``SWEEP_TIMEOUT_MS``,
even when it then dispatches nothing.

**One build or drop at a time per project, decided here and never in Postgres'
lock queue.** Every index this module creates is on the one shared table, and a
second ``CREATE``/``DROP INDEX CONCURRENTLY`` queued on that table while a build
runs is not merely slow: it holds a snapshot while it waits, the build's last
phase waits for that snapshot, and the deadlock detector kills the build after
all of its work -- observed on a production project as seven deadlocks in about
36 hours, the largest build killed at its end twice after eight hours of work, with
a third earlier death consistent with it (issue #95 and its follow-up comment of
the same week). ``_TableGate`` carries the
mechanism, the evidence and the operator rule that follows from it; the short
form is that a reconcile which cannot take the table issues nothing, returns
``table_busy``, and its task comes back on a clock of its own.

Nothing here touches the shared per-dimension index. Replacing it with a
residual one is what turns the transitional write cost of maintaining two graphs
into a large write *gain*, and it is deliberately a follow-up (**#88**). Two
things about it belong here, on the build side, because they are facts about the
index this module builds:

* **The residual predicate has to exclude the population, not the knowledge
  base**: ``AND NOT (knowledge_base_id IN (...) AND item_table = 'chunks')``, never
  ``AND knowledge_base_id NOT IN (...)``. The index built here holds ``chunks``
  alone (``PER_KB_INDEX_ITEM_TABLE``), so the weaker form excludes a covered
  knowledge base's other three populations from both indexes at once and every
  document-store vector search on it sequential-scans the table.
* **It cannot land while any process still issues the older query shape**, which
  is a stronger condition than the query change having been deployed: a rolling
  deploy runs both shapes side by side, and against a residual index the old shape
  matches no HNSW index at all. "Deployed everywhere" is the weaker statement and
  not the constraint.

``base_vector_store.ensure_embedding_index`` owns the sequencing and states it in
its strict form -- read it there rather than from here, which describes only what
this module's own index covers.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from typing import Any

from sqlalchemy import text

from ..db import AI_SCHEMA
from .pg_bm25_index import (
    first_error_line,
    is_transient_db_error,
    partition_build_lock_sql,
    partition_build_unlock_sql,
)
from .settings_registry import SETTINGS_REGISTRY, get_setting

logger = logging.getLogger(__name__)

# The one item population these indexes cover. ``ai.embeddings`` is polymorphic --
# ``item_table`` is a NOT NULL column on it, and four item tables share the
# relation -- so an index whose predicate names only ``(knowledge_base_id, dims)``
# spans all four. A knowledge base then crossed the build threshold on the *sum*
# over its populations and got one index mixing them, and the chunk search this
# feature steers onto it walked entries that cannot join: measured at 1536
# dimensions, the same 1,000 chunk rows scored recall 0.858 indexed alone against
# 0.383 in an index that also held 9,000 ``full_documents`` rows, and 6,000 chunk
# rows 0.925 against 0.812 with 6,000 graph-node rows beside them, tail minimum
# 0.700 against 0.300.
#
# So the predicate names it, and both counts that decide eligibility are restricted
# to it, which makes a knowledge base's eligibility a fact about the population the
# index will actually cover.
#
# The index *name* is deliberately not qualified by it. There is only ever one
# population indexed, so there is nothing to disambiguate, and the name is the key
# every catalog lookup, drop path and failure record in this module is derived
# from -- a second component would be a rename of all of them for no gain. If a
# second population is ever indexed, that is when the name has to grow.
#
# This is the single source of the string: ``base_vector_store`` needs the same
# literal in the search query for the planner to be able to prove the predicate,
# and it already imports this module, so it imports this rather than restating it.
PER_KB_INDEX_ITEM_TABLE = "chunks"

# Index names are ``hnsw_kb_<32 hex>_<dims>``: 8 + 32 + 1 + 4 = 45 bytes at
# most, inside Postgres' 63-byte identifier limit, and free of the dashes a
# UUID's canonical form would need quoting for.
INDEX_NAME_PREFIX = "hnsw_kb_"

# pgvector's own bound on a vector's dimensions; the same range
# ``ensure_embedding_index`` enforces, and the range an index *name* is derived
# for -- an existing index has to be found and dropped whatever its dimension.
MIN_DIMS = 1
MAX_DIMS = 8192

# pgvector refuses an HNSW index on a vector wider than this, so a partial HNSW
# index above it cannot be built at all: the attempt fails, and under
# CONCURRENTLY it fails *after* the catalog entry exists, leaving an INVALID
# index behind that answers no query and is maintained on every write. Embedding
# models of 3,072 dimensions are in ordinary use and ``MAX_DIMS`` lets them
# through, which is why this is a second, lower limit rather than a tightening of
# that one. A knowledge base above it is declined once, at WARNING, and
# ``index_action`` stops asking -- otherwise every source that finished indexing
# dispatched the same doomed build, each attempt holding a worker slot with no
# statement timeout. Its searches stay exact: the shared per-dimension index is
# HNSW too, so there is no index of any kind to fall back to at this width.
MAX_HNSW_DIMS = 2_000

# Ceiling on how many of these a single project may hold. The planner opens and
# locks *every* index of a relation while planning any query on it, so partial
# indexes on one table are not free at scale: measured on a Postgres 15 with
# default ``max_locks_per_transaction``, 500 of them cost 1.9 ms of planning and
# 513 locks per backend (comfortable), while 5,000 cost 25 ms of planning and
# made the seventh concurrent search fail with "out of shared memory". 200 is
# far below the point where either matters, and at the default threshold it
# already means two million indexed rows in one project. A project that reaches
# it keeps the shared index for the rest of its knowledge bases and says so at
# WARNING, rather than quietly degrading every query on the table.
#
# The cap is soft under concurrency: two workers reconciling different knowledge
# bases hold different locks and count the same catalog, so the overshoot is
# bounded by one index per worker running at that moment, which is far inside the
# margin above.
MAX_PER_KB_INDEXES = 200

# How many consecutive failed builds of one index are attempted before the
# reconcile gives up on it. A build can fail for a reason no retry gets past --
# a disk with no room for a 573 MB index is the obvious one -- and every source
# that finishes indexing dispatches another reconcile, so
# without a bound a permanently failing build is an unbounded loop of
# drop-rebuild-fail, each attempt holding a worker slot with no statement
# timeout and reaching for the disk that was full the last time.
#
# ``MAX_HNSW_DIMS`` guards the one failure that can be predicted from the
# catalog. This bounds the *permanent-looking* ones -- a full disk is the
# obvious one -- which is the class a retry is not expected to get past.
# ``MAX_CONSECUTIVE_INTERRUPTED_BUILDS`` bounds the other class. Between them
# every attempt that can write its own record is counted against something; an
# earlier version of this comment claimed this constant "guards the rest, which
# can only be learnt by trying", and it did not -- a build that failed
# transiently was counted against nothing at all, so the bound was unreachable
# and the loop it exists to stop ran for ever (see the other constant).
#
# Three rather than one because a permanent-looking failure may still be a
# one-off (a disk that was full and has been cleared), and because the only
# remedy for reaching the bound is a manual ``DROP INDEX``.
#
# It only covers a build that got as far as creating a catalog entry, which is
# where the count is kept: a ``CREATE INDEX CONCURRENTLY`` that fails before that
# -- a syntax error, a missing relation, a refused permission -- leaves nothing to
# write the count on, so those failures are unbounded here and bounded only by the
# task's own retry policy.
#
# ``XX000 internal_error`` is counted here, and the BM25 sibling deliberately
# excuses it (``Bm25IndexBuildFailed``, for a known pg_search bug that makes a
# concurrent build fail under writes). The two disagree on purpose: that bug is in
# the bm25 index type, and an HNSW build has no equivalent, so here an internal
# error is a failure like any other. If pgvector ever grows one, this is where the
# exception would go.
MAX_CONSECUTIVE_BUILD_FAILURES = 3

# The same bound for a build that failed for a reason ``is_transient_db_error``
# *does* recognise -- a lock conflict, a deadlock, a cancelled statement, a lost
# connection. Separate and far larger, rather than not counted at all.
#
# Not counted at all was the previous answer, and it made the whole bound
# unreachable: ``_repair_invalid``'s drop takes the record away with the index it
# is written on, so a build that failed transiently wrote nothing and the next
# reconcile started again from zero. Measured before this constant existed --
# seven consecutive reconciles of an INVALID index whose build fails ``55P03``
# every time: failure comments written ``[]`` on all seven, recorded failures
# ``0`` against a bound of ``3``, and ``index_action`` asking for a build again
# every time. That is a drop-rebuild-fail loop with nothing bounding it, on every
# source that finishes indexing and every boot.
#
# Separate and larger is what keeps the reason the count used to skip these: the
# task that runs these builds retries a transient failure
# ``PG_BM25_TASK_MAX_RETRIES`` times, so one contention episode is about seven
# attempts, and a single episode must not turn the index off until an operator
# drops it by hand. 25 is more than three such episodes back to back, and still a
# bound.
#
# The largest source of these on a project with several large knowledge bases was
# this module fighting itself: a ``deadlock detected`` at the end of every build
# that another knowledge base's DDL had queued behind (issue #95), each one counted
# here. ``_TableGate`` removes that source; a table another build owns is now a
# ``table_busy`` outcome, which is not an attempt and is counted nowhere.
#
# **What this does not cover:** a build whose backend does not come back -- an OOM
# kill, a server restart -- cannot write anything, because the write needs the
# connection that just died, and the record it would have updated was destroyed by
# the repair drop before the build started. So that one failure mode still resets
# the count on every reconcile. Closing it needs a record that outlives the index,
# which a ``pg_class`` comment on the index cannot be; it is not fixed here, and
# it is the reason ``VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB`` has to be set against
# the database's real memory rather than to the registry maximum.
MAX_CONSECUTIVE_INTERRUPTED_BUILDS = 25

# The drift path's own bound, and the **third** of these -- deliberately neither of
# the other two, and it counts something neither of them counts.
#
# A drift drop is charged to neither build bound, correctly: nothing failed, the
# index was working, and the reason it is going away is that this module's own DDL
# changed, so spending a budget whose only remedy is a manual ``DROP INDEX`` would
# let three predicate changes turn the feature off (``_drop_a_drifted_index``). The
# consequence is that the *only* thing stopping the next reconcile calling the same
# index stale again is the fingerprint the rebuild records -- and that write is
# ``_write_index_comment``, which is deliberately best effort. Reproduced two ways:
# with that write no-op'd, 6 reconciles produced 6 distinct index oids, failures 0
# against a bound of 3; with ``COMMENT`` refused server-side, 4 consecutive
# rebuilds with ``status`` staying ``ready`` throughout. Each iteration is a real
# ``DROP`` plus ``CREATE INDEX CONCURRENTLY`` with ``statement_timeout = 0``,
# ``lock_timeout = 0`` and up to 4 GB of ``maintenance_work_mem``. That is the
# unbounded drop-and-rebuild property whose absence was the whole reason for
# choosing a fingerprint over comparing ``pg_get_indexdef``, arriving through a
# different door.
#
# So this count is written **before** the drop, on the index about to be replaced,
# and read straight back (``_count_a_definition_rebuild``); it is then carried in
# memory across the drop and written back by whatever ends the attempt -- the
# rebuild that succeeds (``_record_the_definition_built``), the build that fails
# (``_count_a_failed_build``) or the repair drop that fails
# (``_record_build_failure``). Three different loops are bounded between the
# read-back, the number that survives a drop which fails, and the number carried
# across a drop which does not -- and it takes all three:
#
# * The read-back is what bounds a comment that does not stick. A write that
#   quietly does nothing cannot be counted -- the count needs the same write -- so
#   the only sound answer is not to start work whose termination depends on a
#   record this module has just failed to make. Both reproductions above end at
#   *zero* rebuilds with the read-back in place. What it bounds is the destructive
#   work and not the dispatch: ``index_action`` reads the same unwritable comment,
#   so a source that finishes indexing still asks for a reconcile, which now costs
#   a catalog read, a bounded count and a refused write rather than a drop and a
#   rebuild. A database where this module cannot comment is one where it has no
#   memory at all, so that is the most any record kept on the object can do.
# * The count is what bounds a drift *drop* that keeps failing, which nothing
#   bounded before: the drop raising leaves the index and this comment in place, so
#   the next reconcile reads the number and adds to it, and the fourth keeps the
#   stale index instead. It also catches a comment writer that works sometimes.
# * Carrying it across the drop is what bounds a **rolling-deploy rebuild war**:
#   two pod versions computing two different fingerprints, each reading the other's
#   index as drifted and each rebuilding it. Every rebuild there *lands*, so this
#   is the one shape where nothing fails and the loop still runs; what makes it
#   terminate is that the number rides through the successful rebuild instead of
#   being zeroed by it.
#
# **What clears it, and why not a rebuild that lands.** Only a later reconcile that
# finds the index already built from today's definition
# (``_settle_a_definition_rebuild``). A rebuild that lands proves a
# ``CREATE INDEX CONCURRENTLY`` finished and a ``COMMENT`` stuck; it does not prove
# that the definition being emitted has stopped moving, and in the war above every
# rebuild lands, so clearing on that alone leaves the count at zero for ever --
# measured at 8 reconciles, 8 distinct index oids and a count that never left 0.
# The number is therefore "consecutive rebuilds started and not yet confirmed
# settled", not "consecutive rebuilds that did not land". Clearing it on *something*
# is not optional -- the budget is spent one per definition change and its only
# remedy is an operator clearing the comment -- which is why the boot sweep
# dispatches an index carrying an unconfirmed count: nothing else brings a reconcile
# back to a knowledge base whose index is already correct, and without that pass a
# handful of ordinary definition changes over a project's life would spend the whole
# bound.
#
# 3 rather than 25: unlike an interrupted build there is no retrying task multiplying
# each episode, and unlike a failed build the outcome of giving up is mild -- the
# index stays and answers searches at whatever recall its old definition gives, which
# is the state the whole drift mechanism starts from. Nothing needs an operator to
# drop anything for the searches to keep working.
#
# **What this still cannot bound, and it is the same limit as the one
# ``MAX_CONSECUTIVE_INTERRUPTED_BUILDS`` names: a worker that dies between the
# rebuild and the ``COMMENT`` that records it.** A minutes-long
# ``CREATE INDEX CONCURRENTLY`` followed by a SIGTERM, an eviction or an OOM kill
# leaves a new index with no comment at all -- which reads as drifted, so the next
# reconcile rebuilds it, and so on. Reproduced at 5 reconciles and 5 distinct index
# oids. Carrying the count does not help here and nothing kept on the index can: the
# drop destroyed the old record, and the only statement that could put the number on
# the new index is exactly the one that did not run. Closing it needs a record that
# outlives the index, which a ``pg_class`` comment on the index cannot be.
#
# **And the war's bound is a bound, not a guarantee.** The pass that clears the count
# is a pass that sees no drift, and in a mixed fleet the version whose fingerprint is
# currently on the index sees none -- so a reconcile that lands on one of its workers
# gives the budget back and the war starts again from zero. It advances whenever
# consecutive passes see drift, and terminates rather than running for ever, but the
# number of real rebuilds it takes to get there is not 3 -- measured at 3 to 11 across
# six trials of a fleet where either version may pick the task up, terminating in every
# one, against exactly 3 when the versions strictly alternate. The real answer to a war is
# still sequencing rather than a constant: let the new definition reach every process
# before any process acts on the drift it sees, which is the same shape as the
# constraint in the module docstring above, for the same reason. *This* change is
# immune to all of it, because the code it replaces records no fingerprint at all, so
# old pods never see today's index as drifted and never fight back.
MAX_CONSECUTIVE_DEFINITION_REBUILDS = 3

# Where that count is kept. There is no builds table -- the logs are the whole
# of the history -- and an in-process counter would forget on every deploy,
# every worker restart and every other worker, which is exactly the population
# this loop runs across. The index's own catalog comment is durable, needs no
# migration, and is scoped to precisely the object it is about: it exists as
# soon as the failed ``CREATE INDEX CONCURRENTLY`` leaves the INVALID index
# behind, and it goes away with that index, so an operator dropping the index by
# hand re-arms the build without knowing this marker is here.
#
# Nothing in the wording may be ``:word``: ``COMMENT ON`` takes no parameter, so
# this is interpolated, and ``text()`` would read that as a bind parameter.
#
# Four independent facts share the one comment, so it is composed of sentences
# and each is read back by a pattern of its own rather than the whole comment
# being matched at once. The counts keep their fixed order -- failed, then
# interrupted, then unsettled rebuilds -- so an operator reads the same shape every
# time, and the fingerprint comes last because it is the one fact a *valid* index
# carries.
#
# The prose about an INVALID index comes after all three counts rather than between
# them. It used to sit before the rebuild count, which put a sentence describing "a
# valid index still answering searches" inside prose calling the index INVALID --
# and that pair is producible here, by a repair drop that fails after a drift drop
# (``_count_a_failed_repair_drop`` reads the rebuild count back and keeps it). Two
# incompatible claims about the same object in one string, in the comment an
# operator reads first.
_BUILD_FAILURES_SENTENCE = "{n} consecutive failed attempts to build this partial HNSW index."
_INTERRUPTED_BUILDS_SENTENCE = "{n} consecutive builds of this partial HNSW index were interrupted."
_INVALID_INDEX_PROSE = (
    "The last build attempt left it INVALID, and while it is, it answers no query and "
    "is maintained on every write. Drop it once the cause is fixed; the next reconcile "
    "then builds it again. A REINDEX makes it valid without clearing this comment, so "
    "every count above is a history of attempts and not a claim about the index now. "
    "The definition below is a claim about the index now: it is what the index on disk "
    "was built from, reindexed or not."
)
_REBUILDS_SENTENCE = (
    "{n} consecutive rebuilds of this partial HNSW index for a definition change have not settled."
)
_DEFINITION_SENTENCE = "Built from definition {fp}."

# Every comment this module writes starts with this, and every reader refuses one
# that does not.
#
# The four sentences above are ordinary English about an ordinary object, so a
# runbook note, a hand-off or a ticket summary written into the same comment can
# quote one of them back verbatim -- and the fourth fact is the one where being
# misread is silent and permanent. A note quoting the rebuild sentence reads as
# ``MAX_CONSECUTIVE_DEFINITION_REBUILDS`` consecutive rebuilds and freezes the index
# for ever with no log line, where the equivalent build-count quote is loud (the
# give-up ERROR names the index and both counts) and self-limiting (the operator
# drop it asks for re-arms the build).
#
# Anchoring the four patterns to sentence starts would close a sentence quoted
# mid-comment and not one quoted at the front of it; requiring this closes all four
# at once, in one place. It fails in the safe direction: a comment this module
# cannot vouch for records nothing, so the index reads as one with no history and no
# definition -- the state every reader here already handles, and the one an index
# built before this module existed is in.
#
# It is not a signature and cannot be one: an operator may copy it along with the
# rest, and a ``pg_class`` comment is a place anyone may write. What it separates is
# a comment this module composed from a comment that merely talks about it.
_COMMENT_MARKER = "[per-kb-hnsw]"

# Each pattern matches its own whole sentence, not a prefix of it, because the
# sentences now sit beside each other and beside prose. The original count
# pattern was anchored at the start of the comment, which composition makes
# impossible; a long distinctive phrase is the stricter test anyway -- "a comment
# is a place anyone may write, and one this module did not write is no evidence".
#
# None of them is ever run against a comment that does not start with
# ``_COMMENT_MARKER`` (``_module_written``), which is what keeps "a long
# distinctive phrase" from being a phrase anyone can write back.
_BUILD_FAILURES_PATTERN = re.compile(
    r"(\d+) consecutive failed attempts to build this partial HNSW index\."
)
_INTERRUPTED_BUILDS_PATTERN = re.compile(
    r"(\d+) consecutive builds of this partial HNSW index were interrupted\."
)
_REBUILDS_PATTERN = re.compile(
    r"(\d+) consecutive rebuilds of this partial HNSW index for a definition change "
    r"have not settled\."
)
_DEFINITION_PATTERN = re.compile(r"Built from definition ([0-9a-f]{12})\.")

# Bounded counts never read more than this many rows past the threshold, so the
# cost of deciding is bounded by the threshold rather than by the size of the
# knowledge base.
_COUNT_HEADROOM = 1

# Rough bytes of index per vector dimension, for the size a build is about to
# reach for. Derived from one measurement -- 573 MB for 73,290 vectors at 1536
# dimensions, i.e. about 5.3 bytes per dimension: four for the float plus HNSW's
# own links. Only ever used in a log line, so being a little wrong is fine.
_INDEX_BYTES_PER_DIMENSION = 6


class PerKbVectorIndexBuildInProgress(RuntimeError):
    """Another caller holds the build lock for this index."""


class PerKbVectorIndexTableBusy(PerKbVectorIndexBuildInProgress):
    """Another build or drop owns ``ai.embeddings`` right now; nothing was issued.

    ``holders`` is the evidence (``table_ddl_holders``): the other backends holding
    a lock on the table that a ``CREATE``/``DROP INDEX CONCURRENTLY`` would queue
    behind. ``build_alive`` is whether there is any -- the one thing the task spends
    its uncounted wait on. ``lock_held`` says which way the gate was refused: this
    module's own table lock held by another caller, or free and the table held by
    something that does not take it (a manual ``DROP INDEX CONCURRENTLY``, a worker
    from before the gate existed).

    A subclass of the per-index one so that every caller already retrying a held
    lock retries this too; the ones that can wait longer tell them apart.
    """

    def __init__(
        self,
        message: str,
        holders=(),
        lock_held: bool = True,
        evidence_error: str | None = None,
    ):
        super().__init__(message)
        self.holders = list(holders or ())
        self.lock_held = lock_held
        self.evidence_error = evidence_error

    @property
    def build_alive(self) -> bool:
        """Is any holder demonstrably *running* (``_holder_kind``)?

        Not "is there any holder": a session idle in a transaction that happens to
        hold the table, or one this role cannot see, refuses the gate just the same
        but is not evidence that anything will finish, so it does not buy the long
        uncounted wait.
        """
        return any(holder.get("kind") == "running" for holder in self.holders)


class PerKbVectorIndexTableWaitExhausted(RuntimeError):
    """A task gave up waiting for ``ai.embeddings`` after its whole wait budget.

    Raised by the task after its ERROR rather than returned, so the task ends in
    failure and a failure-rate alert sees it: a returned dict is a SUCCESS to every
    monitor that reads task states, which is what the give-up used to be.
    """


class PerKbVectorIndexDropFailed(RuntimeError):
    """An index could not be dropped for a reason a retry cannot get past.

    Raised rather than reported because the only caller is the deleted knowledge
    base's drop, where nothing comes back: the row these indexes are named after
    is gone, so neither the indexing dispatch nor the start-up sweep will ever
    look at them again.
    """

    def __init__(self, message: str, dropped_indexes=(), failed_indexes=()):
        super().__init__(message)
        self.dropped_indexes = list(dropped_indexes)
        self.failed_indexes = list(failed_indexes)


# ---------------------------------------------------------------------------
# Naming and DDL
# ---------------------------------------------------------------------------


def _validated_kb_id(knowledge_base_id: Any) -> str:
    """Canonical UUID string for a KB id, or ValueError.

    The one gate between a caller's string and a SQL literal.
    """
    try:
        return str(uuid.UUID(str(knowledge_base_id)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"knowledge_base_id is not a UUID: {knowledge_base_id!r}") from exc


def _validated_dims(dims: Any) -> int:
    """An embedding dimension safe to interpolate into a type modifier.

    PostgreSQL does not accept a type modifier from a parameter, so ``dims``
    reaches SQL as a literal in both the index expression and its predicate.

    ``int()`` coerces rather than rejects, so a float truncates (``1536.9`` ->
    ``1536``) and ``True`` becomes ``1``. Surprising, and left alone: the value
    is range-checked either way, so nothing unsafe reaches SQL, and no caller
    can get here with a non-integer -- ``dims`` comes from an embedding vector's
    length or from ``pg_class.relname``.
    """
    try:
        value = int(dims)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dims is not an integer: {dims!r}") from exc
    if not (MIN_DIMS <= value <= MAX_DIMS):
        raise ValueError(f"dims must be between {MIN_DIMS} and {MAX_DIMS}, got {value}")
    return value


def per_kb_index_name(knowledge_base_id: Any, dims: Any) -> str:
    """Deterministic index name for one knowledge base and dimension."""
    kb_hex = uuid.UUID(_validated_kb_id(knowledge_base_id)).hex
    return f"{INDEX_NAME_PREFIX}{kb_hex}_{_validated_dims(dims)}"


def per_kb_index_ddl(knowledge_base_id: Any, dims: Any) -> str:
    """CREATE statement for one knowledge base's partial HNSW index.

    ``dims`` is in the predicate as well as the expression. Without it, a
    knowledge base holding rows of two different dimensions would fail the
    build outright -- ``embedding::vector(1536)`` raises on a 768-dimension
    value -- and the index would cover rows the query's own ``e.dims``
    predicate excludes.

    ``item_table`` likewise, because ``ai.embeddings`` is polymorphic and this
    index covers one population of it; see ``PER_KB_INDEX_ITEM_TABLE`` for what
    mixing them cost. The search query has to carry the same literal or the
    planner cannot prove the predicate and matches no partial index at all -- a
    half-landed change is slow, not wrong.

    The operator class and the cast match the shared per-dimension index
    exactly. Both are load-bearing: the index is on an *expression*, so a query
    whose ``ORDER BY`` does not contain the same ``::vector(N)`` cast matches no
    HNSW index at all.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    n = _validated_dims(dims)
    return (
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {per_kb_index_name(kb_id, n)} "
        f'ON "{AI_SCHEMA}".embeddings '
        f"USING hnsw ((embedding::vector({n})) vector_cosine_ops) "
        f"WHERE knowledge_base_id = '{kb_id}' AND dims = {n} "
        f"AND item_table = '{PER_KB_INDEX_ITEM_TABLE}'"
    )


def per_kb_index_drop_ddl(knowledge_base_id: Any, dims: Any) -> str:
    """DROP statement for one knowledge base's partial HNSW index."""
    name = per_kb_index_name(knowledge_base_id, dims)
    return f'DROP INDEX CONCURRENTLY IF EXISTS "{AI_SCHEMA}".{name}'


# ``_`` matches any single character in a LIKE pattern and ``%`` any run of
# them, and every one of these index names carries two underscores. Escaped, the
# catalog lookups below match the literal names they are derived from and
# nothing else, which is what their docstrings claim.
_LIKE_WILDCARDS = str.maketrans({"\\": "\\\\", "_": "\\_", "%": "\\%"})


def _like_prefix(prefix: str) -> str:
    """``prefix`` as a LIKE pattern matching it literally, for ``ESCAPE '\\'``."""
    return prefix.translate(_LIKE_WILDCARDS) + "%"


def index_lock_relation(knowledge_base_id: Any, dims: Any) -> str:
    """Advisory-lock subject for building or dropping one of these indexes."""
    return f"{AI_SCHEMA}.{per_kb_index_name(knowledge_base_id, dims)}"


def table_lock_relation() -> str:
    """Advisory-lock subject for *any* of these builds or drops: one per table.

    Every index this module creates lives on ``ai.embeddings``, so this is one
    subject for the whole project, taken on top of ``index_lock_relation`` before
    each ``CREATE``/``DROP INDEX CONCURRENTLY`` -- see ``_TableGate`` for why.

    Suffixed rather than the bare table name, the convention the BM25 path's move
    gate (``ai.chunks#move``) set: its per-item-table build lock *is* the bare
    name, and a gate that only means "one index DDL at a time" must not share a
    key with a lock that means something else if ``embeddings`` ever becomes a
    table that path manages.
    """
    return f"{AI_SCHEMA}.embeddings#index-ddl"


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


# Upper bound on how long the start-up sweep's settings read waits for a lock
# on ``ai.project_settings``. Deliberately the same 5 s as ``SWEEP_TIMEOUT_MS``
# and for the same reason: nothing a start-up does may wait without a bound.
#
# ``statement_timeout`` would not do on its own. This read is milliseconds of
# work; what makes it slow is queueing, and queueing is what ``lock_timeout``
# bounds while leaving a read that is merely slow alone.
SETTINGS_READ_LOCK_TIMEOUT_MS = 5_000


def read_overrides(conn, *keys: str) -> dict[str, int]:
    """These settings' stored overrides, read on a caller-supplied connection.

    ``get_setting`` reads through ``db.session``, and a read of
    ``ai.project_settings`` that fails -- the table does not exist yet, which is
    exactly the state at the start-up sweep's first run on a new database --
    leaves that session's transaction aborted and takes the caller's next
    statement with it. At start-up ``db.session`` is the boot's own transaction,
    mid-flight and about to be committed, so the sweep reads its settings here
    instead, on a connection of its own, and falls back to the registry defaults
    if that read fails too.

    ``lock_timeout`` because this is the one statement in the sweep that touches
    a table the rest of the system takes DDL locks on. A nightly dump or other
    long-lived snapshot holds a share lock for hours; a single ``ALTER`` queued
    behind it then blocks every later reader, this one included -- and an
    unbounded wait here is a start-up that never finishes, never passes
    readiness, is restarted, and hangs again, with nothing in the log to say
    why. Bounded, the wait becomes an error, the ``except`` below turns it into
    the registry defaults, and the sweep carries on: a start-up that is a little
    wrong about a threshold is worth far more than one that does not happen.
    ``set_config(..., true)`` scopes it to this transaction, so nothing the
    caller does afterwards inherits it -- which does mean the connection has to
    be a transactional one, as the sweep's is. On a connection in AUTOCOMMIT the
    scope would end with the ``set_config`` statement itself and the read would
    be unbounded again, and a session-level ``SET`` is not the answer either: it
    would ride a pooled connection out into unrelated work.
    """
    try:
        conn.execute(
            text("SELECT set_config('lock_timeout', :ms, true)"),
            {"ms": str(SETTINGS_READ_LOCK_TIMEOUT_MS)},
        )
        rows = conn.execute(
            text(f'SELECT key, value FROM "{AI_SCHEMA}".project_settings WHERE key = ANY(:keys)'),
            {"keys": list(keys)},
        ).all()
        conn.rollback()
    except Exception as exc:
        conn.rollback()
        logger.debug(
            "Could not read the vector index settings (%s); using the defaults",
            first_error_line(exc),
        )
        return {}
    found: dict[str, int] = {}
    for key, value in rows:
        try:
            found[key] = int(value)
        except (TypeError, ValueError):
            logger.warning("Bad stored value for %s=%r, using the default", key, value)
    return found


def _clamped_setting(key: str, overrides: dict[str, int] | None = None) -> int:
    """Read an int setting, clamped to the registry's own bounds.

    ``get_setting`` coerces a stored override but does not range-check it --
    that happens on the settings PUT path only -- so a row written before a
    bound was tightened, or by hand, would otherwise be used as-is.

    ``overrides``, when given, replaces the ``get_setting`` read entirely: a
    caller that cannot use ``db.session`` passes what ``read_overrides`` found.
    """
    defn = SETTINGS_REGISTRY[key]
    if overrides is None:
        value = int(get_setting(key))
    else:
        value = int(overrides.get(key, defn.default))
    clamped = value
    if defn.min is not None:
        clamped = max(clamped, int(defn.min))
    if defn.max is not None:
        clamped = min(clamped, int(defn.max))
    if clamped != value:
        logger.warning(
            "%s=%d is outside the allowed range %s-%s; using %d instead",
            key,
            value,
            defn.min,
            defn.max,
            clamped,
        )
    return clamped


def thresholds(overrides: dict[str, int] | None = None) -> tuple[int, int]:
    """``(build_at_or_above, drop_below)`` rows, with the hysteresis enforced.

    Two thresholds rather than one so a knowledge base sitting at the boundary
    cannot have its index built and dropped over and over: each build costs a
    full index build and each drop throws it away. A stored pair that inverts
    the hysteresis (drop at or above build) would do exactly that, so the drop
    threshold is pulled down to half the build threshold and the override is
    reported. Never below 1: the registry gives the drop setting a minimum of 1
    because at 0 the drop test can only be satisfied by a knowledge base with no
    embeddings at all -- an index held open for a handful of rows, and paid for
    on every write -- and a substituted value is no more allowed to be 0 than a
    stored one is.

    The crossover where a partial index starts beating an exact scan was
    bracketed, not bisected: at 2,000 rows the planner does not use a partial
    index at all (an exact bitmap scan and sort is genuinely cheaper, and
    exact), and at 73,290 rows the partial index is two orders of magnitude
    faster.

    **The 50,000 default deliberately leaves a regression window unindexed, and a
    project with a knowledge base in it should lower this setting.** The
    embeddings-side predicate applies from the moment this deploys, with or
    without a partial index, and in roughly the 10,000-25,000-row band at 1536
    dimensions it replaces a fast approximate plan with a slow exact one: measured
    6.18 ms at recall 0.608 against 50.35 ms at recall 1.000 for 10,000 rows, and
    4.35 ms at 0.575 against 132.48 ms at 1.000 for 20,000. The same 10,000-row
    knowledge base *with an index of its own* answers in 1.83 ms at recall 1.000 --
    so at the default it is 27x slower than it needs to be, for nothing.

    Neither edge of "roughly 10,000-25,000" was measured: 20,000 was slow, 30,000
    was unchanged, so 25,000 is an interpolation between them, and nothing below
    10,000 was tried. And the band is a property of the *fixture*, existence
    included -- an earlier version of this said its existence was not, which
    contradicts the module docstring eight hundred lines above and is wrong the same
    way: the flip is a cost race decided by table shape, and one independently built
    fixture measured the new shape *faster* at the same recall with no window at
    all. So this is a shape to look for in a project, not a range to trust. The
    default is high anyway, because an
    index that is built and never scanned is paid for on every write (see the
    module docstring), so the remedy is per project: measure that the index is
    really scanned, then lower ``VECTOR_PER_KB_INDEX_MIN_ROWS`` past the knowledge
    base's size. A project that measures its own crossover can move both.

    ``overrides`` is for a caller that cannot read settings through
    ``db.session``; see ``read_overrides``.
    """
    build_at = _clamped_setting("VECTOR_PER_KB_INDEX_MIN_ROWS", overrides)
    drop_below = _clamped_setting("VECTOR_PER_KB_INDEX_DROP_ROWS", overrides)
    if drop_below >= build_at:
        corrected = max(1, build_at // 2)
        logger.warning(
            "VECTOR_PER_KB_INDEX_DROP_ROWS=%d is not below VECTOR_PER_KB_INDEX_MIN_ROWS=%d, "
            "which would build and drop the same index repeatedly; using %d instead",
            drop_below,
            build_at,
            corrected,
        )
        drop_below = corrected
    return build_at, drop_below


def maintenance_work_mem_mb() -> int:
    """How much memory the builder gives its own session for the build.

    A build that does not fit spills, and says so
    (``NOTICE: hnsw graph no longer fits into maintenance_work_mem``): measured
    3.3x slower at 64 MB than at 1 GB for a 73,290-row, 1536-dimension index, so
    raising it is worth real time. The default is deliberately modest because
    this is memory the database has to have on top of ``shared_buffers``, and
    the smallest project databases have 512 MiB in total -- for those, the
    default is already about as far as it goes.

    The registry's maximum is **not** a safety bound, and an earlier version of
    this docstring wrongly claimed it was: 4096 MB against a 512 MiB database is
    an eightfold overcommit, and nothing here can see how much memory the server
    actually has (``SHOW shared_buffers`` is a fraction of it, not the total, and
    a container's limit is not visible from SQL at all). The clamp enforces the
    registry range and no more. An operator raising this has to know the
    database's own memory; the range exists so a typo cannot ask for terabytes.
    """
    return _clamped_setting("VECTOR_INDEX_MAINTENANCE_WORK_MEM_MB")


# ---------------------------------------------------------------------------
# Reading the current state
# ---------------------------------------------------------------------------


def parsed_per_kb_index_name(relname: str) -> tuple[str, int] | None:
    """``(knowledge base id, dims)`` for one of these index names, or None.

    The one parse of the name, shared by all three catalog readers here. They used
    to disagree: this module's per-knowledge-base reader required the dimension
    suffix to be digits while the project-wide count matched the prefix alone, so
    the day the name grows a component the count would still see an index the
    lifecycle had gone blind to -- consuming a place in ``MAX_PER_KB_INDEXES``
    that nothing could ever free.

    (A component *cannot* simply be added, which is the other half of that note:
    ``hnsw_kb_`` + 32 hex + ``_doc2json_documents`` + ``_1536`` is 64 bytes, one
    over PostgreSQL's identifier limit, so a literal item-table name does not fit
    and the name would have to change shape rather than grow.)
    """
    if not relname.startswith(INDEX_NAME_PREFIX):
        return None
    kb_hex, _, dims_part = relname[len(INDEX_NAME_PREFIX) :].rpartition("_")
    if len(kb_hex) != 32 or not dims_part.isdigit():
        return None
    try:
        return str(uuid.UUID(hex=kb_hex)), int(dims_part)
    except ValueError:
        return None


# The one catalog read behind every question this module asks about its own
# indexes: which ones a knowledge base has, whether each is valid, what each was
# built from and how its builds have gone, and how many the project holds. One
# statement so the three callers cannot drift apart about what counts as one of
# these indexes -- they did, and the note on ``parsed_per_kb_index_name`` says
# what that cost. ``prefix`` is the only difference between them: one knowledge
# base's indexes, or the whole project's.
_INDEX_CATALOG_SQL = (
    "SELECT c.relname, i.indisvalid, obj_description(c.oid, 'pg_class') FROM pg_class c "
    "JOIN pg_index i ON i.indexrelid = c.oid "
    "JOIN pg_namespace n ON n.oid = c.relnamespace "
    r"WHERE n.nspname = :schema AND c.relkind = 'i' "
    r"AND c.relname LIKE :prefix ESCAPE '\'"
)


def _index_catalog_rows(conn, prefix: str) -> list:
    """``(relname, indisvalid, comment)`` for every index whose name starts with ``prefix``."""
    return conn.execute(
        text(_INDEX_CATALOG_SQL), {"schema": AI_SCHEMA, "prefix": _like_prefix(prefix)}
    ).all()


def per_kb_index_states(conn, knowledge_base_id: Any) -> dict[int, tuple[bool, str | None]]:
    """``{dims: (is_valid, comment)}`` for this knowledge base's partial HNSW indexes.

    Read by name rather than by parsing predicates: the name is derived from
    the KB's UUID and the dimension, so the catalog lookup is a literal prefix
    match on ``pg_class.relname`` -- wildcards escaped, so the underscores in
    the name are underscores -- and cannot mistake another KB's index for this
    one's.

    The comment comes back in the same pass because the name is not enough to say
    what an index *is*: ``per_kb_index_ddl`` decides the predicate, the expression
    and the operator class, and none of those are in the name. The comment is where
    the definition the index was built from is recorded
    (``per_kb_index_fingerprint``), alongside its build history -- which is why
    this reads it here rather than making a round trip per dimension, exactly as
    the boot sweep already does.
    """
    kb_hex = uuid.UUID(_validated_kb_id(knowledge_base_id)).hex
    rows = _index_catalog_rows(conn, f"{INDEX_NAME_PREFIX}{kb_hex}_")
    found: dict[int, tuple[bool, str | None]] = {}
    for relname, valid, comment in rows:
        parsed = parsed_per_kb_index_name(relname)
        if parsed is not None:
            found[parsed[1]] = (bool(valid), comment)
    return found


def existing_per_kb_indexes(conn, knowledge_base_id: Any) -> dict[int, bool]:
    """``{dims: is_valid}`` for this knowledge base's partial HNSW indexes.

    What a caller that only needs to know whether an index is there and usable
    reads; ``per_kb_index_states`` is the same query with the comment too.
    """
    return {
        dims: valid for dims, (valid, _) in per_kb_index_states(conn, knowledge_base_id).items()
    }


def per_kb_index_count(conn) -> int:
    """How many of these indexes ``ai.embeddings`` already carries.

    Counted through the same name parse the per-knowledge-base reader uses, so the
    budget and the lifecycle cannot disagree about what one of these indexes is.
    The names are fetched rather than counted in SQL because the parse is not
    expressible as a ``LIKE``; ``MAX_PER_KB_INDEXES`` bounds how many rows that
    can be.
    """
    rows = _index_catalog_rows(conn, INDEX_NAME_PREFIX)
    return sum(1 for row in rows if parsed_per_kb_index_name(row[0]) is not None)


def bounded_row_count(conn, knowledge_base_id: Any, dims: Any, cap: int) -> int:
    """Indexable rows this KB has at this dimension, counted no further than ``cap``.

    The decision only needs to know which side of a threshold the count falls
    on, so the count stops there. Without the bound this would read every
    embedding of the largest knowledge base in the project on every source that
    finishes indexing.

    Restricted to ``PER_KB_INDEX_ITEM_TABLE``, because that is the population the
    index will cover: counting the others too let a knowledge base cross the
    threshold on the sum and be given an index that mostly indexes rows its
    searches cannot use. It cuts the same way for the drop threshold -- a
    knowledge base whose chunks are gone loses the index even if its other
    populations are large, which is right, because the index covered only the
    chunks.

    **Not the index-only read an earlier version of this docstring claimed, and
    the ``LIMIT`` does not bound what it reads.** Measured on 60,000 rows (30,000
    chunks, 30,000 ``full_documents``) with the two btrees this repo's own live
    fixture creates -- ``(item_id)`` and ``(knowledge_base_id)`` -- this plans as a
    sequential scan with all three predicates as *heap* filters:
    ``Rows Removed by Filter: 30000``, ``Buffers: shared hit=3737``, 10.8 ms. The
    ``LIMIT`` bounds the rows it *returns*, not the rows it reads to find them, so a
    knowledge base with a large other population and few chunks reads the whole
    slice. The ``item_table`` clause made this strictly worse by adding a second
    unindexed filter, for a count ``index_action`` runs once per source that
    finishes indexing.

    A btree on ``(knowledge_base_id, item_table, dims)`` would make it the
    index-only read the decision wants. **Deliberately not added here:** no
    migration in this repository creates ``ai.embeddings``, so there is nowhere in
    this PR to put one, and which btrees a real project database already carries
    could not be confirmed from here -- the numbers above are against this repo's
    fixture and nothing else. It belongs in the follow-up that owns the table.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    return int(
        conn.execute(
            text(
                "SELECT count(*) FROM (SELECT 1 FROM "
                f'"{AI_SCHEMA}".embeddings '
                "WHERE knowledge_base_id = CAST(:kb AS uuid) AND dims = :dims "
                "AND item_table = :item_table "
                "LIMIT :cap) s"
            ),
            {
                "kb": kb_id,
                "dims": _validated_dims(dims),
                "cap": max(1, int(cap)),
                "item_table": PER_KB_INDEX_ITEM_TABLE,
            },
        ).scalar()
        or 0
    )


def candidate_dims(conn, knowledge_base_id: Any, cap: int) -> list[int]:
    """Dimensions this knowledge base has enough rows at to be worth looking at.

    Also bounded: the subquery reads at most ``cap`` rows, in heap order. Two
    consequences, both in the safe direction and both deliberate. A knowledge
    base holding two dimensions splits that budget between them, so each looks
    smaller than it is; and a dimension whose rows all sit past ``cap`` is not
    seen at all. Either way the knowledge base keeps the shared index, which is
    what it has today. In practice a knowledge base holds one embedding model at
    a time -- a model change reindexes it -- so it has one dimension.

    A dimension that already *has* an index is never missed: the caller unions
    this with ``existing_per_kb_indexes``, so the model-change case (rows now at
    a new dimension, an index still at the old one) is evaluated for dropping.

    Restricted to ``PER_KB_INDEX_ITEM_TABLE``, like the count that decides, and an
    earlier version of this was not -- which made the ``cap`` budget spendable by a
    population the decision then ignores. Measured: 4,000 (here 30,000)
    ``full_documents`` rows at 64 dimensions written first, then chunks at 128,
    threshold 1,000 -- the unrestricted survey returned ``[64]``,
    ``bounded_row_count`` at 64 returned 0, ``index_action`` returned None, and
    30,000 chunk rows well above the threshold were never dispatched at all. The
    boot sweep recovers it, but a restart is not a recovery for a running project.

    The cost is real and is the reason it was not restricted: the ``LIMIT`` now has
    to read past the other populations to find ``cap`` rows of this one, 0.77 ms to
    2.87 ms on that fixture. It is not a *new* cost, though --
    ``bounded_row_count`` already carries the same filter and took 4.13 ms in the
    same call -- so this adds no worst case the decision did not already have, and
    the btree named there fixes both at once. What it buys is that every query
    deciding eligibility agrees about which rows the index covers, which is what
    ``PER_KB_INDEX_ITEM_TABLE`` exists to make true.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    rows = conn.execute(
        text(
            "SELECT dims FROM (SELECT dims FROM "
            f'"{AI_SCHEMA}".embeddings '
            "WHERE knowledge_base_id = CAST(:kb AS uuid) AND item_table = :item_table "
            "LIMIT :cap) s "
            "GROUP BY dims ORDER BY count(*) DESC"
        ),
        {
            "kb": kb_id,
            "cap": max(1, int(cap)),
            "item_table": PER_KB_INDEX_ITEM_TABLE,
        },
    ).all()
    return [int(r[0]) for r in rows if MIN_DIMS <= int(r[0]) <= MAX_DIMS]


def index_action(conn, knowledge_base_id: Any) -> str | None:
    """``"build"``, ``"drop"`` or None -- is there anything to reconcile here?

    Cheap enough for the indexing path to call once per source, but not free, and
    the number matters at the shipped threshold: one catalog lookup plus two
    bounded reads per dimension in play, each stopping at ``build_at + 1``. At the
    50,000-row default that is a cap of 50,001 twice over -- the dimension survey and
    then the count -- and once more per further dimension.

    **Neither is an index-only read, and the cap bounds what each one returns rather
    than what it reads**, which an earlier version of this docstring had backwards:
    see ``bounded_row_count``, where it is measured as a ``Seq Scan`` with all three
    predicates as heap filters. So a knowledge base with a large *other* population
    reads past it to find ``cap`` rows of this one, and the honest figure is bounded
    by the table's slice rather than by 2 x ``cap``. It is still the cheap side of
    dispatching a build that reads the whole slice and writes a graph over it, and a
    project that lowers the threshold lowers this with it; the btree
    ``bounded_row_count`` names is what would make both index-only.

    Never raises for a knowledge base that has no embeddings at all.

    Nothing is asked for here that the reconcile would decline, because this runs
    once per source that finishes indexing: a build the width forbids
    (``MAX_HNSW_DIMS``), one the project has no room for
    (``MAX_PER_KB_INDEXES``), or one that has reached either build bound would
    otherwise be dispatched again by every source, for a task that can only report
    ``skipped``. ``ensure_per_kb_vector_index`` makes each of these decisions in the
    same order, for the same reason -- the two used to disagree about whether the
    row count or the failure bound came first, and the disagreement stranded a
    doomed index whose knowledge base had since shrunk: this function asked for the
    drop and the reconcile never reached the count that would have made it.
    """
    build_at, drop_below = thresholds()
    states = per_kb_index_states(conn, knowledge_base_id)
    cap = build_at + _COUNT_HEADROOM
    at_index_cap: bool | None = None

    def room_for_one_more() -> bool:
        nonlocal at_index_cap
        if at_index_cap is None:
            # One extra catalog count, and only for a knowledge base that would
            # otherwise have a build dispatched for it.
            at_index_cap = per_kb_index_count(conn) >= MAX_PER_KB_INDEXES
        return not at_index_cap

    for dims in sorted(set(candidate_dims(conn, knowledge_base_id, cap)) | set(states)):
        rows = bounded_row_count(conn, knowledge_base_id, dims, cap)
        valid, comment = states.get(dims, (None, None))
        if valid is False:
            # An INVALID index cannot be left as it is -- it answers no query and
            # Postgres maintains it on every write -- but *which* way it goes is
            # the question the row count answers, exactly as for a valid one. A
            # knowledge base that shrank below the drop threshold while its build
            # was failing wants that index gone, not built at the size it no
            # longer is; the reconcile drops it either way, and saying "build"
            # here was how a rebuild got asked for that the reconcile would then
            # decline.
            if rows <= drop_below or dims > MAX_HNSW_DIMS:
                return "drop"
            if build_is_given_up(comment):
                continue
            return "build"
        if valid is True:
            if rows <= drop_below:
                return "drop"
            if definition_has_drifted(_validated_kb_id(knowledge_base_id), dims, comment):
                # Built from a definition this version no longer emits, so it
                # answers searches at whatever recall its old predicate gives. The
                # reconcile replaces it -- but only where it would build one from
                # scratch, because the replacement is a drop and then a build, and
                # a drop it cannot follow with a build leaves this knowledge base
                # with nothing where it had a stale-but-usable index.
                if (
                    rows >= build_at
                    and dims <= MAX_HNSW_DIMS
                    and not definition_rebuild_is_given_up(comment)
                    and room_for_one_more()
                ):
                    return "build"
            continue
        if rows >= build_at and dims <= MAX_HNSW_DIMS and room_for_one_more():
            return "build"
    return None


# ---------------------------------------------------------------------------
# Connections and locks
# ---------------------------------------------------------------------------


def estimated_index_mb(rows: int, dims: int) -> int:
    """Roughly how much disk one of these indexes will take, for a log line.

    There is no free-space precheck anywhere here because Postgres exposes no
    free-space figure -- no catalog view or function reports what the
    filesystem has left, and the index goes into the database's own tablespace.
    So the size the build is reaching for is logged instead, which is what an
    operator needs when a build fails on a full disk.

    ``rows`` is normally a *bounded* count that stops just past the build
    threshold, which makes this a floor and not an estimate: a knowledge base
    ten times the threshold builds an index ten times this size. The caller
    says so in the log line rather than printing a number that reads like the
    whole answer.
    """
    return max(1, rows * dims * _INDEX_BYTES_PER_DIMENSION // (1024 * 1024))


# The execution option that keeps a connection outside a transaction block, so
# ``CREATE INDEX CONCURRENTLY`` can run on it. Named once because it has to be
# applied in two places: when the connection is opened, and again whenever
# ``_discard_connection`` reconnects the handle.
_AUTOCOMMIT = {"isolation_level": "AUTOCOMMIT"}


def _autocommit_connection(engine):
    """A connection outside any transaction: CONCURRENTLY refuses one."""
    return engine.connect().execution_options(**_AUTOCOMMIT)


def _engine(engine=None):
    if engine is not None:
        return engine
    from ..db import db

    return db.engine


def _try_lock(conn, relation: str) -> bool:
    return bool(conn.execute(text(partition_build_lock_sql()), {"relation": relation}).scalar())


def _discard_connection(conn, holding: str | None = None) -> None:
    """Throw this connection's backend away, and leave the handle usable.

    ``invalidate()`` on its own is only half of it. Every caller here shares one
    connection across the whole reconcile loop, and measured against a real server
    on exactly that AUTOCOMMIT connection, the statement after ``invalidate()``
    raises ``PendingRollbackError`` -- "Can't reconnect until invalid transaction
    is rolled back" -- and goes on raising until the rollback. That error is not
    classified as a transient database error, so a connection lost at one
    dimension would fail the run without the retry that exists for exactly that,
    and would take the other dimensions with it: a knowledge base that has just
    changed embedding model has an index to drop at the old dimension and one to
    build at the new one.

    The rollback is what lets the handle reconnect; it rolls nothing back that the
    caller wanted, because the backend it belonged to is already gone.

    **The AUTOCOMMIT option has to be re-applied.** It is an *execution* option,
    held against the DBAPI connection the handle had, and reconnecting gets a new
    one that SQLAlchemy does not carry it onto -- so without this line everything
    after a discard runs in an implicit transaction, and the next
    ``CREATE INDEX CONCURRENTLY`` fails with ``25001 CREATE INDEX CONCURRENTLY
    cannot run inside a transaction block``. Measured on exactly the scenario
    above -- drop at 4 dimensions, connection lost at ``pg_advisory_unlock``,
    build at 8 -- the drop succeeded, the loop reached the next dimension, and the
    build then raised ``25001``, which ``is_transient_db_error`` does *not*
    recognise: the one retry this function exists to preserve was lost, and every
    later ``RESET`` in the same ``finally`` failed with "current transaction is
    aborted", so the warnings described the wrong cause too.
    """
    try:
        conn.invalidate()
    except Exception:
        if holding is None:
            logger.debug("Could not invalidate the connection", exc_info=True)
        else:
            # The one case where this is not housekeeping: the backend being thrown
            # away holds a session advisory lock, and it goes back to the pool still
            # holding it. For ``table_lock_relation()`` that gates every index build
            # and drop in the project until the backend ends.
            logger.error(
                "Could not discard the connection holding the advisory lock on %s; the "
                "lock stays held until that backend ends, and every build or drop it "
                "guards waits for it (pg_terminate_backend on its pid releases it)",
                holding,
                exc_info=True,
            )
    try:
        conn.rollback()
    except Exception:
        logger.debug("Could not give the invalidated connection back", exc_info=True)
    try:
        conn.execution_options(**_AUTOCOMMIT)
    except Exception:
        # Nothing here can run without it -- every statement the callers issue
        # after this point is either DDL that refuses a transaction block or a
        # read that has to see the DDL's effect -- so a handle that cannot be put
        # back into AUTOCOMMIT is worth the warning even though the caller's own
        # error may be on its way up.
        logger.warning(
            "Could not put the reconnected connection back into AUTOCOMMIT; "
            "CREATE INDEX CONCURRENTLY cannot run on it",
            exc_info=True,
        )


def _release_lock(conn, relation: str) -> None:
    """Give the session-scoped lock back, or throw the session away.

    The lock outlives a transaction on purpose (a concurrent build is not one),
    so a pooled connection handed back still holding it would make every later
    build of this index skip. Ending the backend releases it for certain.

    ``pg_advisory_unlock`` answering False is read, not ignored: it means this
    session did not hold the lock, and on these connections the only way that
    happens is a discard in between (``_discard_connection``) -- the lock went with
    the old backend, so whatever it guarded after that ran on the new one without
    it. The table gate re-takes itself when that happens (``_TableGate.hold``);
    this is what says so when nothing did.
    """
    try:
        released = conn.execute(text(partition_build_unlock_sql()), {"relation": relation}).scalar()
    except Exception:
        logger.warning(
            "Could not release the advisory lock on %s; discarding the connection so the "
            "lock cannot outlive it",
            relation,
            exc_info=True,
        )
        _discard_connection(conn, holding=relation)
        return
    if released is False:
        logger.warning(
            "The advisory lock on %s was not held by this session when it was given back: "
            "its connection was replaced after the lock was taken, and the lock ended with "
            "the old backend",
            relation,
        )


def _reset_session_setting(conn, name: str) -> None:
    """RESET one session setting without hiding the error a build already raised.

    A server that restarts mid-build ends the session, and SQLAlchemy then
    refuses any further statement on the connection. Raised from a ``finally``,
    that would replace a lost connection -- which the task retries -- with an
    error it does not.
    """
    try:
        conn.execute(text(f"RESET {name}"))
    except Exception as exc:
        logger.warning(
            "Could not reset %s after a build (%s); discarding the connection",
            name,
            first_error_line(exc),
        )
        _discard_connection(conn)


def _release_settings_session() -> None:
    """End the transaction a settings read left open on ``db.session``.

    ``get_setting`` reads ``ai.project_settings`` through ``db.session`` and
    nothing on that path commits or rolls back, so the session holds a
    connection -- and an open transaction -- from the first threshold read until
    the task ends. A build would then occupy two connections rather than one,
    the second idle in a transaction for as long as the build runs, which is
    minutes on a large knowledge base.

    That is not only a wasted connection. ``CREATE INDEX CONCURRENTLY`` waits
    for every transaction whose snapshot predates its own before it can finish,
    so the build would be waiting on its own task's session -- the stall
    ``_create_index``'s docstring warns about, caused by the build itself.

    Only the out-of-band task calls the functions that call this, so there is
    never caller work to lose. A session that was never opened, or no
    application context at all (a test passing its own engine), is nothing to
    give back.
    """
    try:
        from ..db import db

        db.session.rollback()
    except Exception:
        logger.debug("No settings session to give back", exc_info=True)


def _qualified_index(kb_id: str, dims: int) -> str:
    """``"ai".hnsw_kb_<hex>_<dims>`` -- both parts derived, neither from a caller."""
    return f'"{AI_SCHEMA}".{per_kb_index_name(kb_id, dims)}'


def per_kb_index_fingerprint(knowledge_base_id: Any, dims: Any) -> str:
    """A short digest of the definition an index of this name would be built from.

    The index *name* carries the knowledge base and the dimension and nothing
    else, while ``per_kb_index_ddl`` also decides the indexed expression, the
    operator class, the ``::vector(N)`` cast and the predicate -- which now names
    ``item_table`` as well. So two indexes of the same name can be built from
    different definitions, and the catalog readers here could not tell them apart:
    an index built before the ``item_table`` clause existed has the old two-clause
    predicate, today's query *implies* it, so the planner still matches it, and
    ``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` no-ops against the name it holds.
    Measured on a 1,000-chunk / 9,000-``full_documents`` fixture at 1536
    dimensions, that index answers every chunk search at recall 0.562 where 0.989
    was available, for ever, with no log line.

    This is the fact the name does not carry, written into the index's own
    ``pg_class`` comment by the build and read back by the reconcile and the boot
    sweep -- which already reads that column in bulk. The ``pg_bm25_index``
    sibling has the same problem and the same shape of answer
    (``indexdef_matches_tokenizer``), for the same reason: its index's identity
    also depends on something its name does not carry.

    **Deliberately not a comparison against ``pg_get_indexdef``.** What
    PostgreSQL prints back differs from what is emitted here in at least six ways
    -- ``((embedding)::vector(N))`` against ``(embedding::vector(N))``,
    ``'...'::uuid`` against ``'...'``, ``(item_table)::text = 'chunks'::text``
    against ``item_table = 'chunks'``, and the parenthesisation of the predicate
    -- so the two are never equal and an equality test would call *every* index
    stale. That is worse than the hole it closes: the failure record dies with the
    index a drift drop takes away, so ``MAX_CONSECUTIVE_BUILD_FAILURES`` would
    never engage and every reconcile of every knowledge base would drop and
    rebuild. A digest of what this module emits is exact and immune to
    normalisation.

    An index built by a version of this code that did not write a fingerprint
    carries none, so it reads as drifted on the first look -- which is correct,
    because that is exactly the index this exists to find.

    Twelve hex characters: the whole comment is an operator-facing sentence, and
    48 bits is far more than enough to separate the handful of definitions one
    deploy of this module can emit. It is not a security boundary -- nothing acts
    on an attacker-chosen value here -- so a short digest of a string this module
    generated is the right size.
    """
    ddl = per_kb_index_ddl(knowledge_base_id, dims)
    return hashlib.sha256(ddl.encode("utf-8")).hexdigest()[:12]


def per_kb_index_comment(
    failures: int = 0,
    interrupted: int = 0,
    fingerprint: str | None = None,
    rebuilds: int = 0,
) -> str | None:
    """The ``pg_class`` comment recording what is known about one of these indexes.

    ``None`` when there is nothing to record, which is what ``COMMENT ON ... IS
    NULL`` writes. The prose about an INVALID index is appended only when one of the
    two *build* counts is being recorded, because that is the only time the index is
    INVALID -- and it is appended after **all three** counts, including the rebuild
    count, which describes a valid index still answering searches. The two are
    producible together, so the prose may not be allowed to swallow one of them.

    Composed here, and read back by the four ``_in`` functions below, so the
    wording lives in one place and a reworded sentence moves both directions at
    once. Everything composed here is prefixed with ``_COMMENT_MARKER``, which is
    what those readers require.
    """
    parts: list[str] = []
    if failures:
        parts.append(_BUILD_FAILURES_SENTENCE.format(n=int(failures)))
    if interrupted:
        parts.append(_INTERRUPTED_BUILDS_SENTENCE.format(n=int(interrupted)))
    if rebuilds:
        parts.append(_REBUILDS_SENTENCE.format(n=int(rebuilds)))
    if failures or interrupted:
        parts.append(_INVALID_INDEX_PROSE)
    if fingerprint:
        parts.append(_DEFINITION_SENTENCE.format(fp=fingerprint))
    return " ".join([_COMMENT_MARKER, *parts]) if parts else None


def _module_written(comment: str | None) -> str:
    """The comment this module may read facts out of, or an empty string.

    Every reader below goes through this, so the four patterns are only ever run
    against a comment this module composed. See ``_COMMENT_MARKER``.
    """
    body = comment or ""
    return body if body.startswith(_COMMENT_MARKER) else ""


def build_failures_in(comment: str | None) -> int:
    """The consecutive permanent-looking failure count a comment records, or zero.

    Split out from ``recorded_build_history`` because the start-up sweep reads
    these comments in bulk, as a column of the catalog SELECT it already runs,
    rather than one round trip per index on the boot path.
    """
    match = _BUILD_FAILURES_PATTERN.search(_module_written(comment))
    return int(match.group(1)) if match else 0


def interrupted_builds_in(comment: str | None) -> int:
    """The consecutive interrupted-build count a comment records, or zero."""
    match = _INTERRUPTED_BUILDS_PATTERN.search(_module_written(comment))
    return int(match.group(1)) if match else 0


def definition_rebuilds_in(comment: str | None) -> int:
    """How many consecutive definition rebuilds of this index have not settled, or zero.

    A third count, on a *third* bound (``MAX_CONSECUTIVE_DEFINITION_REBUILDS``), and
    deliberately not part of either build bound: it is written before a drift drop,
    carried across that drop and written back by whatever ends the attempt, and
    cleared only by a later pass that finds the index already built from today's
    definition (``_settle_a_definition_rebuild``). So a number here means rebuilds
    have been started that nothing has since confirmed -- which is a wider thing than
    rebuilds that failed, and deliberately: the shape that runs for ever is the one
    where every rebuild succeeds. See ``MAX_CONSECUTIVE_DEFINITION_REBUILDS`` for
    what it bounds and what it cannot.
    """
    match = _REBUILDS_PATTERN.search(_module_written(comment))
    return int(match.group(1)) if match else 0


def definition_fingerprint_in(comment: str | None) -> str | None:
    """The definition fingerprint a comment records, or None if it records none.

    None for an index built before fingerprints were written, for one built or
    commented by hand, and for one whose comment says something else entirely --
    all of which are treated as a definition this module cannot vouch for.
    """
    match = _DEFINITION_PATTERN.search(_module_written(comment))
    return match.group(1) if match else None


def definition_has_drifted(kb_id: str, dims: int, comment: str | None) -> bool:
    """Was this index built from a definition ``per_kb_index_ddl`` no longer emits?"""
    return definition_fingerprint_in(comment) != per_kb_index_fingerprint(kb_id, dims)


def build_is_given_up(comment: str | None) -> bool:
    """Has this index reached either *build* bound, so no further build is attempted?

    Deliberately not also the drift bound. These two say "do not build this index",
    and the index they say it about is INVALID; ``definition_rebuild_is_given_up``
    says "do not *replace* this index", about one that is valid and answering
    searches. Folding the third in would make a spent rebuild budget suppress the
    build that a knowledge base with no index at all still needs.
    """
    return (
        build_failures_in(comment) >= MAX_CONSECUTIVE_BUILD_FAILURES
        or interrupted_builds_in(comment) >= MAX_CONSECUTIVE_INTERRUPTED_BUILDS
    )


def definition_rebuild_is_given_up(comment: str | None) -> bool:
    """Have this index's definition rebuilds stopped settling, so no more are started?

    The drift path's bound, and nothing to do with the two above: reaching it leaves
    a *valid* index in place, still answering searches at whatever recall its old
    definition gives. See ``MAX_CONSECUTIVE_DEFINITION_REBUILDS``.
    """
    return definition_rebuilds_in(comment) >= MAX_CONSECUTIVE_DEFINITION_REBUILDS


def index_comment(conn, kb_id: str, dims: int) -> str | None:
    """This index's ``pg_class`` comment, or None if it has none or does not exist.

    A read that *fails* is not swallowed, unlike the writes below. It is an
    ordinary catalog read on the caller's connection, so a failure here means
    every other read around it would fail too, and swallowing it would hand the
    caller back an aborted transaction while reporting a fact. Both callers
    already handle that: the reconcile lets it fail the run, and the dispatch
    check turns it into "this knowledge base keeps whatever it has".
    """
    return conn.execute(
        text("SELECT obj_description(to_regclass(:index)::oid, 'pg_class')"),
        {"index": _qualified_index(kb_id, dims)},
    ).scalar()


def recorded_build_history(conn, kb_id: str, dims: int) -> tuple[int, int]:
    """``(permanent failures, interrupted builds)`` for this index, from the catalog.

    Zeros for an index that has never failed, that does not exist, or whose
    comment says something else: a comment is a place anyone may write, and one
    this module did not write is no evidence that a build is doomed.
    """
    comment = index_comment(conn, kb_id, dims)
    return build_failures_in(comment), interrupted_builds_in(comment)


def recorded_build_failures(conn, kb_id: str, dims: int) -> int:
    """How many consecutive builds of this index have failed permanently."""
    return recorded_build_history(conn, kb_id, dims)[0]


def _write_index_comment(
    conn,
    kb_id: str,
    dims: int,
    failures: int,
    interrupted: int,
    fingerprint: str | None,
    rebuilds: int = 0,
) -> None:
    """Record what is known about this index on the index itself. Best effort.

    Deliberately best effort: the failure paths run while a build's exception is
    on its way up, and losing the record must not replace the error that says why
    the build failed. A failure so early that no catalog entry exists yet leaves
    nothing to comment on, and the next reconcile is then exactly as it is today.

    **Best effort is why the drift path may not trust it.** The fingerprint written
    here is the only thing that stops the next reconcile calling the same index
    stale again, and a drift drop is (correctly) charged to neither build bound, so
    a write that quietly does nothing turns the drift detector into an unbounded
    drop-and-rebuild loop -- reproduced at 6 reconciles and 6 distinct index oids
    with this function no-op'd, and 4 consecutive rebuilds with ``COMMENT``
    refused server-side, each iteration a real ``DROP`` plus
    ``CREATE INDEX CONCURRENTLY`` with both timeouts lifted. So that caller writes
    through ``_count_a_definition_rebuild``, which reads the write back before it
    drops anything.

    The comment is generated here in full -- the only values from outside are two
    integers this module counted and a hex digest it computed -- so there is
    nothing in it to quote.
    """
    body = per_kb_index_comment(failures, interrupted, fingerprint, rebuilds)
    literal = "NULL" if body is None else f"'{body}'"
    try:
        conn.execute(text(f"COMMENT ON INDEX {_qualified_index(kb_id, dims)} IS {literal}"))
    except Exception as exc:
        logger.warning(
            "Could not record the build history of %s (%d failed, %d interrupted, %d "
            "unsettled rebuilds, definition %s) (%s); a later reconcile will count from what "
            "it can read, and a definition it cannot read is treated as one this module did "
            "not build",
            _qualified_index(kb_id, dims),
            failures,
            interrupted,
            rebuilds,
            fingerprint or "unknown",
            first_error_line(exc),
        )


def _record_build_failure(
    conn, kb_id: str, dims: int, failures: int, interrupted: int | None = None
) -> None:
    """Record a failed attempt, keeping whatever definition the index really carries.

    The entry point for every caller that is *not* recording an attempt it just
    made against today's DDL. It reads the definition back off the index rather
    than assuming today's, so recording a failure on an index built from an older
    definition does not quietly relabel it as current. ``interrupted`` is read back
    too when the caller has no number of its own.

    That is a behavioural distinction, not a tidiness one, because the mis-stamp is
    not recoverable. ``REINDEX`` -- what the PostgreSQL manual recommends for an
    INVALID index, and so what an operator reaches for -- rebuilds from the
    *stored* predicate and preserves the ``pg_class`` comment, measured both plain
    and ``CONCURRENTLY``. An older-definition index stamped with today's
    fingerprint and then reindexed is therefore valid, stale, and certified
    current: ``definition_has_drifted`` says no, ``index_action`` returns None, the
    start-up sweep returns nothing, and no log line is written ever again.
    Measured on a stale two-clause index, mean recall@20 0.559 against 0.993 for
    the same index built from today's definition.
    """
    comment = index_comment(conn, kb_id, dims)
    _write_index_comment(
        conn,
        kb_id,
        dims,
        failures,
        interrupted_builds_in(comment) if interrupted is None else interrupted,
        definition_fingerprint_in(comment),
        definition_rebuilds_in(comment),
    )


def _record_the_definition_built(conn, kb_id: str, dims: int, rebuilds: int = 0) -> None:
    """A build succeeded: record what it was built from, and forget the failures.

    Both in one write, because they are one comment. The two *build* counts are
    consecutive failures, so a success clears them; the fingerprint has to be
    written on every success rather than only after a failure, because it is the
    only record that this index matches the definition this module now emits.

    ``rebuilds`` is the one count a success does **not** clear, and it is carried in
    from the caller for the reason ``_counted_attempt`` carries the other two: the
    drop that precedes a rebuild takes the record away with the index it is written
    on, so the number has to survive in memory across it. Writing it back here is
    what makes the count durable across a rebuild that lands -- see
    ``MAX_CONSECUTIVE_DEFINITION_REBUILDS`` for why a landed rebuild is not by itself
    evidence that anything settled, and ``_settle_a_definition_rebuild`` for what
    clears it.
    """
    _write_index_comment(conn, kb_id, dims, 0, 0, per_kb_index_fingerprint(kb_id, dims), rebuilds)


def _settle_a_definition_rebuild(conn, kb_id: str, dims: int, comment: str | None) -> None:
    """A reconcile found this index built from today's definition: clear the count.

    The only thing that clears ``MAX_CONSECUTIVE_DEFINITION_REBUILDS``, and
    deliberately not the rebuild itself. A rebuild that lands proves that a
    ``CREATE INDEX CONCURRENTLY`` completed and that a ``COMMENT`` stuck; it does not
    prove that the definition this module emits has stopped moving, and in the one
    shape where it has not -- two deploys emitting two definitions, each reading the
    other's index as drifted -- every rebuild lands and clearing on that alone leaves
    the count at zero for ever. What this branch has that the rebuild does not is a
    *later* pass, by a process that recomputed the fingerprint from scratch, finding
    the index already current.

    Clearing at all is not optional: the count is spent one per definition change,
    and its only remedy is an operator clearing the comment by hand, so without this
    a handful of ordinary DDL changes over a project's life would turn the feature
    off. This is the event that gives the count back, and a boot is what produces it
    (``kbs_needing_a_per_kb_index`` dispatches an index carrying an unsettled count
    so that one of these passes happens at all -- nothing else reconciles a knowledge
    base whose index is already correct).

    A count that has *reached* the bound is left alone, so giving up stays as
    permanent as the two build bounds it sits beside: in a rebuild war a pass by
    whichever version's definition is currently on the index sees no drift, and would
    otherwise clear the number the war put there and start it again. Recovery from there is the ERROR's own instruction -- clear the comment
    or drop the index -- which is a decision about a deploy and not one this module
    can make from a single catalog read.
    """
    rebuilds = definition_rebuilds_in(comment)
    if not 0 < rebuilds < MAX_CONSECUTIVE_DEFINITION_REBUILDS:
        return
    logger.info(
        "Partial HNSW index %s.%s is built from the definition this version emits, so the "
        "%d rebuild(s) on record for it have settled and the count is cleared",
        AI_SCHEMA,
        per_kb_index_name(kb_id, dims),
        rebuilds,
    )
    _write_index_comment(
        conn,
        kb_id,
        dims,
        build_failures_in(comment),
        interrupted_builds_in(comment),
        definition_fingerprint_in(comment),
        0,
    )


def _counted_attempt(
    prior_failures: int, prior_interrupted: int, exc: BaseException
) -> tuple[int, int]:
    """``(failures, interrupted)`` after charging one failed attempt to its bound.

    A failure ``is_transient_db_error`` recognises goes against the larger
    ``MAX_CONSECUTIVE_INTERRUPTED_BUILDS``, because the task that runs these
    builds retries such a failure several times and one contention episode must
    not spend a budget whose only remedy is a manual ``DROP INDEX``. Anything else
    goes against ``MAX_CONSECUTIVE_BUILD_FAILURES``.

    Both counts are carried in from the caller rather than re-read, because the
    repair drop that precedes a rebuild takes the record away with the index it is
    written on -- so the numbers have to survive in memory across it. Writing
    nothing for a transient failure, which is what this did before, handed the
    whole budget back on every reconcile and made the bound unreachable.

    Split from the two writes below because the *classification* is common to both
    and the *definition each one records* is not.

    A reconcile refused by ``_TableGate`` is never an attempt and never reaches
    this: the gate is taken before the first statement a dimension would issue
    -- before the definition-rebuild count too -- so ``table_busy`` writes no
    comment and moves neither bound. Charging it would have spent one of these
    counts per look at a table another build owns, which is up to
    ``PER_KB_TABLE_MAX_WAITS`` looks in the task.
    """
    if is_transient_db_error(exc):
        return prior_failures, prior_interrupted + 1
    return prior_failures + 1, prior_interrupted


def _count_a_failed_build(
    conn,
    kb_id: str,
    dims: int,
    prior_failures: int,
    prior_interrupted: int,
    exc: BaseException,
    prior_rebuilds: int = 0,
) -> None:
    """Count a ``CREATE INDEX CONCURRENTLY`` that failed, recording today's definition.

    Today's is right **here and only here**: the INVALID index this failure just
    left behind really was created from today's DDL, so writing today's fingerprint
    on it states a fact about the object on disk.

    A drop that fails is the same attempt and the same two bounds, but not the same
    fact -- see ``_count_a_failed_repair_drop``, where the index on disk is the one
    that was already there.

    ``prior_rebuilds`` is carried through for the same reason the sibling path reads
    it back: this write is a full comment, so a count left out of it is a count
    erased. The failing build may *be* a drift rebuild -- the drop happened, the
    number that bounds it was on the index the drop took away, and it is in the
    caller's hands -- and dropping it here would hand the whole rebuild budget back
    on every reconcile whose rebuild fails, which is the shape
    ``MAX_CONSECUTIVE_INTERRUPTED_BUILDS`` was added for one bound down. It is the
    same decision ``_record_build_failure`` makes by reading the count off the index,
    stated here because this path has no index to read it from.
    """
    failures, interrupted = _counted_attempt(prior_failures, prior_interrupted, exc)
    _write_index_comment(
        conn,
        kb_id,
        dims,
        failures,
        interrupted,
        per_kb_index_fingerprint(kb_id, dims),
        prior_rebuilds,
    )


def _count_a_failed_repair_drop(
    conn,
    kb_id: str,
    dims: int,
    prior_failures: int,
    prior_interrupted: int,
    exc: BaseException,
) -> None:
    """Count a repair drop that raised, leaving the recorded definition as it was.

    Charged to the same two bounds as the build it precedes, by the same rule and
    for the same reason: it is the second way one reconcile of an INVALID index can
    end without an index, and counting only the first left the bound reachable from
    one side and not the other.

    What it must **not** do is write today's fingerprint, which is the one thing
    ``_count_a_failed_build`` may. The drop failed, so the index on disk is the
    pre-existing one, built from whatever definition built it -- and an index
    carrying an *older* definition reaches exactly this path: a drift drop
    cancelled part-way leaves the old-definition index INVALID
    (``indisvalid = false, indisready = true``, the state ``_drop_index``
    documents), the next reconcile repairs it, and that drop fails too. Stamping
    today's fingerprint there certifies a stale index as current, permanently once
    an operator reindexes it. ``_record_build_failure`` reads the definition back
    instead, and takes the numbers this caller carried across the drop.
    """
    failures, interrupted = _counted_attempt(prior_failures, prior_interrupted, exc)
    _record_build_failure(conn, kb_id, dims, failures, interrupted)


def _build_in_progress(conn, kb_id: str, dims: int) -> bool:
    """Is another backend building or reindexing *this* index right now?

    Scoped to the one index, by ``index_relid``, not to ``ai.embeddings``. The
    BM25 path's equivalent scopes by ``relid`` because each of its indexes is on
    a relation of its own (that knowledge base's partition), so a relation is a
    knowledge base there; here every index is on the one shared table, so the
    same shape would report *any* concurrent build on it -- another knowledge
    base's, or ``ensure_embedding_index`` creating a shared per-dimension one.
    That question is a different one, about the table, and it has its own
    answer: ``table_ddl_holders``, asked by ``_TableGate`` before any DDL. This
    one decides only whether *this* INVALID index belongs to a build still
    running -- in which case it must not be dropped at all -- and it is asked
    before the gate, because leaving the index alone issues nothing.

    ``pg_stat_progress_create_index.index_relid`` is populated for
    ``CREATE INDEX CONCURRENTLY`` from the moment the catalog entry exists
    (verified against PostgreSQL 15 -- the documentation's "during CREATE INDEX
    it's 0" describes the non-concurrent case). An INVALID index always has a
    catalog entry, which is the case this guard exists for.
    """
    row = conn.execute(
        text(
            "SELECT 1 FROM pg_stat_progress_create_index "
            "WHERE index_relid = to_regclass(:index)::oid AND pid <> pg_backend_pid()"
        ),
        {"index": f'"{AI_SCHEMA}".{per_kb_index_name(kb_id, dims)}'},
    ).first()
    return row is not None


# ---------------------------------------------------------------------------
# One build or drop at a time on the table
# ---------------------------------------------------------------------------

# The lock modes a ``CREATE``/``DROP INDEX CONCURRENTLY`` on the table would queue
# behind: its own ``ShareUpdateExclusiveLock`` conflicts with itself and with every
# stronger mode. A plain ``CREATE INDEX`` (``ShareLock``) is in the list on purpose
# -- ``base_vector_store.ensure_embedding_index`` issues one, inside an indexing
# transaction, the first time a dimension appears.
#
# A manual ``VACUUM`` or ``ANALYZE`` of the table takes ``ShareUpdateExclusiveLock``
# too, so it now defers even a single knowledge base's build until it finishes,
# where before the build simply queued behind it (a plain wait, no cycle). That is
# the price of not being able to tell, from the lock alone, a statement that will
# never wait on the build from one that will; autovacuum is the exception the
# holder query can make, because it yields to lock waiters on its own.
_TABLE_DDL_LOCK_MODES = (
    "ShareUpdateExclusiveLock",
    "ShareLock",
    "ShareRowExclusiveLock",
    "ExclusiveLock",
    "AccessExclusiveLock",
)

# How long the backend holding this module's table gate may sit idle between two
# statements and still count as a caller in the middle of its work. The gate is
# taken immediately before a dimension's first DDL and given back after its last
# write, so the windows in which it is held with no table lock to show for it are
# the ones between statements of one reconcile -- the lock try and the build's
# grant, a repair DROP and the CREATE after it, the build and the comment and
# RESETs that follow -- all of them round trips from a worker that is running.
# A holder idle longer than this is not in the middle of anything and does not
# buy the long uncounted wait (``_holder_kind``).
_GATE_HOLDER_IDLE_GRACE_S = 60

# The three ages every holder is reported with, in seconds: its current (or last)
# statement, its transaction, and its last state change -- which, for a session
# idle in a transaction, is how long it has been idle. ``query_start`` alone said
# 0 for a session that had held the table idle for hours.
_HOLDER_AGES_SQL = (
    "floor(extract(epoch FROM clock_timestamp() - a.query_start))::bigint, "
    "floor(extract(epoch FROM clock_timestamp() - a.xact_start))::bigint, "
    "floor(extract(epoch FROM clock_timestamp() - a.state_change))::bigint, "
    "left(a.query, 200) "
)

_THIS_DATABASE_SQL = "(SELECT oid FROM pg_database WHERE datname = current_database())"

# Read from ``pg_locks`` rather than by matching ``CONCURRENTLY`` in query text:
# a concurrent build or drop holds its table lock as a *session* lock across all
# of its internal transactions, so it is here, granted, from its first phase to
# its last -- including the final wait, when ``pg_stat_activity.query`` of a
# pooled caller may already say something else. ``pg_locks`` is readable by every
# role; the ``pg_stat_activity`` half of the join is not: for another role's
# backend a role without ``pg_read_all_stats`` (or ``pg_monitor``) sees NULL
# ``backend_type`` and ``state`` and a query of ``<insufficient privilege>``.
# Such a holder is kept and classified *unknown* rather than dropped, because it
# may be a build (see ``_holder_kind``). ``granted`` because a backend still
# queued is not a build that is running; autovacuum is left out -- where it can be
# seen -- because it cancels itself for a lock waiter (anti-wraparound excepted,
# which is a plain wait with no cycle), so queueing behind it costs about a second.
_TABLE_HOLDERS_SQL = (
    "SELECT a.pid, a.backend_type, 'relation', l.mode, a.state, "
    + _HOLDER_AGES_SQL
    + "FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid "
    "WHERE l.locktype = 'relation' "
    f"AND l.database = {_THIS_DATABASE_SQL} "
    "AND l.relation = to_regclass(:table) "
    "AND l.granted AND l.pid <> pg_backend_pid() "
    "AND l.mode IN (" + ", ".join(f"'{mode}'" for mode in _TABLE_DDL_LOCK_MODES) + ") "
    "AND a.backend_type IS DISTINCT FROM 'autovacuum worker' "
    "ORDER BY a.query_start NULLS LAST, a.pid"
)

# The backend holding this module's own table gate. ``pg_try_advisory_lock(bigint)``
# files its key as ``classid`` (high 32 bits) and ``objid`` (low 32 bits) with
# ``objsubid = 1``, so the key is put back together and compared with the same
# ``hashtextextended`` the lock was taken with. Asked only when the gate is
# refused by its lock: that is when the holder may be between two statements and
# show no table lock at all.
_GATE_HOLDER_SQL = (
    "SELECT a.pid, a.backend_type, 'advisory', l.mode, a.state, "
    + _HOLDER_AGES_SQL
    + "FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid "
    "WHERE l.locktype = 'advisory' "
    f"AND l.database = {_THIS_DATABASE_SQL} "
    "AND l.granted AND l.objsubid = 1 AND l.pid <> pg_backend_pid() "
    "AND ((l.classid::bigint << 32) | l.objid::bigint) = hashtextextended(:subject, 0)"
)

# What a backend this role may not inspect shows as its query.
_HIDDEN_QUERY = "<insufficient privilege>"

# Warned about once per process: the evidence cannot tell a build from autovacuum
# without the privilege, and every refusal after the first would say the same thing.
_warned_invisible_holders = False


def _age(value) -> int | None:
    return None if value is None else int(value)


def _holder_kind(lock: str, state: str | None, idle_s: int | None) -> str:
    """``running``, ``stalled`` or ``unknown`` -- and only ``running`` is a live build.

    * A holder of the table's own lock is *running* while its state is ``active``.
      A concurrent build is ``active`` from its first phase to its last, its waits
      included. ``idle in transaction`` -- a ``LOCK TABLE``, a plain ``CREATE INDEX``
      or an ``ANALYZE`` someone left open -- is *stalled*: it refuses the gate,
      because queueing behind it would hold a snapshot too, but nothing about it
      will finish on its own, so it gets the counted retry and a WARNING instead of
      48 hours of uncounted waits.
    * The holder of this module's gate is *running* while ``active`` or idle for at
      most ``_GATE_HOLDER_IDLE_GRACE_S`` -- a worker between two statements of its
      reconcile.
    * A holder whose state this role cannot see is *unknown*, and gets the counted
      path: it may be a build, and it may be autovacuum, which the query can only
      leave out when it can see ``backend_type``.
    """
    if state is None:
        return "unknown"
    if state == "active":
        return "running"
    if lock == "advisory" and state == "idle" and idle_s is not None:
        return "running" if idle_s <= _GATE_HOLDER_IDLE_GRACE_S else "stalled"
    return "stalled"


def _holder_evidence(row) -> dict:
    """One holder row as the outcome and the logs report it."""
    pid, backend_type, lock, mode, state, running_s, xact_s, idle_s, query = row
    idle = _age(idle_s)
    return {
        "pid": int(pid),
        "backend_type": backend_type,
        "lock": lock,
        "mode": mode,
        "state": state,
        "running_s": _age(running_s),
        "xact_s": _age(xact_s),
        "idle_s": idle,
        "query": None if query in (None, "", _HIDDEN_QUERY) else query,
        "kind": _holder_kind(lock, state, idle),
    }


def table_ddl_holders(conn, include_gate_holder: bool = False) -> list[dict]:
    """The other backends holding ``ai.embeddings`` against an index build or drop.

    Per *table*, where ``_build_in_progress`` is per index, and the difference is
    what issue #95 turned on: during the fight it described below, the repair path
    logged that no build was running on ``ai.embeddings`` -- true of the one index it
    had asked about -- while another knowledge base's ``CREATE INDEX CONCURRENTLY``
    had been running on that table for hours. This is the question the gate needs
    answered, and it is the evidence the task spends its uncounted wait on
    (``PerKbVectorIndexTableBusy.build_alive``): each holder carries a ``kind``
    (``_holder_kind``), and only a ``running`` one is a live build.

    ``include_gate_holder`` adds the backend holding this module's own gate
    (``_GATE_HOLDER_SQL``), for the refusal where the table lock is not what
    refused it.

    Plain reads of two statistics views: they take no lock that can wait, so they
    are safe to ask on the gate's AUTOCOMMIT connection with the table busy.

    **Needs ``pg_read_all_stats`` (or ``pg_monitor``) to be exact** when the service
    connects as a role other than the one other backends run as: without it every
    other role's holder is ``unknown``, autovacuum included, so each vacuum of the
    table costs a counted retry where it would have cost nothing. Warned once per
    process.
    """
    global _warned_invisible_holders
    rows = list(
        conn.execute(text(_TABLE_HOLDERS_SQL), {"table": f'"{AI_SCHEMA}".embeddings'}).all()
    )
    if include_gate_holder:
        rows += conn.execute(text(_GATE_HOLDER_SQL), {"subject": table_lock_relation()}).all()
    holders: dict[int, dict] = {}
    for row in rows:
        evidence = _holder_evidence(row)
        holders.setdefault(evidence["pid"], evidence)
    if not _warned_invisible_holders and any(h["kind"] == "unknown" for h in holders.values()):
        _warned_invisible_holders = True
        logger.warning(
            "This database role cannot see what the backends holding %s.embeddings are "
            "doing (pg_stat_activity shows another role's state as NULL without "
            "pg_read_all_stats or pg_monitor), so it cannot tell an index build from "
            "autovacuum: every such holder is treated as unknown and costs a counted "
            "retry. Grant pg_read_all_stats to the service's role to make the per-table "
            "index gate exact",
            AI_SCHEMA,
        )
    return list(holders.values())


def _backend_pid(conn):
    return conn.execute(text("SELECT pg_backend_pid()")).scalar()


class _TableGate:
    """One ``CREATE``/``DROP INDEX CONCURRENTLY`` on ``ai.embeddings`` at a time, per project.

    **Why any DDL queued on the table while a build runs kills that build, at its
    end.** Every concurrent build and drop on the table takes its
    ``ShareUpdateExclusiveLock``, which conflicts with itself. The per-index lock
    (``index_lock_relation``) keeps two callers off *one* index and nothing more,
    so a second knowledge base's build or drop used to go straight to Postgres and
    queue there on the table lock -- with ``lock_timeout = 0``, for as long as the
    running build took, and with a snapshot already taken (its ``backend_xmin`` is
    set while it waits). The running build's last step, ``WaitForOlderSnapshots``,
    waits for every backend whose snapshot is older than its reference snapshot:
    the queued one among them. The queued one waits for the table lock the build
    holds. That is a cycle, the build is the later waiter, and the deadlock
    detector kills it -- after its catalog entry, its whole graph build and its
    validation scan, leaving an INVALID index behind. Observed on a production
    project with three qualifying knowledge bases (2.85M, 1.09M and 73k chunk rows
    at 1536 dimensions, all dispatched together): seven ``deadlock detected`` in
    about 36 hours, every build that reached its end killed. The 2.85M-row build
    was lost twice within the incident, each time after eight hours or more and
    about 20 GB of index, and a third, earlier death of the same build is
    consistent with it though its log did not survive (issue #95 and its follow-up
    comment). The retries livelocked, because a retry of a killed build begins
    with a repair drop, which is DDL that queues in turn and kills whichever build
    is finishing next. Revoking the other tasks by hand let each one build alone
    without incident: 25 s, 63 min and 9.6 h. Zero deadlocks after that.

    **Why the gate is here, and not in the lock queue.** No build or drop can
    wait for that lock inside Postgres safely: it has taken its snapshot before it
    queues and holds it while it waits, and an older snapshot is exactly what the
    running build's last phase waits on. ``lock_timeout`` does not
    help either -- the build's own is 0 for the reasons ``_create_index`` gives, and
    a bounded one on the *waiter* only changes which of the two dies. So the waiting
    moves out of the database: a session advisory lock on ``table_lock_relation``,
    only ever *tried*, on the same AUTOCOMMIT connection as the DDL and held until
    that dimension's last statement, so a caller that does not get it has issued
    nothing, holds no transaction and leaves nothing for a running build to wait
    on. It reports ``PerKbVectorIndexTableBusy`` and the task comes back later on
    a clock of its own -- ``PER_KB_TABLE_WAIT_COUNTDOWN_S`` apart, up to
    ``PER_KB_TABLE_MAX_WAITS`` times, uncounted while ``table_ddl_holders`` shows a
    build alive and counted like any retry when it does not (both in
    ``tasks.indexing``, which owns the retry budget). The gate is what removes the
    only waiter that could form the cycle, so the DDL itself keeps
    ``lock_timeout = 0``.

    Not every holder is a build, and only a build is worth a long wait:
    ``_holder_kind`` sorts holders into *running* (the uncounted wait),
    *stalled* (a session idle in a transaction that happens to hold the table) and
    *unknown* (a backend this role cannot see without ``pg_read_all_stats``) -- the
    last two refuse the gate just the same, and cost the task a counted retry.

    The advisory lock only stops callers of this module, so a free lock is not
    taken as a free table: ``table_ddl_holders`` is asked as well, and anything
    holding the table against a build -- an operator's ``DROP INDEX
    CONCURRENTLY``, a worker still running a version from before this gate during a
    rolling deploy, the shared per-dimension index being created -- refuses the gate
    exactly as a held lock does, with the lock given straight back. There is a
    window between that check and the DDL that nothing can close from here; it is
    the width of two round trips rather than the length of a build.

    **Operator rule: never run a manual ``CREATE`` or ``DROP INDEX CONCURRENTLY``
    on ``ai.embeddings`` while a build is running** (``table_ddl_holders``, or
    ``pg_stat_progress_create_index`` on the table) -- it queues, it holds a
    snapshot while it queues, and it kills the build at its very end, by the
    mechanism above. Wait for the build, or cancel it deliberately.

    A long read is different and is not gated: ``pg_dump``'s ``COPY`` of the table,
    or any transaction holding an old snapshot, makes the build's last phase wait
    for it too, but that is a plain wait -- the reader never waits on the build --
    so the build finishes when the reader does.

    Taken only where a dimension reaches DDL, not at the top of every reconcile: a
    knowledge base whose index is already right (a settle pass, a between-thresholds
    no-op) reads the catalog and writes at most a comment on its own index, which
    no running build waits on, and must not sit out a nine-hour build to do it. It
    is taken *after* the per-index lock and given back before it; both are only
    ever tried, so the order cannot deadlock, and it is the order that lets two
    tasks for one knowledge base still see ``build_lock_held`` rather than both
    queueing on the table.

    Released on every exit through ``_release_lock``, which discards the
    connection if the unlock fails: a pooled connection that kept this lock would
    gate the whole table until its backend ended.
    """

    def __init__(self, conn):
        self._conn = conn
        self.held = False
        self._pid = None

    def _evidence(self, include_gate_holder: bool) -> tuple[list[dict], str | None]:
        """The holders, or none and why: an unreadable answer refuses the gate too."""
        try:
            return table_ddl_holders(self._conn, include_gate_holder), None
        except Exception as exc:
            reason = first_error_line(exc)
            logger.warning(
                "Could not read which backends hold %s.embeddings (%s); not issuing any "
                "index DDL on it this time, because the answer is what tells a free table "
                "from a running build",
                AI_SCHEMA,
                reason,
            )
            return [], reason

    def hold(self) -> None:
        """Take the table, or raise ``PerKbVectorIndexTableBusy`` having issued nothing.

        Idempotent for as long as the backend it was taken on is the one behind the
        handle. It may not be: a ``RESET`` that fails after a repair or drift ``DROP``
        makes ``_discard_connection`` reconnect, the session lock ends with the old
        backend, and a ``held`` flag that outlived it let the build that follows --
        hours of ``CREATE INDEX CONCURRENTLY`` -- run with no gate at all, while
        ``release`` unlocked nothing. So the pid it was taken on is kept and checked,
        and a replaced connection takes the gate again, holder check included, before
        the next statement.
        """
        if self.held:
            if _backend_pid(self._conn) == self._pid:
                return
            logger.warning(
                "The connection holding the %s index-build gate was replaced (backend %s "
                "is gone, and its session lock with it); taking the gate again before the "
                "next statement",
                AI_SCHEMA + ".embeddings",
                self._pid,
            )
            self.held = False
        subject = table_lock_relation()
        if not _try_lock(self._conn, subject):
            holders, error = self._evidence(include_gate_holder=True)
            raise PerKbVectorIndexTableBusy(
                f"another index build or drop owns {AI_SCHEMA}.embeddings "
                f"({describe_table_holders(holders, error)})",
                holders=holders,
                lock_held=True,
                evidence_error=error,
            )
        # Held from here, so a failure in anything below still gives it back.
        self.held = True
        self._pid = _backend_pid(self._conn)
        holders, error = self._evidence(include_gate_holder=False)
        if holders or error:
            self.release()
            raise PerKbVectorIndexTableBusy(
                f"{AI_SCHEMA}.embeddings is held against an index build by a backend that "
                f"does not take this module's table lock ({describe_table_holders(holders, error)})",
                holders=holders,
                lock_held=False,
                evidence_error=error,
            )

    def release(self) -> None:
        if not self.held:
            return
        self.held = False
        _release_lock(self._conn, table_lock_relation())


def _describe_holder(holder: dict) -> str:
    ages = []
    if holder.get("running_s") is not None:
        ages.append(f"statement {holder['running_s']} s")
    if holder.get("xact_s") is not None:
        ages.append(f"transaction {holder['xact_s']} s")
    if holder.get("state") not in (None, "active") and holder.get("idle_s") is not None:
        ages.append(f"{holder['state']} for {holder['idle_s']} s")
    return (
        f"pid {holder['pid']} ({holder.get('kind', 'unknown')}: "
        f"{holder.get('backend_type') or 'backend type not visible'}, "
        f"{holder.get('lock', 'relation')} {holder.get('mode')}, "
        f"{holder.get('state') or 'state not visible'}"
        + (f", {', '.join(ages)}" if ages else "")
        + f"): {holder.get('query') or '(query not visible)'}"
    )


def describe_table_holders(holders, evidence_error: str | None = None) -> str:
    """The holders as one log fragment: pid, what kind, which lock, how long, and what."""
    if evidence_error:
        return f"which backends hold the table could not be read: {evidence_error}"
    if not holders:
        return "no backend holding the table or its index-build lock could be found"
    return "; ".join(_describe_holder(holder) for holder in holders)


# ---------------------------------------------------------------------------
# Build and drop
# ---------------------------------------------------------------------------


def _create_index(
    conn,
    kb_id: str,
    dims: int,
    mem_mb: int | None = None,
    prior_failures: int = 0,
    prior_interrupted: int = 0,
    prior_rebuilds: int = 0,
) -> None:
    """Build one index online, with room to do it in memory.

    Session-level rather than ``SET LOCAL``: this connection is in AUTOCOMMIT
    because ``CREATE INDEX CONCURRENTLY`` refuses a transaction block, and
    ``SET LOCAL`` outside a transaction affects nothing at all. Both settings
    are put back before the connection can return to the pool -- a pooled
    connection left with no statement timeout, or with a large
    ``maintenance_work_mem``, would carry them into unrelated work.

    ``statement_timeout = 0`` because a concurrent build waits for every
    transaction holding a conflicting snapshot and legitimately takes minutes on
    a large knowledge base, so a bound would turn a slow build into a failed one
    -- the same trade the BM25 index build makes. The cost is that one client
    idle in a transaction can stall this build, and with it one worker slot,
    indefinitely; it blocks no writes while it waits
    (``ShareUpdateExclusiveLock`` only).

    ``lock_timeout = 0`` for the same trade and the same wait, because
    ``statement_timeout`` does not cover it. ``CREATE INDEX CONCURRENTLY``'s
    first phase waits for every transaction that could still write a row the
    build has not seen, and it waits on that transaction's *virtual transaction
    id* -- a lock wait, which ``lock_timeout`` bounds and ``statement_timeout``
    does not. A role-level ``lock_timeout`` is exactly as realistic as a
    role-level ``statement_timeout``: images in ordinary use ship an application
    role with both set to a few seconds. Measured against a real server with a
    role-level ``lock_timeout`` of 2 s and one open write transaction, the build
    failed in 2.02 s, five attempts out of five, leaving the INVALID index this
    function then has to count.

    Unbounded is safe only because nothing of this module's can be *queued* on
    the table while it runs: the caller holds ``_TableGate``, so a second build or
    drop never reaches Postgres to wait with a snapshot the build's last phase
    would wait on. What the build can still wait for is a plain wait -- an open
    transaction, or ``pg_dump``'s long ``COPY`` of the table holding its snapshot
    -- which ends when the reader does. What it cannot survive is a statement run
    by hand on the table while it builds (see the operator rule on ``_TableGate``);
    a bound here would not change that, only which of the two dies.

    A build that runs out of disk leaves an INVALID index behind, and the next
    ensure drops and rebuilds it rather than reporting it as built (see
    ``_repair_invalid`` and its caller). ``estimated_index_mb`` says why there is
    no free-space precheck. That rebuild is how a *permanent* failure becomes a
    loop, so every failed attempt is counted onto the index it leaves behind and
    ``MAX_CONSECUTIVE_BUILD_FAILURES`` bounds it; a success clears the count,
    because it is consecutive failures that say a build is doomed.

    Both ``SET``s are inside the ``try``, and ``mem_mb`` is known before the
    first of them, so there is no window in which a statement can fail with a
    setting raised and no ``finally`` to put it back. A session-level ``SET``
    survives the pool's rollback-on-return, so a connection leaving that window
    would carry ``statement_timeout = 0`` into unrelated work for the rest of
    its life. ``mem_mb`` is passed in rather than read here for the same reason
    it is read once per ensure: the read goes through ``db.session``
    (``_release_settings_session``).
    """
    if mem_mb is None:
        mem_mb = maintenance_work_mem_mb()
    try:
        conn.execute(text("SET statement_timeout = 0"))
        conn.execute(text("SET lock_timeout = 0"))
        conn.execute(text(f"SET maintenance_work_mem = '{mem_mb}MB'"))
        conn.execute(text(per_kb_index_ddl(kb_id, dims)))
    except Exception as exc:
        # Where the loop is bounded: the attempt is counted on the INVALID index
        # the failure just left behind, so the next reconcile -- in another
        # process, after a deploy, whenever -- can see how many times this has
        # already been tried. Both prior counts are the ones read before the
        # repair drop took the previous record away with the index.
        _count_a_failed_build(
            conn, kb_id, dims, prior_failures, prior_interrupted, exc, prior_rebuilds
        )
        raise
    else:
        # On every success, not only after a failure: this write is also what
        # records the definition the index was built from, which is the only way a
        # later reconcile can tell it apart from one built before the predicate
        # changed (``per_kb_index_fingerprint``). ``prior_rebuilds`` rides through it
        # because this write is the whole comment and the drop that preceded this
        # build took the previous one away.
        _record_the_definition_built(conn, kb_id, dims, prior_rebuilds)
    finally:
        _reset_session_setting(conn, "maintenance_work_mem")
        _reset_session_setting(conn, "lock_timeout")
        _reset_session_setting(conn, "statement_timeout")


def _drop_index(conn, kb_id: str, dims: int) -> None:
    """Drop one index online, with no bound on how long the wait takes.

    ``statement_timeout = 0`` for the same reason the build gets it, and it is
    the same wait: ``DROP INDEX CONCURRENTLY`` also waits for every transaction
    holding a snapshot that could still be using the index. A role- or
    database-level ``statement_timeout`` therefore cancels a drop that is doing
    nothing wrong, and a cancelled ``DROP INDEX CONCURRENTLY`` leaves the index
    in place with ``indisvalid = false``: charged to every insert into
    ``ai.embeddings``, answering no query. Measured against a real server with a
    2 s role timeout and one open write transaction -- cancelled after 2.02 s,
    leaving exactly that state.

    ``lock_timeout = 0`` because ``statement_timeout`` never bounded that wait in
    the first place, and is therefore not the setting that has been cancelling
    these drops. The wait is on the conflicting transaction's *virtual
    transaction id*, which is a lock wait: ``statement_timeout`` does not cover
    it and ``lock_timeout`` does. Measured on the same server with a role-level
    ``lock_timeout`` of 2 s: the drop failed in 2.01 s, five attempts out of
    five, leaving ``indisvalid = false, indisready = true`` -- and on this
    function's other caller, a deleted knowledge base's drop, that state is
    permanent, because the row the index is named after is gone and nothing will
    reconcile it again.

    An ensure finds that state on its next run and repairs it. The deleted
    knowledge base's drop cannot: the row these indexes are named after is gone,
    so neither ``index_action`` nor the start-up sweep will ever look at them
    again, and a cancelled drop there is permanent. The cost of the bound being
    gone is that one client idle in a transaction can hold a drop, and with it
    one worker slot, for as long as it likes; it blocks no writes while it waits
    (``ShareUpdateExclusiveLock`` only).

    Session-level rather than ``SET LOCAL``, and reset in a ``finally``, for the
    reasons ``_create_index`` gives: the connection is in AUTOCOMMIT, so
    ``SET LOCAL`` would affect nothing, and a pooled connection left with no
    statement timeout would carry that into unrelated work.
    """
    try:
        conn.execute(text("SET statement_timeout = 0"))
        conn.execute(text("SET lock_timeout = 0"))
        conn.execute(text(per_kb_index_drop_ddl(kb_id, dims)))
    finally:
        _reset_session_setting(conn, "lock_timeout")
        _reset_session_setting(conn, "statement_timeout")


def _repair_invalid(
    conn, kb_id: str, dims: int, prior_failures: int = 0, prior_interrupted: int = 0
) -> None:
    """Drop an INVALID index left behind by a failed concurrent build.

    ``CREATE INDEX CONCURRENTLY`` that is cancelled, killed or fails leaves the
    index in place and marked invalid: it answers no query, Postgres still
    maintains it on every write, and ``IF NOT EXISTS`` makes a re-run a no-op,
    so without this the knowledge base would never get a usable index.

    **The caller decides whether to call this at all**, because two other things
    it knows come first. A build of this index still running means the entry must
    be left alone -- dropping it would pull the ground out from under that build
    -- so the caller checks ``_build_in_progress`` and reports
    ``invalid_index_build_in_progress``. That check used to live here and returned
    ``False``, which put it *after* the caller's failure-bound check: an index
    that had reached the bound with a live build on it reported
    ``build_repeatedly_failed`` and suppressed the reschedule, for the one case
    the reschedule exists for.

    A drop that *raises* is counted against the same bounds the build's own
    failure is, and by the same rule, because it is the same attempt: this is the
    second way one reconcile of an INVALID index can end without an index, and
    counting only the first left the bound reachable from one side and not the
    other. Measured before it was: five consecutive reconciles whose repair drop
    failed each recorded one failure and each asked for a build again, because the
    count is written past this point, in the build.

    It is counted through ``_count_a_failed_repair_drop`` and not through the build's
    ``_count_a_failed_build``, because the index the count lands on is the one that
    was already there: the drop is what failed, so its recorded definition has to be
    read back rather than assumed to be today's.

    The caller holds ``_TableGate`` before this runs. This drop is DDL on the shared
    table like any build, and during issue #95 it was the statement that turned one
    killed build into a livelock: each retry began here, queued on the table behind
    whichever build was running, and killed that one at its end in turn.
    """
    # "No build *of it*": this caller asked ``_build_in_progress`` about this one
    # index. It used to say "no build is running on ai.embeddings", which read as a
    # claim about the table and was logged, during issue #95, while another
    # knowledge base's build had been running on the table for hours. What makes
    # this drop safe against *those* is the table gate its caller holds.
    logger.warning(
        "Partial HNSW index %s.%s is INVALID and no build of it is running (an earlier "
        "CREATE INDEX CONCURRENTLY failed or was cancelled); dropping it",
        AI_SCHEMA,
        per_kb_index_name(kb_id, dims),
    )
    try:
        _drop_index(conn, kb_id, dims)
    except Exception as exc:
        _count_a_failed_repair_drop(conn, kb_id, dims, prior_failures, prior_interrupted, exc)
        raise


def _count_a_definition_rebuild(conn, kb_id: str, dims: int, comment: str | None) -> bool:
    """Record that a rebuild for a definition change is starting, and read it back.

    ``True`` when the record is readable off the index afterwards, which is the drift
    path's precondition for dropping anything -- see
    ``MAX_CONSECUTIVE_DEFINITION_REBUILDS`` for what each half of that bounds.

    Written before the drop rather than after the rebuild, on the index that is
    about to be replaced, for the reason every count here is written where it is:
    the record has to exist at the moment the work starts, because the work is what
    may not come back. The caller then carries the number it wrote across the drop
    that destroys this comment, because the next statement is that drop -- the
    reason this count only ever advanced when the drop *failed*, and the reason
    ``_counted_attempt`` carries the other two the same way.

    The read-back is a second catalog round trip, which is free next to a
    ``CREATE INDEX CONCURRENTLY`` that runs for minutes, and it is the only thing
    that can tell a write that worked from ``_write_index_comment`` swallowing a
    failure -- which it must, on the failure paths, and therefore does here too.

    The recorded *definition* is the index's own, never today's: the index on disk is
    still the pre-existing one at this point. ``_count_a_failed_repair_drop``
    carries the same rule and says what stamping today's costs. The two build
    counts are written back as zero because this branch runs only on a valid index,
    whose counts a successful build cleared.
    """
    rebuilds = definition_rebuilds_in(comment) + 1
    _write_index_comment(conn, kb_id, dims, 0, 0, definition_fingerprint_in(comment), rebuilds)
    return definition_rebuilds_in(index_comment(conn, kb_id, dims)) >= rebuilds


def _drop_a_drifted_index(conn, kb_id: str, dims: int) -> None:
    """Drop a *valid* index that was built from a definition this module no longer emits.

    Not ``_repair_invalid``: this drop must **not** be counted against either
    build bound. The index was working, nothing failed, and the reason it is going
    away is that this module's own DDL changed -- so counting it would spend a
    budget whose only remedy is a manual ``DROP INDEX`` on a deploy, and three
    predicate changes would turn the feature off.

    A drop that raises therefore propagates untouched, exactly as the
    below-threshold drop beside it does, and the task retries it if it was
    transient. **What it leaves behind is not always an index still answering
    queries**, which this used to claim. ``DROP INDEX CONCURRENTLY`` marks the index
    ``indisvalid = false`` and commits that before it waits, so a failure in the
    wait -- the realistic one, a role-level ``lock_timeout`` against an open
    transaction, five attempts out of five in the measurement ``_drop_index``
    records -- leaves ``indisvalid = false, indisready = true``: answering no query,
    maintained on every write, and repaired by the *build* path on the next
    reconcile rather than by this one. Only a failure before that first commit (a
    refused permission, a lock this statement never got) leaves the index untouched
    and answering, which is the case ``MAX_CONSECUTIVE_DEFINITION_REBUILDS`` then
    counts to a stop.

    Untouched by the two *build* bounds, that is. The attempt is on record before
    this is called, against ``MAX_CONSECUTIVE_DEFINITION_REBUILDS``, so a drop that
    keeps raising does not retry for ever -- which it did, because "counted against
    neither build bound" had been left to mean "counted by nothing at all".
    """
    logger.warning(
        "Rebuilding partial HNSW index %s.%s for knowledge base %s at %d dimensions: it was "
        "built from a definition this version no longer emits (its %s comment records %s, and "
        "the current definition is %s), so it covers a population or orders by an expression "
        "the searches steered onto it no longer match. Dropping and rebuilding it; until the "
        "rebuild finishes those searches use the shared per-dimension index",
        AI_SCHEMA,
        per_kb_index_name(kb_id, dims),
        kb_id,
        dims,
        "pg_class",
        definition_fingerprint_in(index_comment(conn, kb_id, dims)) or "no definition",
        per_kb_index_fingerprint(kb_id, dims),
    )
    _drop_index(conn, kb_id, dims)


def outcome_waits_for_the_table(outcome: dict) -> bool:
    """Did this ensure stop short because another build or drop owns the table?

    Its own question rather than a case of ``outcome_needs_another_attempt``,
    because the answer is on a different clock: that one is a counted, backed-off
    retry of this index's own repair, and this one can legitimately wait for a
    nine-hour build of another knowledge base's index. Nothing was issued and
    nothing counted (``_TableGate``), so the whole reconcile is simply run again
    later; ``outcome["build_alive"]`` says whether the evidence justified the long
    wait or the lock was held with nothing running on the table.
    """
    return outcome.get("status") == "table_busy"


def outcome_needs_another_attempt(outcome: dict) -> bool:
    """Did this ensure leave work only a later run can finish?

    True for exactly one case: an ``INVALID`` index that had to be left in place
    because a build of it is still running. Nothing else comes back to that
    knowledge base on its own -- the index answers no query and is maintained on
    every write until a reconcile runs again -- whereas a plain lock conflict
    means another caller is doing this index's work right now and will finish it.

    Meant for the caller that can actually reschedule (the task), so the
    condition lives here with the code that produces it rather than being
    re-derived from the dict.
    """
    return bool(outcome.get("reschedule"))


def ensure_per_kb_vector_index(knowledge_base_id: Any, engine=None, on_progress=None) -> dict:
    """Give this knowledge base a partial HNSW index per dimension it is big enough for.

    Idempotent, and a no-op whenever a partial index is not the right answer:
    too few rows, one already there and valid, more dimensions than pgvector
    will build an HNSW index for, or the project already at
    ``MAX_PER_KB_INDEXES``. At or below ``VECTOR_PER_KB_INDEX_DROP_ROWS`` an
    index that exists is dropped, so a knowledge base that shrinks -- sources
    deleted, a reindex to a different embedding model -- does not keep paying
    for one. An ``INVALID`` index is dropped and rebuilt.

    ``on_progress(status, **fields)`` is called with ``"building"`` before each
    build and ``"dropping"`` before each drop, and is given the ``dims`` and the
    ``rows`` that decided it. ``rows`` is a bounded count, so
    ``rows_are_a_floor`` says whether it stopped at the bound rather than at the
    knowledge base's real size -- reported as a plain number it would understate
    a large knowledge base by as much as the disk figure once did. A hook that
    raises is logged and does not fail the reconcile.

    Every dimension in play is attempted: one dimension being locked, declined
    or over the cap no longer abandons the rest, because a knowledge base that
    has just changed embedding model has an index to drop at the old dimension
    and one to build at the new one.

    Returns a dict with ``status``:

    * ``ready`` -- nothing left to do, and ``built``/``dropped`` say what was
      done. ``repaired_invalid_indexes`` lists the INVALID indexes dropped.
    * ``building`` -- at least one dimension belongs to another caller right
      now. ``reason`` distinguishes ``build_lock_held`` (that caller is doing
      the work) from ``invalid_index_build_in_progress``, which also sets
      ``reschedule`` -- see ``outcome_needs_another_attempt``.
    * ``skipped`` with a ``reason`` of ``index_cap_reached``,
      ``dims_above_hnsw_limit``, ``definition_rebuild_not_settling`` or
      ``build_repeatedly_failed`` -- a build this project, this embedding width or
      this database cannot have. ``definition_rebuild_not_settling`` is the mildest
      of them and the only one where the knowledge base still has a working index:
      the rebuild its definition change asks for is not completing, so it keeps the
      index it has (``MAX_CONSECUTIVE_DEFINITION_REBUILDS``), and the dimensions are
      listed in ``definition_rebuilds_not_settling``.
      ``build_repeatedly_failed`` lists the dimensions given up on in
      ``build_repeatedly_failed``, and is the one that needs an operator: the
      INVALID index stays until it is dropped by hand, which is also what lets a
      later reconcile try again. ``building`` outranks all three, because it is
      the one that is still moving.
    * ``table_busy`` -- at least one dimension needed DDL and another build or
      drop owns ``ai.embeddings`` (``_TableGate``), so nothing was issued for it and
      nothing was counted. ``table_busy`` lists those indexes, ``table_holders`` the
      evidence (``table_ddl_holders``, merged by pid across dimensions) and
      ``build_alive`` whether any holder is demonstrably running
      (``_holder_kind``). ``reason`` is ``table_ddl_in_progress`` for a running
      holder, ``table_held_without_a_build`` for holders that are stalled or that
      this role cannot see, ``table_holders_unreadable`` when the evidence could
      not be read (``evidence_error`` says why), and ``table_lock_held`` when the
      gate is held by nobody that could be found.
      Outranks everything, ``building`` included: it is the one outcome with work
      left that only this knowledge base's task will come back for
      (``outcome_waits_for_the_table``).
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    engine = _engine(engine)

    def progress(status: str, **fields: Any) -> None:
        if on_progress is None:
            return
        try:
            on_progress(status, **fields)
        except Exception as exc:
            # This is a reporting hook. Losing one event is acceptable; losing
            # the index because the recorder raised is not -- and the caller is
            # mid-loop, holding this index's build lock.
            logger.warning(
                "The vector index progress hook failed for %s at %d dimensions (%s); the "
                "reconcile carries on without the record",
                status,
                fields.get("dims", 0),
                first_error_line(exc),
                exc_info=True,
            )

    build_at, drop_below = thresholds()
    mem_mb = maintenance_work_mem_mb()
    # Both reads went through ``db.session``; give it back before a build that
    # can run for minutes starts waiting on it.
    _release_settings_session()
    cap = build_at + _COUNT_HEADROOM
    built: list[str] = []
    dropped: list[str] = []
    repaired: list[str] = []
    rebuilt: list[str] = []
    stale_kept: list[str] = []
    blocked: list[str] = []
    doomed: list[int] = []
    unsettled: list[int] = []
    reschedule: str | None = None
    cap_reached: int | None = None
    above_limit: list[int] = []
    # Dimensions that reached DDL and found the table owned by another build or
    # drop (``_TableGate``), and the evidence every refusal came with, merged by pid:
    # a later dimension that happened to look in a window with nothing to see must
    # not turn the earlier one's live build into "nothing is running".
    table_busy: list[str] = []
    table_holders: dict[int, dict] = {}
    evidence_error: str | None = None

    with _autocommit_connection(engine) as conn:
        existing = per_kb_index_states(conn, kb_id)
        dims_in_play = sorted(set(candidate_dims(conn, kb_id, cap)) | set(existing))
        if not dims_in_play:
            return {"status": "ready", "reason": "no_embeddings", "built": [], "dropped": []}

        for dims in dims_in_play:
            name = per_kb_index_name(kb_id, dims)
            lock = index_lock_relation(kb_id, dims)
            if not _try_lock(conn, lock):
                # Whoever holds it is building or dropping this very index, and
                # will finish it. The other dimensions are still ours.
                blocked.append(name)
                continue
            # Taken only where this dimension reaches DDL -- ``gate.hold()`` before
            # each of the four statements below that can issue one -- and given
            # back before the per-index lock.
            gate = _TableGate(conn)
            try:
                # Re-read under the lock: another caller may have finished
                # between the survey above and this point.
                valid, comment = per_kb_index_states(conn, kb_id).get(dims, (None, None))
                # Before the failure bound, because the bound only decides whether
                # to *build*, and this decides whether the index should exist at
                # all. ``index_action`` reads it in the same order, and when the
                # two disagreed a doomed index whose knowledge base had since
                # shrunk was dispatched for ever and never dropped: the dispatch
                # asked for the drop, this loop hit the bound first and skipped,
                # and only a manual ``DROP INDEX`` could clear it -- which also
                # left the "giving up on the repair is not giving up on the drop"
                # path in the sweep's docstring dead.
                rows = bounded_row_count(conn, kb_id, dims, cap)
                # The count stops at ``cap``, so at the bound it is a floor and
                # not the knowledge base's size. Everything that reports it says so.
                floored = rows >= cap
                prior_failures = prior_interrupted = prior_rebuilds = 0
                # Set by a drift drop, and read by the build's cap gate: this
                # dimension held a place in ``MAX_PER_KB_INDEXES`` until a moment ago.
                holds_a_place = False

                if valid is False:
                    # Read before the repair drop, which takes the record away
                    # with the index it is written on. All three, because the write
                    # that lands after the drop composes the whole comment and a
                    # count left out of it is a count erased -- and an INVALID index
                    # can be carrying a rebuild count, from a drift rebuild whose
                    # build failed.
                    prior_failures = build_failures_in(comment)
                    prior_interrupted = interrupted_builds_in(comment)
                    prior_rebuilds = definition_rebuilds_in(comment)
                    if _build_in_progress(conn, kb_id, dims):
                        # The invalid entry stays. Dropping it would pull the
                        # ground out from under that build, and falling through
                        # would reach `CREATE INDEX CONCURRENTLY IF NOT EXISTS`,
                        # which no-ops against the name the invalid index holds --
                        # reporting an index as built while it answers no query and
                        # is maintained on every write. (The BM25 path does the
                        # same, for the same reason.)
                        #
                        # We hold this index's build lock, so that build belongs
                        # to no live caller of this module: it is an orphan
                        # backend, most often from a worker that did not survive
                        # its own build. Nothing else comes back to this
                        # knowledge base, so the outcome asks to be run again --
                        # which is why this is decided before the failure bound
                        # below: an index that had reached the bound with a live
                        # build on it used to report ``build_repeatedly_failed``
                        # and suppress the one reschedule that case exists for.
                        logger.warning(
                            "Partial HNSW index %s.%s is INVALID and a build of it is still "
                            "running, so it has to be left in place: dropping it would pull "
                            "the ground out from under that build, and a rebuild would "
                            "no-op against the name it holds and report an index that "
                            "answers no query as built. Its backend outlived whatever "
                            "started it (this caller holds the build lock). Until a later "
                            "reconcile succeeds, every insert into %s.embeddings maintains "
                            "an index no search can use",
                            AI_SCHEMA,
                            name,
                            AI_SCHEMA,
                        )
                        blocked.append(name)
                        reschedule = reschedule or name
                        continue
                    # Given up on building it -- but not on dropping it. An index
                    # whose knowledge base has since fallen to the drop threshold,
                    # or whose dimension pgvector will not index at all, is dropped
                    # here whatever its history: the drop is what re-arms the build,
                    # and refusing it was what stranded the index for ever.
                    if build_is_given_up(comment) and rows > drop_below and dims <= MAX_HNSW_DIMS:
                        logger.error(
                            "Giving up on partial HNSW index %s.%s: %d consecutive builds of "
                            "it have failed and %d were interrupted, so this one is not "
                            "attempted again. Until an operator drops it by hand it stays "
                            "INVALID -- answering no query, and maintained on every write to "
                            "%s.embeddings -- and knowledge base %s keeps the shared "
                            "per-dimension index, which is what it had before this index "
                            "existed. The reason is in the failure of the last attempt (a "
                            "full disk is the usual one); dropping the index is also what "
                            "lets a later reconcile try again",
                            AI_SCHEMA,
                            name,
                            prior_failures,
                            prior_interrupted,
                            AI_SCHEMA,
                            kb_id,
                        )
                        doomed.append(dims)
                        continue
                    gate.hold()
                    _repair_invalid(conn, kb_id, dims, prior_failures, prior_interrupted)
                    repaired.append(name)
                    valid = None

                if valid is True:
                    if rows <= drop_below:
                        gate.hold()
                        progress("dropping", dims=dims, rows=rows, rows_are_a_floor=floored)
                        logger.info(
                            "Dropping partial HNSW index %s.%s: knowledge base %s now has "
                            "%d chunk rows at %d dimensions, at or below the drop threshold "
                            "of %d",
                            AI_SCHEMA,
                            name,
                            kb_id,
                            rows,
                            dims,
                            drop_below,
                        )
                        _drop_index(conn, kb_id, dims)
                        dropped.append(name)
                        continue
                    if not definition_has_drifted(kb_id, dims, comment):
                        # A pass that recomputed the fingerprint from scratch and
                        # found the index already current is the only evidence that a
                        # rebuild *settled*, as opposed to merely landing, so it is
                        # where the rebuild count is given back.
                        _settle_a_definition_rebuild(conn, kb_id, dims, comment)
                        continue
                    # Built from a definition this version no longer emits. The
                    # index still answers searches, at whatever recall its old
                    # definition gives, so it is only replaced where the
                    # replacement can actually be completed: a drop this loop
                    # cannot follow with a build leaves the knowledge base with no
                    # index where it had a stale-but-usable one.
                    if rows < build_at or dims > MAX_HNSW_DIMS:
                        # Inside the hysteresis band, or a width pgvector will not
                        # index: a build would be declined below, so keeping it is
                        # strictly better than dropping it.
                        why = (
                            f"{rows} chunk rows is below the build threshold of {build_at}"
                            if rows < build_at
                            else f"{dims} dimensions is above pgvector's HNSW limit of "
                            f"{MAX_HNSW_DIMS}"
                        )
                        logger.warning(
                            "Partial HNSW index %s.%s was built from a definition this "
                            "version no longer emits, and is being kept rather than "
                            "rebuilt, because %s -- so a rebuild would be declined and the "
                            "drop would leave knowledge base %s with no index of its own. "
                            "Its searches keep whatever recall the old definition gives",
                            AI_SCHEMA,
                            name,
                            why,
                            kb_id,
                        )
                        stale_kept.append(name)
                        continue
                    if definition_rebuild_is_given_up(comment):
                        # The drift path's own bound, reached. Free to read, so it
                        # comes before the catalog count -- but after the two gates
                        # above, because where a rebuild would be declined anyway
                        # this bound has nothing to say and its log line would send
                        # an operator after the wrong thing.
                        logger.error(
                            "Not rebuilding partial HNSW index %s.%s for its definition "
                            "change: %d consecutive rebuilds of it have been started and none "
                            "has settled, so no more are started and it is kept as it is. This "
                            "is not the build bound and nothing here is INVALID -- the index is "
                            "valid and still answering knowledge base %s at whatever recall its "
                            "old definition gives. Either its drop keeps failing, or the "
                            "%s comment recording the new definition is not sticking (check "
                            "that this role may COMMENT ON the index), or two deploys are "
                            "emitting two definitions and rebuilding each other's index -- for "
                            "which the fix is sequencing the deploy and not this index. "
                            "Clearing the comment, or dropping the index by hand, starts the "
                            "rebuild again",
                            AI_SCHEMA,
                            name,
                            definition_rebuilds_in(comment),
                            kb_id,
                            "pg_class",
                        )
                        stale_kept.append(name)
                        unsettled.append(dims)
                        continue
                    # Only now, because the three cheap gates above decide it more
                    # often and this is a catalog count.
                    total = per_kb_index_count(conn)
                    if total >= MAX_PER_KB_INDEXES:
                        # The drop would free a place another knowledge base's
                        # reconcile can take before this one rebuilds, and this
                        # knowledge base would be left with nothing. A stale index
                        # answers its searches; no index does not.
                        logger.warning(
                            "Partial HNSW index %s.%s was built from a definition this "
                            "version no longer emits, and is being kept rather than "
                            "rebuilt, because %s.embeddings already carries %d "
                            "per-knowledge-base indexes (the cap is %d): the drop would free "
                            "a place another knowledge base can take before the rebuild, "
                            "leaving knowledge base %s with no index at all. Its searches "
                            "keep whatever recall the old definition gives. Lower the number "
                            "of indexed knowledge bases, or drop this one by hand once there "
                            "is room",
                            AI_SCHEMA,
                            name,
                            AI_SCHEMA,
                            total,
                            MAX_PER_KB_INDEXES,
                            kb_id,
                        )
                        stale_kept.append(name)
                        cap_reached = total
                        continue
                    # Before the count, not just before the drop: the count spends one
                    # of ``MAX_CONSECUTIVE_DEFINITION_REBUILDS``, and a busy table
                    # refusing the drop after it would spend it on a rebuild that
                    # never started -- three busy passes and the index is frozen on
                    # its old definition.
                    gate.hold()
                    if not _count_a_definition_rebuild(conn, kb_id, dims, comment):
                        # The attempt could not be put on record, so the bound above
                        # cannot advance and nothing would stop the next reconcile
                        # doing this again -- a real DROP and CREATE INDEX
                        # CONCURRENTLY per reconcile, with both timeouts lifted and
                        # its own maintenance_work_mem, for ever. Not starting is the
                        # only sound answer: the stale index answers its searches.
                        logger.error(
                            "Not rebuilding partial HNSW index %s.%s for its definition "
                            "change: the attempt could not be recorded on the index (the "
                            "%s comment does not read back), so nothing would stop the next "
                            "reconcile dropping and rebuilding it again, and the one after "
                            "that. Keeping it instead -- knowledge base %s keeps whatever "
                            "recall its old definition gives, which is what it had. Check that "
                            "this role may COMMENT ON the index; the warning just above says "
                            "why the write failed",
                            AI_SCHEMA,
                            name,
                            "pg_class",
                            kb_id,
                        )
                        stale_kept.append(name)
                        unsettled.append(dims)
                        continue
                    # The number just written, carried across the drop that is about
                    # to destroy the object it is written on. Everything else in this
                    # module that survives a drop survives the same way
                    # (``_counted_attempt``); this count did not, so it only ever
                    # advanced when the drop *failed*.
                    prior_rebuilds = definition_rebuilds_in(comment) + 1
                    progress("dropping", dims=dims, rows=rows, rows_are_a_floor=floored)
                    _drop_a_drifted_index(conn, kb_id, dims)
                    rebuilt.append(name)
                    # The definition drop is not a build failure, so neither build
                    # count moves; the index is gone, so the build below is the
                    # rebuild, and the cap may not decline it (``holds_a_place``).
                    holds_a_place = True
                    valid = None

                if rows < build_at:
                    continue
                if dims > MAX_HNSW_DIMS:
                    logger.warning(
                        "Not building partial HNSW index %s.%s: %d dimensions is above "
                        "pgvector's HNSW limit of %d, so this build would fail every time it "
                        "was attempted, and is not attempted or dispatched again. Searches of "
                        "knowledge base %s stay exact -- at this width no HNSW index is "
                        "possible at all, the shared per-dimension one included -- until it "
                        "is reindexed with a narrower embedding model",
                        AI_SCHEMA,
                        name,
                        dims,
                        MAX_HNSW_DIMS,
                        kb_id,
                    )
                    above_limit.append(dims)
                    continue
                # The cap is not counted again when the drift drop a moment ago freed
                # this index's own place. It used to be, and the window between the
                # two counts lost the index: another knowledge base's reconcile taking
                # the freed place in between made this rebuild declined, leaving this
                # knowledge base with *nothing* where it had a stale-but-usable index
                # -- reported as ``rebuilt_stale_definitions: [name]`` with
                # ``built: []``. That outcome is worse than either of the two the cap
                # is allowed to choose between (keep the stale one, or replace it), and
                # the cap's purpose -- bound how many indexes one relation carries
                # while any query on it is planned -- is not served by refusing to put
                # back an index that was inside the bound a moment ago. The gate before
                # the drop is where the cap decides this knowledge base's case, and it
                # decides it once.
                if not holds_a_place:
                    total = per_kb_index_count(conn)
                    if total >= MAX_PER_KB_INDEXES:
                        logger.warning(
                            "Not building partial HNSW index %s.%s: %s.embeddings already "
                            "carries %d per-knowledge-base indexes (the cap is %d), and every "
                            "index on a relation is opened and locked while planning any query "
                            "on it. This knowledge base keeps the shared index",
                            AI_SCHEMA,
                            name,
                            AI_SCHEMA,
                            total,
                            MAX_PER_KB_INDEXES,
                        )
                        cap_reached = total
                        continue
                gate.hold()
                progress("building", dims=dims, rows=rows, rows_are_a_floor=floored)
                floor = "at least " if floored else ""
                logger.info(
                    "Building partial HNSW index %s.%s for knowledge base %s (%s%d chunk rows at "
                    "%d dimensions, threshold %d); it needs %s%d MB of disk, and blocks no "
                    "writes.%s",
                    AI_SCHEMA,
                    name,
                    kb_id,
                    floor,
                    rows,
                    dims,
                    build_at,
                    floor,
                    estimated_index_mb(rows, dims),
                    (
                        f" Both figures are floors: the row count stops at {cap}, so a "
                        f"knowledge base ten times that size builds an index ten times this "
                        f"one."
                        if floored
                        else ""
                    ),
                )
                _create_index(
                    conn,
                    kb_id,
                    dims,
                    mem_mb,
                    prior_failures=prior_failures,
                    prior_interrupted=prior_interrupted,
                    prior_rebuilds=prior_rebuilds,
                )
                built.append(name)
            except PerKbVectorIndexTableBusy as busy:
                # Refused before this dimension issued anything: no DDL, no session
                # setting lifted, no count written -- so neither build bound moves,
                # which is right, because nothing was attempted. The task comes back
                # on a clock of its own (``outcome_waits_for_the_table``) and logs the
                # wait with the evidence, so this line is only for a debug trace.
                logger.debug(
                    "Not touching partial HNSW index %s.%s yet: %s",
                    AI_SCHEMA,
                    name,
                    busy,
                )
                progress(
                    "waiting_for_table",
                    dims=dims,
                    holder_pids=[holder["pid"] for holder in busy.holders],
                )
                table_busy.append(name)
                for holder in busy.holders:
                    table_holders.setdefault(holder["pid"], holder)
                evidence_error = evidence_error or busy.evidence_error
                continue
            finally:
                gate.release()
                _release_lock(conn, lock)

    outcome: dict = {"status": "ready", "built": built, "dropped": dropped}
    if repaired:
        outcome["repaired_invalid_indexes"] = repaired
    if rebuilt:
        outcome["rebuilt_stale_definitions"] = rebuilt
    if stale_kept:
        outcome["stale_definitions_kept"] = stale_kept
    if above_limit:
        outcome["dims_above_hnsw_limit"] = above_limit
        outcome.update({"status": "skipped", "reason": "dims_above_hnsw_limit"})
    if cap_reached is not None:
        outcome.update(
            {"status": "skipped", "reason": "index_cap_reached", "index_count": cap_reached}
        )
    if unsettled:
        # Reported before the build bound and after the cap, in the same order the
        # loop decides: a knowledge base with no index at all is the worse state, and
        # this one has a working index. Its own reason so an operator is not told
        # "build_repeatedly_failed" about an index that is valid and answering.
        outcome["definition_rebuilds_not_settling"] = unsettled
        outcome.update({"status": "skipped", "reason": "definition_rebuild_not_settling"})
    if doomed:
        # Reported after the other skips: a build that has failed every time is
        # the one an operator has to act on, so its reason is the one that
        # survives.
        outcome["build_repeatedly_failed"] = doomed
        outcome.update({"status": "skipped", "reason": "build_repeatedly_failed"})
    if blocked:
        # Still moving somewhere, which outranks a skip: reported last so its
        # reason is the one that survives.
        outcome["status"] = "building"
        outcome["index"] = reschedule or blocked[0]
        outcome["reason"] = "invalid_index_build_in_progress" if reschedule else "build_lock_held"
        if reschedule:
            outcome["reschedule"] = True
    if table_busy:
        # Outranks even ``building``: it is the one outcome that has work left
        # which only this knowledge base's own task will come back for, and it
        # comes back on a clock of its own. ``build_alive`` is what the task spends
        # its uncounted wait on, so the evidence travels with it; the skip lists
        # set above stay in the dict for the task's summary line.
        holders = list(table_holders.values())
        alive = any(holder["kind"] == "running" for holder in holders)
        if alive:
            reason = "table_ddl_in_progress"
        elif holders:
            reason = "table_held_without_a_build"
        elif evidence_error:
            reason = "table_holders_unreadable"
        else:
            reason = "table_lock_held"
        outcome.update(
            {
                "status": "table_busy",
                "reason": reason,
                "index": table_busy[0],
                "table_busy": table_busy,
                "table_holders": holders,
                "build_alive": alive,
            }
        )
        if evidence_error:
            outcome["evidence_error"] = evidence_error
    return outcome


def drop_per_kb_vector_indexes(knowledge_base_id: Any, engine=None) -> dict:
    """Drop every partial HNSW index this knowledge base owns.

    What a deleted knowledge base needs: the row is gone, so nothing will ever
    reconcile the index again, and Postgres would keep maintaining an index
    named after a knowledge base that no longer exists. Every dimension it has
    one at, not only the current one: the embedding model may have changed
    since.

    A transient database error is re-raised so the task retries rather than
    reporting a drop that did not happen, and a *permanent* one raises
    ``PerKbVectorIndexDropFailed`` for the same reason turned up to ERROR: it is
    the case where the index really is orphaned, and no retry, dispatch or
    start-up sweep will ever reach it again. A dimension that cannot be dropped
    for a permanent reason does not strand the others: the loop carries on and
    the failures are reported together at the end.

    A *lock conflict* is the one thing that does stop the loop, by raising
    ``PerKbVectorIndexBuildInProgress`` where it happens. That is deliberate and
    the opposite case: another caller is working on this index right now, so the
    whole drop is worth retrying rather than partly completing, and the retry
    reaches the dimensions this attempt did not. A table owned by another build
    raises its subclass, ``PerKbVectorIndexTableBusy``, having issued nothing, and
    the task waits for that build on the longer clock the reconcile's task uses.

    "Permanent" here means only "not on ``is_transient_db_error``'s list", and that
    list is about a *statement* failing. Some whole-server conditions are not on it
    -- ``53300 too_many_connections`` is the reachable one -- so a connection storm
    during a knowledge base's deletion raises the ERROR below about an orphaned
    index that in fact drops cleanly on the next attempt. The ERROR says so rather
    than the classifier being widened here: it is shared with the BM25 path, where
    the same sqlstate means something else.
    """
    kb_id = _validated_kb_id(knowledge_base_id)
    engine = _engine(engine)

    dropped: list[str] = []
    failed: list[str] = []
    with _autocommit_connection(engine) as conn:
        for dims in sorted(existing_per_kb_indexes(conn, kb_id)):
            name = per_kb_index_name(kb_id, dims)
            lock = index_lock_relation(kb_id, dims)
            if not _try_lock(conn, lock):
                raise PerKbVectorIndexBuildInProgress(
                    f"{AI_SCHEMA}.{name} is being built or dropped by another caller"
                )
            # The same DDL on the same table as the reconcile's, so the same gate:
            # a deleted knowledge base's drop queued behind another knowledge base's
            # build kills it at its end exactly as a repair drop does (``_TableGate``).
            gate = _TableGate(conn)
            try:
                gate.hold()
                _drop_index(conn, kb_id, dims)
                dropped.append(name)
            except PerKbVectorIndexTableBusy:
                # Nothing issued. Stops the loop like a held index lock does, and for
                # the same reason: the whole drop is worth retrying, and the retry
                # reaches the dimensions this attempt did not.
                raise
            except Exception as exc:
                if is_transient_db_error(exc):
                    raise
                failed.append(name)
                logger.warning(
                    "Could not drop partial HNSW index %s.%s", AI_SCHEMA, name, exc_info=True
                )
            finally:
                gate.release()
                _release_lock(conn, lock)

    if failed:
        names = ", ".join(f"{AI_SCHEMA}.{name}" for name in failed)
        logger.error(
            "Could not drop %d of deleted knowledge base %s's partial HNSW index(es), for a "
            "reason a retry cannot get past: %s. They are orphaned -- named after a knowledge "
            "base that no longer exists, answering no query, and maintained by Postgres on "
            "every write to %s.embeddings -- and have to be dropped by hand. Check the "
            "warning above each name first: a whole-server condition that is not classified "
            "as transient, such as too_many_connections, reaches this message too, and that "
            "drop succeeds on the next attempt. Dropped successfully: %s",
            len(failed),
            kb_id,
            names,
            AI_SCHEMA,
            ", ".join(dropped) or "(none)",
        )
        raise PerKbVectorIndexDropFailed(
            f"could not drop {names} of deleted knowledge base {kb_id}",
            dropped_indexes=dropped,
            failed_indexes=failed,
        )
    return {"status": "dropped", "indexes": dropped}


# ---------------------------------------------------------------------------
# Start-up sweep
# ---------------------------------------------------------------------------

# Upper bound on the one grouped count the start-up sweep runs. It reads
# ``ai.embeddings`` once, which on a large project is seconds rather than
# milliseconds; past this it is abandoned, because a start-up must not wait on
# it. Nothing is lost by abandoning it: the next source to finish indexing in
# each knowledge base dispatches the same reconcile, and the catalog cases
# (an INVALID index, an index whose knowledge base has emptied) are found
# without this count at all.
#
# 5 s is not borrowed from another bound -- the migrations' own boot-path
# ``lock_timeout`` is 10 s. It comes from the measurement: 14 ms over 66,000
# embeddings, so about 1.1 s extrapolated to 5.3 million rows. That leaves
# several times the slowest count worth waiting for, while staying short enough
# that a start-up cannot look hung on it.
SWEEP_TIMEOUT_MS = 5_000

# Upper bound on how many reconciles one start-up sets off. ``MAX_PER_KB_INDEXES``
# caps how many of these indexes a project may hold, not how many builds may be
# in flight, and every dispatched build runs with ``statement_timeout = 0`` and
# asks for ``maintenance_work_mem`` of its own. The first boot after this
# deploys, on a project with 30 knowledge bases over the threshold -- exactly the
# population the feature is for -- would otherwise queue 30 of them at once,
# against a database that may have 512 MiB in total. 10 bounds that at about
# 1.3 GB of build memory at the default setting even if the queue runs them all
# in parallel, and is more than a project crosses the threshold with between two
# boots in practice.
#
# The queue no longer runs them in parallel against the database, and nothing
# here has to arrange that: every dispatched reconcile that needs DDL takes the
# table gate (``_TableGate``) first, so one builds and the others return
# ``table_busy`` at once and wait in the task queue, not in Postgres' lock queue.
# The dispatch stays plain -- no stagger, no countdown -- because the gate is the
# only serialiser, and a second one here would be a second place to get it wrong.
#
# Nothing is dropped by the cap: the next source to finish indexing in each
# knowledge base dispatches the same reconcile, and so does the next start-up.
# The ``INVALID`` indexes are first in the list because they answer no query
# while Postgres maintains them on every write, so they are the ones that must
# not be deferred.
MAX_SWEEP_DISPATCH = 10

_THRESHOLD_KEYS = ("VECTOR_PER_KB_INDEX_MIN_ROWS", "VECTOR_PER_KB_INDEX_DROP_ROWS")


def kbs_needing_a_per_kb_index(engine=None) -> list[str]:
    """Knowledge bases whose partial HNSW indexes are out of step, for the start-up sweep.

    Three cases, in one pass: a knowledge base at or above the build threshold
    with no index, one at or below the drop threshold that has one, and one whose
    index is ``INVALID``. The last of those is read from the catalog, which is
    cheap; the first two need the grouped count, which is not, so it runs under
    ``SWEEP_TIMEOUT_MS``.

    That count is restricted to ``PER_KB_INDEX_ITEM_TABLE`` for the reason
    ``bounded_row_count`` is: the two thresholds are about the population the index
    covers, and this query and that one have to agree about which rows those are or
    the boot dispatches builds the reconcile then declines.

    Two more come from the same catalog read and neither needs the count: an index
    built from a definition this version no longer emits, and -- last of all -- one
    that is correct but still carries rebuilds nothing has confirmed. The second
    exists because a boot is the only event that can confirm them: this process
    recomputed the fingerprint from its own DDL and the index already matches it,
    which is what ``_settle_a_definition_rebuild`` accepts and what gives the rebuild
    budget back. Nothing else ever brings a reconcile back to a knowledge base whose
    index is already right, so without it the count written by one definition change
    would outlive the index and a handful of changes would spend the whole bound. The
    reconcile it asks for reads a comment and writes a shorter one; it starts no
    build and drops nothing.

    An ``INVALID`` index that has already reached
    ``MAX_CONSECUTIVE_BUILD_FAILURES`` is not one of them: the reconcile would
    read the same count and decline, so dispatching it only spends one of
    ``MAX_SWEEP_DISPATCH`` places -- and because the list is ``INVALID``-first,
    that many given-up indexes would fill it on every boot. Its history comes from
    the comment column on the catalog SELECT above, not from a read per index. It
    is still dispatched for the other two reasons: giving up on the *repair* is not
    giving up on the *drop*, and a knowledge base that has since fallen below the
    drop threshold wants the index gone -- which is also what re-arms the build.

    An index past ``MAX_CONSECUTIVE_DEFINITION_REBUILDS`` is left out for the same
    reason and is the one state in this module nothing else would ever mention: it is
    *valid*, so ``index_action`` skips it and no reconcile runs to log the give-up
    ERROR, and fixing the cause does not re-arm it. It gets a warning of its own
    here, which is the only place it is ever named.

    An abandoned count is not evidence about any knowledge base, so when it
    fails nothing is concluded from it -- only the ``INVALID`` indexes are
    returned. In particular the "a knowledge base whose rows are all gone still
    has its index" case cannot be told apart from a count that never ran, so it
    is only considered when the count finished.

    One case it does not catch: an index at a dimension the knowledge base no
    longer has any rows at -- an embedding model change, where the grouped count
    has a group for the new dimension and none for the old one, so neither the
    "no index" nor the "all rows gone" test fires for the stale one. Left to
    ``index_action``, which unions the dimensions in play with the dimensions that
    have an index and asks for the drop on the next source that finishes indexing.

    At most ``MAX_SWEEP_DISPATCH`` ids come back, because each one can start an
    unbounded index build.

    Never raises, and never touches ``db.session``: this runs inside the boot's
    migration transaction, where a failing statement would abort the boot's own
    work (see ``read_overrides``).
    """
    engine = _engine(engine)
    needing: dict[str, None] = {}
    drifted: dict[str, None] = {}
    settling: dict[str, None] = {}
    given_up: list[str] = []
    given_up_rebuilds: list[str] = []

    try:
        with engine.connect() as conn:
            build_at, drop_below = thresholds(read_overrides(conn, *_THRESHOLD_KEYS))
            rows = _index_catalog_rows(conn, INDEX_NAME_PREFIX)
            indexed: dict[str, list[int]] = {}
            for relname, valid, comment in rows:
                parsed = parsed_per_kb_index_name(relname)
                if parsed is None:
                    continue
                kb_id, dims = parsed
                indexed.setdefault(kb_id, []).append(dims)
                if not valid:
                    if build_is_given_up(comment):
                        # Nothing would come of dispatching this one: the reconcile
                        # reads the same counts and declines. Left in the list it
                        # would take one of ``MAX_SWEEP_DISPATCH`` places -- and the
                        # list is INVALID-first, so that many given-up indexes would
                        # consume the whole start-up budget on every boot while a
                        # repairable one was never reached.
                        given_up.append(
                            f"{AI_SCHEMA}.{relname} ({build_failures_in(comment)} failed, "
                            f"{interrupted_builds_in(comment)} interrupted)"
                        )
                        continue
                    needing[kb_id] = None
                elif definition_has_drifted(kb_id, dims, comment):
                    if definition_rebuild_is_given_up(comment):
                        # The drift bound reached: nothing dispatches this index
                        # again, so this boot is the only place it is ever mentioned
                        # -- see the warning below.
                        given_up_rebuilds.append(
                            f"{AI_SCHEMA}.{relname} "
                            f"({definition_rebuilds_in(comment)} rebuilds started)"
                        )
                    else:
                        # Built from a definition this version no longer emits: it
                        # still answers searches, at whatever recall the old
                        # definition gives, so it is the least urgent of the cases
                        # and is added after the others rather than here. A catalog
                        # fact, like INVALID, so it does not depend on the count
                        # below finishing.
                        drifted[kb_id] = None
                elif 0 < definition_rebuilds_in(comment) < MAX_CONSECUTIVE_DEFINITION_REBUILDS:
                    # Correct, and carrying rebuilds that nothing has confirmed. A
                    # boot is the event that confirms them: this process recomputed
                    # the fingerprint from its own DDL and the index already matches
                    # it, which is the only evidence ``_settle_a_definition_rebuild``
                    # accepts. Nothing else brings a reconcile back to a knowledge
                    # base whose index is already right -- ``index_action`` returns
                    # None for it and none of the cases above fires -- so without
                    # this dispatch the count would never be given back and a handful
                    # of ordinary definition changes over a project's life would
                    # spend the whole bound. Least urgent of all, and it happens
                    # once: the pass it asks for clears the count.
                    settling[kb_id] = None
            conn.rollback()
    except Exception:
        logger.warning(
            "Could not read the per-knowledge-base HNSW indexes at start-up", exc_info=True
        )
        return []

    if given_up:
        # Both numbers, per index, because the gate is either bound: naming only
        # ``MAX_CONSECUTIVE_BUILD_FAILURES`` reported an index with 0 failures and 25
        # interrupted builds as "3 consecutive failed builds", which sends the first
        # log read after a restart looking for a full disk when the cause was
        # contention -- and the interrupted count is new, so that is the likely branch
        # for a while.
        logger.warning(
            "%d partial HNSW index(es) are INVALID and have reached one of the two build "
            "bounds (%d consecutive failed builds, or %d interrupted), so this start-up does "
            "not reconcile them: %s%s. Each answers no query and is maintained on "
            "every write to %s.embeddings until an operator drops it by hand, which is also what "
            "lets a later reconcile try again",
            len(given_up),
            MAX_CONSECUTIVE_BUILD_FAILURES,
            MAX_CONSECUTIVE_INTERRUPTED_BUILDS,
            ", ".join(given_up[:MAX_SWEEP_DISPATCH]),
            f" (and {len(given_up) - MAX_SWEEP_DISPATCH} more)"
            if len(given_up) > MAX_SWEEP_DISPATCH
            else "",
            AI_SCHEMA,
        )

    if given_up_rebuilds:
        # The one permanent state in this module that nothing else ever says a word
        # about. An index past either *build* bound is INVALID, so it is in the list
        # above and in the reconcile's own give-up ERROR; an index past the *drift*
        # bound is valid, answers every search from a definition this version no
        # longer emits, and is skipped by ``index_action`` and by the dispatch above
        # -- so the reconcile that would log about it is never run, and fixing
        # whatever caused it does not re-arm anything. Without this line the whole
        # state is invisible, across restarts, for as long as the index exists.
        logger.warning(
            "%d partial HNSW index(es) are valid but built from a definition this version no "
            "longer emits, and have reached the rebuild bound (%d consecutive rebuilds "
            "started, none confirmed settled), so no more are started and this start-up does "
            "not reconcile them: %s%s. Each answers its knowledge base at whatever recall its "
            "old definition gives, and keeps doing so until an operator clears the %s comment "
            "or drops the index by hand. Either its drop keeps failing, or the comment "
            "recording the new definition is not sticking, or two deploys are emitting two "
            "definitions and rebuilding each other's index",
            len(given_up_rebuilds),
            MAX_CONSECUTIVE_DEFINITION_REBUILDS,
            ", ".join(given_up_rebuilds[:MAX_SWEEP_DISPATCH]),
            f" (and {len(given_up_rebuilds) - MAX_SWEEP_DISPATCH} more)"
            if len(given_up_rebuilds) > MAX_SWEEP_DISPATCH
            else "",
            "pg_class",
        )

    counted_ok = True
    try:
        with engine.connect() as conn:
            conn.execute(
                text("SELECT set_config('statement_timeout', :ms, true)"),
                {"ms": str(SWEEP_TIMEOUT_MS)},
            )
            counted = conn.execute(
                text(
                    "SELECT knowledge_base_id::text, dims, count(*) FROM "
                    f'"{AI_SCHEMA}".embeddings '
                    "WHERE item_table = :item_table GROUP BY 1, 2"
                ),
                {"item_table": PER_KB_INDEX_ITEM_TABLE},
            ).all()
            conn.rollback()
    except Exception as exc:
        counted_ok = False
        logger.warning(
            "Could not count embeddings per knowledge base at start-up (%s); only the "
            "knowledge bases whose index is INVALID are reconciled now, and the rest are "
            "picked up as their sources finish indexing or at the next start-up",
            first_error_line(exc),
        )
        counted = []

    if counted_ok:
        for kb_id, dims, count in counted:
            has_index = int(dims) in indexed.get(kb_id, ())
            if (not has_index and count >= build_at) or (has_index and count <= drop_below):
                needing[kb_id] = None
        # A knowledge base whose rows are all gone still has its index, and the
        # grouped count above cannot see it (no rows, no group). Only sound
        # because the count finished: an abandoned one is empty for a reason
        # that says nothing about any knowledge base.
        for kb_id in indexed:
            if not any(r[0] == kb_id for r in counted):
                needing[kb_id] = None

    # Last, so a stale-but-usable index only ever spends a place
    # ``MAX_SWEEP_DISPATCH`` had left over. The other three cases are more urgent:
    # an INVALID index answers nothing, an index past the drop threshold is charged
    # to every write, and a knowledge base over the build threshold is what the
    # feature is for. ``needing`` is a dict so a knowledge base already in it for
    # another reason does not move.
    for kb_id in drifted:
        needing[kb_id] = None
    # After the drifted ones, because this one starts no build and drops nothing: it
    # asks a reconcile to read a comment and write a shorter one.
    for kb_id in settling:
        needing[kb_id] = None

    pending = list(needing)
    if len(pending) > MAX_SWEEP_DISPATCH:
        logger.warning(
            "%d knowledge bases need their partial HNSW index reconciled; dispatching the "
            "first %d and leaving %d for their next indexed source or the next start-up, "
            "because each dispatch can start an index build with no statement timeout",
            len(pending),
            MAX_SWEEP_DISPATCH,
            len(pending) - MAX_SWEEP_DISPATCH,
        )
        pending = pending[:MAX_SWEEP_DISPATCH]
    return pending
