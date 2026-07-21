"""Per-knowledge-base sparse index storage with caching.

Manages sparse index files on the filesystem, one index per knowledge base
per item table (chunks, full_documents, graph_index_nodes).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import threading
from collections import OrderedDict

from .bm25_index import BM25IndexManager
from .config import SPARSE_INDEX_BASE_PATH, SPARSE_INDEX_CACHE_SIZE

logger = logging.getLogger(__name__)


class SparseIndexStore:
    """Per-KB sparse index storage with LRU caching.

    Manages the lifecycle of sparse indexes stored on the filesystem.
    Each knowledge base gets its own directory structure:

        {base_path}/{kb_id}/{item_table}/bm25/
            - index files (from bm25s.save())
            - item_ids.json (ID mapping)

    Thread-safe with in-memory caching of loaded indexes. Uses LRU eviction
    to prevent unbounded memory growth.
    """

    # Class-level cache shared across instances (OrderedDict for LRU)
    _cache_lock = threading.Lock()
    _managers: OrderedDict[str, BM25IndexManager] = OrderedDict()
    _loading_events: dict[str, threading.Event] = {}
    _max_cache_size: int = SPARSE_INDEX_CACHE_SIZE

    def __init__(
        self,
        knowledge_base_id: str,
        base_path: str = SPARSE_INDEX_BASE_PATH,
    ):
        """Initialize sparse index store.

        Args:
            knowledge_base_id: KB identifier — accepted as either ``str`` or
                ``uuid.UUID``. Coerced to ``str`` so the value can flow into
                ``os.path.join()`` regardless of how the caller obtained it
                (SQLAlchemy hands out ``uuid.UUID`` instances, the API layer
                hands out strings).
            base_path: Root directory for all sparse indexes.
        """
        self.kb_id = str(knowledge_base_id)
        self.base_path = base_path

    def get_index_path(self, item_table: str = "chunks") -> str:
        """Get filesystem path for this KB's sparse index.

        Args:
            item_table: Table name (chunks, full_documents, graph_index_nodes).

        Returns:
            Directory path for the index.
        """
        return os.path.join(self.base_path, self.kb_id, item_table, "bm25")

    def index_exists(self, item_table: str = "chunks") -> bool:
        """Check if index files exist on disk.

        Args:
            item_table: Table name.

        Returns:
            True if index files exist.
        """
        path = self.get_index_path(item_table)
        item_ids_file = os.path.join(path, "item_ids.json")
        return os.path.exists(item_ids_file)

    def _metadata_path(self, item_table: str) -> str:
        return os.path.join(self.get_index_path(item_table), "metadata.json")

    def write_metadata(self, item_table: str, item_count: int) -> None:
        """Write the post-build metadata.json sidecar.

        Records the item_count and built_at timestamp; used by
        bm25_status to detect when the on-disk index is stale relative
        to the current items table.
        """
        path = self._metadata_path(item_table)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "item_count": int(item_count),
        }
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)

    def read_metadata(self, item_table: str) -> dict | None:
        """Read the metadata.json sidecar; returns None if missing or malformed."""
        path = self._metadata_path(item_table)
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    @classmethod
    def _evict_if_needed(cls) -> None:
        """Evict oldest cache entries if exceeding max size.

        Must be called while holding _cache_lock.
        """
        while len(cls._managers) > cls._max_cache_size:
            key, _ = cls._managers.popitem(last=False)
            # If there's a loading event, set it to unblock waiters before evicting
            event = cls._loading_events.pop(key, None)
            if event:
                event.set()
            logger.debug("Evicted sparse index cache entry: %s", key)

    def get_or_load_manager(self, item_table: str = "chunks") -> BM25IndexManager:
        """Get cached manager or load from disk.

        Thread-safe: uses locking and events to prevent concurrent threads from
        receiving unloaded managers. Loading is done outside the lock to avoid
        blocking unrelated cache operations.

        Args:
            item_table: Table name.

        Returns:
            BM25IndexManager instance (may be empty if no index exists).
        """
        cache_key = f"{self.kb_id}:{item_table}"

        while True:
            with self._cache_lock:
                # Check if manager exists in cache
                if cache_key in self._managers:
                    manager = self._managers[cache_key]
                    # If no loading event, manager is fully loaded and ready
                    if cache_key not in self._loading_events:
                        self._managers.move_to_end(cache_key)
                        return manager
                    # Otherwise, get the event to wait on
                    event = self._loading_events[cache_key]
                else:
                    # We're the first thread - create manager and loading event
                    manager = BM25IndexManager()
                    self._managers[cache_key] = manager
                    event = threading.Event()
                    self._loading_events[cache_key] = event
                    self._evict_if_needed()
                    break  # Exit loop to perform loading

            # Wait for another thread to finish loading, then retry
            event.wait()

        # We're the loader - load OUTSIDE the lock to avoid blocking
        try:
            path = self.get_index_path(item_table)
            if self.index_exists(item_table):
                try:
                    manager.load(path)
                    logger.debug(
                        "Loaded sparse index: %s (%d docs)",
                        cache_key,
                        len(manager._item_ids),
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to load sparse index %s: %s",
                        cache_key,
                        e,
                    )
                    # Return empty manager, will fall back to legacy search
        finally:
            # Signal completion and remove loading event
            with self._cache_lock:
                event.set()
                self._loading_events.pop(cache_key, None)

        return manager

    def build_and_save(
        self,
        documents: list[str],
        item_ids: list[str],
        item_table: str = "chunks",
    ) -> dict:
        """Build index from scratch and save to disk.

        Use this for initial indexing or full rebuilds.

        Args:
            documents: Document texts to index.
            item_ids: Corresponding item IDs.
            item_table: Table name.

        Returns:
            Stats dict from build_index().
        """
        manager = BM25IndexManager()
        stats = manager.build_index(documents, item_ids)

        if not manager.is_empty():
            manager.save(self.get_index_path(item_table))

        # Update cache, clearing any in-progress loading
        cache_key = f"{self.kb_id}:{item_table}"
        with self._cache_lock:
            self._managers[cache_key] = manager
            # Clear loading event - we're replacing with a fully built manager
            event = self._loading_events.pop(cache_key, None)
            if event:
                event.set()
            self._evict_if_needed()

        logger.info(
            "Built and saved sparse index: %s (%d docs)",
            cache_key,
            stats.get("doc_count", 0),
        )

        return stats

    def rebuild_from_scratch(
        self,
        documents: list[str],
        item_ids: list[str],
        item_table: str = "chunks",
    ) -> None:
        """Rebuild this KB's BM25 index from scratch and overwrite the on-disk files.

        Used by the `build_bm25_for_kb` Celery task. Unlike `add_and_save`,
        this does NOT merge with an existing on-disk index — it builds a
        fresh one from the provided corpus and writes it in place.

        Failure mode: bm25s writes multiple files directly; if the worker
        crashes mid-save, the on-disk index may be partially written.
        The metadata.json sidecar (written last, atomically) reflects the
        last successful rebuild's item_count, so a subsequent
        `_compute_bm25_status` call will report 'stale' and prompt the
        user to re-trigger this method. No data corruption — just a
        rebuild round-trip.

        No-op if `documents` is empty.
        """
        if not documents:
            return

        # Heavy work outside the lock — does NOT block other KB lookups
        manager = BM25IndexManager()
        manager.build_index(documents=documents, item_ids=item_ids)
        if not manager.is_empty():
            manager.save(self.get_index_path(item_table))

        # Only the cache eviction needs the lock
        cache_key = f"{self.kb_id}:{item_table}"
        with self._cache_lock:
            self._managers.pop(cache_key, None)
            event = self._loading_events.pop(cache_key, None)
            if event:
                event.set()

        self.write_metadata(item_table=item_table, item_count=len(item_ids))

        logger.info(
            "Rebuilt BM25 index from scratch: kb=%s item_table=%s docs=%d",
            self.kb_id,
            item_table,
            len(item_ids),
        )

    def add_and_save(
        self,
        documents: list[str],
        item_ids: list[str],
        item_table: str = "chunks",
    ) -> None:
        """Incrementally add documents and save.

        Thread-safe: holds lock during modification to prevent concurrent writes.

        Args:
            documents: New document texts to add.
            item_ids: Corresponding item IDs.
            item_table: Table name.
        """
        if not documents:
            return

        cache_key = f"{self.kb_id}:{item_table}"

        # Combined wait-and-operate loop to prevent TOCTOU race
        while True:
            with self._cache_lock:
                # Check if loading in progress
                if cache_key in self._loading_events:
                    event = self._loading_events[cache_key]
                else:
                    # No loading in progress - perform operation NOW while holding lock
                    if cache_key in self._managers:
                        manager = self._managers[cache_key]
                    else:
                        manager = BM25IndexManager()
                        self._managers[cache_key] = manager
                        self._evict_if_needed()

                    # Load from disk if manager is empty and index exists
                    if manager.is_empty() and self.index_exists(item_table):
                        try:
                            manager.load(self.get_index_path(item_table))
                        except Exception as e:
                            logger.warning("Failed to load sparse index %s: %s", cache_key, e)

                    manager.add_documents(documents, item_ids)

                    if not manager.is_empty():
                        manager.save(self.get_index_path(item_table))

                    self._managers.move_to_end(cache_key)
                    break  # Done!

            # Wait outside lock for loading to complete, then retry
            event.wait()

        logger.debug(
            "Added %d docs to sparse index: %s:%s",
            len(documents),
            self.kb_id,
            item_table,
        )

    def remove_and_save(
        self,
        item_ids: list[str],
        item_table: str = "chunks",
    ) -> None:
        """Remove documents by ID and save.

        Thread-safe: holds lock during modification to prevent concurrent writes.

        Args:
            item_ids: IDs of documents to remove.
            item_table: Table name.
        """
        if not item_ids:
            return

        if not self.index_exists(item_table):
            return

        cache_key = f"{self.kb_id}:{item_table}"

        # Combined wait-and-operate loop to prevent TOCTOU race
        while True:
            with self._cache_lock:
                # Check if loading in progress
                if cache_key in self._loading_events:
                    event = self._loading_events[cache_key]
                else:
                    # No loading in progress - perform operation NOW while holding lock
                    if cache_key in self._managers:
                        manager = self._managers[cache_key]
                    else:
                        manager = BM25IndexManager()
                        self._managers[cache_key] = manager
                        self._evict_if_needed()

                    # Load from disk if manager is empty and index exists
                    if manager.is_empty() and self.index_exists(item_table):
                        try:
                            manager.load(self.get_index_path(item_table))
                        except Exception as e:
                            logger.warning("Failed to load sparse index %s: %s", cache_key, e)

                    manager.remove_documents(item_ids)

                    if manager.is_empty():
                        # Delete empty index (also clears cache)
                        self._delete_index_unlocked(item_table)
                    else:
                        manager.save(self.get_index_path(item_table))
                        self._managers.move_to_end(cache_key)
                    break  # Done!

            # Wait outside lock for loading to complete, then retry
            event.wait()

        logger.debug(
            "Removed %d docs from sparse index: %s:%s",
            len(item_ids),
            self.kb_id,
            item_table,
        )

    def _delete_index_unlocked(self, item_table: str) -> None:
        """Delete index files and clear cache (must hold _cache_lock).

        Internal method for use when lock is already held.

        Args:
            item_table: Table name.
        """
        path = self.get_index_path(item_table)

        if os.path.exists(path):
            shutil.rmtree(path)
            logger.info("Deleted sparse index: %s:%s", self.kb_id, item_table)

        cache_key = f"{self.kb_id}:{item_table}"
        self._managers.pop(cache_key, None)
        # Also clear loading event if present, unblocking any waiters
        event = self._loading_events.pop(cache_key, None)
        if event:
            event.set()

    def delete_index(self, item_table: str = "chunks") -> None:
        """Delete index files and clear cache.

        Args:
            item_table: Table name.
        """
        with self._cache_lock:
            self._delete_index_unlocked(item_table)

    def delete_all_indexes(self) -> None:
        """Delete all indexes for this knowledge base."""
        kb_path = os.path.join(self.base_path, self.kb_id)

        if os.path.exists(kb_path):
            shutil.rmtree(kb_path)
            logger.info("Deleted all sparse indexes for KB: %s", self.kb_id)

        # Clear all cache entries for this KB
        with self._cache_lock:
            keys_to_remove = [k for k in self._managers if k.startswith(f"{self.kb_id}:")]
            for key in keys_to_remove:
                self._managers.pop(key, None)
                # Set event before popping to unblock any waiting threads
                event = self._loading_events.pop(key, None)
                if event:
                    event.set()

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached managers (for testing/cleanup)."""
        with cls._cache_lock:
            cls._managers.clear()
            # Set all events before clearing to unblock any waiting threads
            for event in cls._loading_events.values():
                event.set()
            cls._loading_events.clear()
