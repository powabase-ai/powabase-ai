"""Unit tests for the indexed_sources watchdog."""

import json
import uuid
from unittest.mock import MagicMock, patch


def test_watchdog_tick_reschedules_self():
    """The tick task must re-enqueue itself with countdown=300."""
    from agentic_project_service.tasks import watchdog

    fake_redis = MagicMock()
    fake_redis.set.return_value = True  # lock acquired
    fake_redis.delete.return_value = 1
    fake_redis.lrange.return_value = []

    fake_inspect = MagicMock()
    fake_inspect.active.return_value = {"worker": []}
    fake_inspect.reserved.return_value = {"worker": []}

    with (
        patch.object(watchdog, "_get_redis", return_value=(fake_redis, "test:lock")),
        patch.object(watchdog.celery_app.control, "inspect", return_value=fake_inspect),
        patch.object(watchdog.indexed_sources_watchdog_tick, "apply_async") as mock_apply,
    ):
        watchdog.indexed_sources_watchdog_tick()
        mock_apply.assert_called_once()
        assert mock_apply.call_args.kwargs.get("countdown") == 300


def test_watchdog_tick_skips_when_lock_held():
    """If another worker holds the lock, the tick returns immediately."""
    from agentic_project_service.tasks import watchdog

    fake_redis = MagicMock()
    fake_redis.set.return_value = False  # lock NOT acquired
    fake_inspect = MagicMock()

    with (
        patch.object(watchdog, "_get_redis", return_value=(fake_redis, "test:lock")),
        patch.object(watchdog.celery_app.control, "inspect", return_value=fake_inspect),
        patch.object(watchdog.indexed_sources_watchdog_tick, "apply_async") as mock_apply,
    ):
        watchdog.indexed_sources_watchdog_tick()
        # Lock not acquired -> no inspect, but reschedule MUST still happen
        fake_inspect.active.assert_not_called()
        mock_apply.assert_called_once()


def _mock_envelope(task_id: str, task_name: str) -> bytes:
    """Build a Celery message envelope as bytes (what LRANGE returns)."""
    return json.dumps(
        {
            "headers": {"id": task_id, "task": task_name},
            "properties": {"correlation_id": task_id},
            "body": "",
        }
    ).encode()


def test_collect_alive_task_ids_unions_three_sources():
    """active + reserved + LRANGE — only the index_source-named ones count."""
    from agentic_project_service.tasks import watchdog

    fake_redis = MagicMock()
    fake_redis.lrange.return_value = [
        _mock_envelope("queued-1", "agentic_project_service.tasks.indexing.index_source"),
        _mock_envelope("not-ours", "agentic_project_service.tasks.extraction.extract_source"),
        _mock_envelope("queued-2", "agentic_project_service.tasks.indexing.index_source"),
    ]

    fake_inspect = MagicMock()
    fake_inspect.active.return_value = {
        "worker-a": [
            {"id": "active-1", "name": "agentic_project_service.tasks.indexing.index_source"},
            {
                "id": "active-other",
                "name": "agentic_project_service.tasks.scheduler.scheduler_tick",
            },
        ],
    }
    fake_inspect.reserved.return_value = {
        "worker-a": [
            {"id": "reserved-1", "name": "agentic_project_service.tasks.indexing.index_source"},
        ],
    }

    ids, queue_msgs, _, _ = watchdog._collect_alive_task_ids(
        fake_redis, fake_inspect, queue_key="test:celery"
    )
    assert ids == {"queued-1", "queued-2", "active-1", "reserved-1"}
    assert len(queue_msgs) == 3  # LRANGE returned 3 raw messages


def test_collect_alive_task_ids_handles_malformed_envelope():
    """A garbage message in the queue must not crash the watchdog."""
    from agentic_project_service.tasks import watchdog

    fake_redis = MagicMock()
    fake_redis.lrange.return_value = [
        b"not-json",
        _mock_envelope("queued-1", "agentic_project_service.tasks.indexing.index_source"),
        b'{"missing": "headers"}',
    ]
    fake_inspect = MagicMock()
    fake_inspect.active.return_value = {}
    fake_inspect.reserved.return_value = {}

    ids, _, _, _ = watchdog._collect_alive_task_ids(
        fake_redis, fake_inspect, queue_key="test:celery"
    )
    assert ids == {"queued-1"}  # the garbage entries are skipped


def test_collect_alive_task_ids_handles_non_utf8_bytes():
    """Non-UTF-8 bytes in the queue must not crash (UnicodeDecodeError escape path)."""
    from agentic_project_service.tasks import watchdog

    fake_redis = MagicMock()
    fake_redis.lrange.return_value = [
        b"\x80\x81\x82",  # non-UTF-8, triggers UnicodeDecodeError in json.loads
        _mock_envelope("queued-1", "agentic_project_service.tasks.indexing.index_source"),
    ]
    fake_inspect = MagicMock()
    fake_inspect.active.return_value = {}
    fake_inspect.reserved.return_value = {}

    ids, _, _, _ = watchdog._collect_alive_task_ids(
        fake_redis, fake_inspect, queue_key="test:celery"
    )
    assert ids == {"queued-1"}


def test_watchdog_skips_recovery_when_all_silent():
    """If alive_ids is empty AND queue is empty AND inspect returned no workers,
    treat as 'workers unreachable' — skip recovery."""
    from agentic_project_service.tasks import watchdog

    fake_redis = MagicMock()
    fake_redis.set.return_value = True
    fake_redis.delete.return_value = 1
    fake_redis.lrange.return_value = []  # empty queue

    fake_inspect = MagicMock()
    fake_inspect.active.return_value = None  # no worker responded
    fake_inspect.reserved.return_value = None

    with (
        patch.object(watchdog, "_get_redis", return_value=(fake_redis, "k")),
        patch.object(watchdog.celery_app.control, "inspect", return_value=fake_inspect),
        patch.object(watchdog, "_find_and_recover_orphans") as mock_recover,
        patch.object(watchdog.indexed_sources_watchdog_tick, "apply_async"),
    ):
        watchdog.indexed_sources_watchdog_tick()
        mock_recover.assert_not_called()


def test_watchdog_runs_recovery_when_workers_responding_but_idle():
    """Idle workers (responded with empty list) is NOT 'unreachable'.

    Should still attempt recovery to catch indexing-orphans whose celery
    task IDs are stored on the row but absent from every alive set.
    """
    from agentic_project_service.tasks import watchdog

    fake_redis = MagicMock()
    fake_redis.set.return_value = True
    fake_redis.delete.return_value = 1
    fake_redis.lrange.return_value = []  # empty queue

    fake_inspect = MagicMock()
    fake_inspect.active.return_value = {"worker-a": []}  # responded, empty
    fake_inspect.reserved.return_value = {"worker-a": []}

    with (
        patch.object(watchdog, "_get_redis", return_value=(fake_redis, "k")),
        patch.object(watchdog.celery_app.control, "inspect", return_value=fake_inspect),
        patch.object(watchdog, "_find_and_recover_orphans") as mock_recover,
        patch.object(watchdog.indexed_sources_watchdog_tick, "apply_async"),
    ):
        watchdog.indexed_sources_watchdog_tick()
        mock_recover.assert_called_once()


def test_find_and_recover_orphans_dispatches_for_each_orphan():
    """Each row returned by the orphan query gets a fresh index_source.delay()."""
    from agentic_project_service.tasks import watchdog

    # Mock db.session.execute to return 2 orphan rows, then accept the UPDATE.
    orphan_rows = [
        MagicMock(id="row-1", source_id="src-1", knowledge_base_id="kb-1"),
        MagicMock(id="row-2", source_id="src-2", knowledge_base_id="kb-2"),
    ]
    mock_session = MagicMock()
    select_result = MagicMock()
    select_result.fetchall.return_value = orphan_rows
    mock_session.execute.return_value = select_result

    mock_db = MagicMock()
    mock_db.session = mock_session

    mock_delay = MagicMock()
    with (
        patch.object(watchdog, "db", mock_db),
        patch("agentic_project_service.tasks.indexing.index_source.delay", mock_delay),
        patch.object(watchdog, "get_all_user_provider_keys", return_value={}),
    ):
        recovered = watchdog._find_and_recover_orphans(alive_ids={"some-other-id"})

    assert recovered == 2
    assert mock_delay.call_count == 2
    # Each call: index_source.delay(kb_id, source_id, indexed_source_id=..., provider_keys=...)
    for call, row in zip(mock_delay.call_args_list, orphan_rows):
        args, kwargs = call
        assert args == (row.knowledge_base_id, row.source_id)
        assert kwargs["indexed_source_id"] == row.id


def test_find_and_recover_orphans_dispatches_str_args_even_when_row_columns_are_uuid():
    """Regression: dispatch must pass plain ``str`` to ``index_source.delay`` even
    when SQLAlchemy hands back ``uuid.UUID`` column values.

    The other four dispatch sites (``/add_source_to_kb``, ``/reindex`` selective,
    ``/reindex`` failed_only, ``reindex_knowledge_base``) all call ``str(row[...])``
    before ``.delay()``. The watchdog historically did not — it passed
    ``row.source_id`` and ``row.knowledge_base_id`` raw.

    That matters because ``kombu.utils.json`` (Celery's JSON serializer) does NOT
    flatten ``uuid.UUID`` to a string. It encodes a typed marker
    (``{"__type__": "uuid", "__value__": {"hex": "..."}}``) and the worker
    deserializes it back into a ``uuid.UUID`` instance. The UUID then flows
    through ``index_source`` → ``run_indexing`` → ``ChunkAndEmbedAlgorithm.aindex``
    → ``RecursiveChunking.chunk`` → ``TextChunk(source_id=...)``, where Pydantic
    v2 strict-validates ``source_id: str | None`` and raises::

        pydantic_core._pydantic_core.ValidationError: 1 validation error for TextChunk
        source_id
          Input should be a valid string [type=string_type,
          input_value=UUID('...'), input_type=UUID]

    Observed in production on 2026-05-18 after PR #282 unmasked it (the prior
    ``SparseIndexStore`` UUID crash in the cleanup branch was hiding this one).
    """
    from agentic_project_service.tasks import watchdog

    # SQLAlchemy Row attributes are real ``uuid.UUID`` instances for UUID columns.
    kb_uuid = uuid.uuid4()
    src_uuid_1 = uuid.uuid4()
    isid_uuid_1 = uuid.uuid4()
    src_uuid_2 = uuid.uuid4()
    isid_uuid_2 = uuid.uuid4()

    orphan_rows = [
        MagicMock(id=isid_uuid_1, source_id=src_uuid_1, knowledge_base_id=kb_uuid),
        MagicMock(id=isid_uuid_2, source_id=src_uuid_2, knowledge_base_id=kb_uuid),
    ]
    mock_session = MagicMock()
    select_result = MagicMock()
    select_result.fetchall.return_value = orphan_rows
    mock_session.execute.return_value = select_result

    mock_db = MagicMock()
    mock_db.session = mock_session

    mock_delay = MagicMock()
    with (
        patch.object(watchdog, "db", mock_db),
        patch("agentic_project_service.tasks.indexing.index_source.delay", mock_delay),
        patch.object(watchdog, "get_all_user_provider_keys", return_value={}),
    ):
        watchdog._find_and_recover_orphans(alive_ids={"some-other-id"})

    assert mock_delay.call_count == 2
    expected_pairs = [
        (str(kb_uuid), str(src_uuid_1), str(isid_uuid_1)),
        (str(kb_uuid), str(src_uuid_2), str(isid_uuid_2)),
    ]
    for call, (exp_kb, exp_src, exp_isid) in zip(
        mock_delay.call_args_list, expected_pairs, strict=True
    ):
        args, kwargs = call

        # Positional args: kb_id, source_id — must be plain str, not UUID.
        assert len(args) == 2, f"Expected 2 positional args, got {args!r}"
        for i, (label, val, expected) in enumerate(
            [("kb_id", args[0], exp_kb), ("source_id", args[1], exp_src)]
        ):
            assert isinstance(val, str), (
                f"Positional arg #{i} ({label}) must be str (kombu's JSON "
                f"serializer preserves UUID type, which leads to a Pydantic "
                f"ValidationError downstream at TextChunk). "
                f"Got: {type(val).__name__}({val!r})"
            )
            assert val == expected

        # indexed_source_id kwarg must also be str (same reason).
        isid = kwargs.get("indexed_source_id")
        assert isinstance(
            isid, str
        ), f"indexed_source_id kwarg must be str, got: {type(isid).__name__}({isid!r})"
        assert isid == exp_isid


def test_find_and_recover_orphans_uses_correct_sql_filters():
    """The SQL must filter on status, dispatch-race guard, and alive_ids exclusion.

    Scope is limited to ``index_status = 'indexing'`` — pending-rows are
    deliberately excluded (see module docstring). The SQL must therefore
    NOT match pending rows even if their celery_task_id is NULL.
    """
    from agentic_project_service.tasks import watchdog

    mock_session = MagicMock()
    select_result = MagicMock()
    select_result.fetchall.return_value = []
    mock_session.execute.return_value = select_result

    mock_db = MagicMock()
    mock_db.session = mock_session

    with (
        patch.object(watchdog, "db", mock_db),
        patch.object(watchdog, "get_all_user_provider_keys", return_value={}),
    ):
        watchdog._find_and_recover_orphans(alive_ids={"alive-1", "alive-2"})

    # Verify the SELECT was called with the alive_ids and the filter clauses.
    select_calls = [
        c for c in mock_session.execute.call_args_list if "SELECT" in str(c.args[0]).upper()
    ]
    assert len(select_calls) >= 1
    sql, params = select_calls[0].args
    sql_str = str(sql)
    assert "index_status = 'indexing'" in sql_str
    # Must NOT include pending in the scope.
    assert "IN ('pending', 'indexing')" not in sql_str
    assert "last_dispatched_at" in sql_str
    assert "INTERVAL '2 minutes'" in sql_str
    # Pending-orphan branch removed: query must require celery_task_id IS NOT NULL.
    assert "celery_task_id IS NOT NULL" in sql_str
    assert "celery_task_id <> ALL(:alive_ids)" in sql_str
    assert set(params.get("alive_ids", [])) == {"alive-1", "alive-2"}


def test_orphan_query_excludes_pending_rows():
    """Regression: the orphan SELECT must not match rows in ``index_status='pending'``.

    Pending-orphan detection was removed because the previous implementation
    matched on ``celery_task_id IS NULL`` — which is the *normal* state of a
    pending row that's still waiting in the queue (the worker doesn't write
    ``celery_task_id`` until the task starts running). The previous logic
    therefore re-dispatched every pending row >2 min old, causing duplicate
    Celery tasks to race on the same indexed_source and triggering the
    SparseIndexStore UUID crash in the cleanup branch.
    """
    from agentic_project_service.tasks import watchdog

    sql_str = str(watchdog.ORPHAN_QUERY)
    # Must NOT include 'pending' as a target status.
    assert (
        "'pending'" not in sql_str.split("WHERE", 1)[1].split("AND")[0]
    ), "Orphan query must only target index_status='indexing' rows"
    # The status filter must be exactly 'indexing'.
    assert "index_status = 'indexing'" in sql_str
    # And celery_task_id IS NULL must NOT be a recovery trigger.
    assert "celery_task_id IS NULL" not in sql_str
    assert "celery_task_id IS NOT NULL" in sql_str


def test_seed_watchdog_calls_delay_once():
    """The worker_ready hook calls indexed_sources_watchdog_tick.delay() once."""
    from agentic_project_service.tasks import watchdog

    with patch.object(watchdog.indexed_sources_watchdog_tick, "delay") as mock_delay:
        watchdog.seed_watchdog()
        mock_delay.assert_called_once_with()


def test_find_and_recover_orphans_redispatches_with_indexing_key_inputs(monkeypatch):
    """Watchdog re-dispatch MUST thread the indexing key inputs, otherwise the
    recovered orphan completes indexing and never bills — a silent under-charge
    per spec line 132 (retries must remain idempotent AND must charge once).

    The key inputs match the /reindex + batch paths: the literal "indexing"
    namespace + the indexed_source_id. Because they are derived deterministically
    from the indexed_source_id, repeated watchdog ticks for the same row produce
    the same key (idempotent at the billing service via
    UNIQUE(org_id, idempotency_key)), and a watchdog recovery converges with a
    subsequent manual reindex of the same row. Identity (org/project) is added by
    the billing adapter — no verbatim key is threaded here.
    """
    from agentic_project_service.tasks import watchdog

    # These envs are normally injected by the deployment tooling; the watchdog
    # no longer reads them directly — the adapter does — but set them here so
    # the test mirrors a real deployment.
    monkeypatch.setenv("BILLING_ORG_ID", "org-watchdog")
    monkeypatch.setenv("PROJECT_ID", "proj-watchdog")

    orphan_row = MagicMock(
        id="00000000-0000-0000-0000-000000000001",
        source_id="00000000-0000-0000-0000-000000000002",
        knowledge_base_id="00000000-0000-0000-0000-000000000003",
    )
    mock_session = MagicMock()
    select_result = MagicMock()
    select_result.fetchall.return_value = [orphan_row]
    mock_session.execute.return_value = select_result

    mock_db = MagicMock()
    mock_db.session = mock_session

    mock_delay = MagicMock()
    with (
        patch.object(watchdog, "db", mock_db),
        patch("agentic_project_service.tasks.indexing.index_source.delay", mock_delay),
        patch.object(watchdog, "get_all_user_provider_keys", return_value={}),
    ):
        watchdog._find_and_recover_orphans(alive_ids={"some-other-id"})

    assert mock_delay.call_count == 1
    _, kwargs = mock_delay.call_args
    # The key inputs: literal "indexing" namespace + the indexed_source_id (str).
    assert kwargs.get("idempotency_action") == "indexing"
    assert kwargs.get("idempotency_parts") == ["00000000-0000-0000-0000-000000000001"]
    # No verbatim key / identity threaded — the port owns identity.
    assert "billing_idempotency_key" not in kwargs
    assert "billing_org_id" not in kwargs


def test_find_and_recover_orphans_threads_key_inputs_even_without_billing_env(monkeypatch):
    """When the PS pod has no BILLING_ORG_ID (BYOC / local dev), the watchdog
    STILL threads the same key inputs — it no longer branches on billing
    context. The no-charge-when-unconfigured guarantee moved to the billing
    adapter (covered by test_billing_cloud_adapter
    test_charge_returns_noop_outcome_when_billing_unconfigured), so threading
    the key inputs unconditionally is harmless and keeps the dispatch simple."""
    from agentic_project_service.tasks import watchdog

    monkeypatch.delenv("BILLING_ORG_ID", raising=False)
    monkeypatch.delenv("PROJECT_ID", raising=False)
    monkeypatch.delenv("BILLING_PROJECT_ID", raising=False)

    orphan_row = MagicMock(
        id="row-1",
        source_id="src-1",
        knowledge_base_id="kb-1",
    )
    mock_session = MagicMock()
    select_result = MagicMock()
    select_result.fetchall.return_value = [orphan_row]
    mock_session.execute.return_value = select_result

    mock_db = MagicMock()
    mock_db.session = mock_session

    mock_delay = MagicMock()
    with (
        patch.object(watchdog, "db", mock_db),
        patch("agentic_project_service.tasks.indexing.index_source.delay", mock_delay),
        patch.object(watchdog, "get_all_user_provider_keys", return_value={}),
    ):
        watchdog._find_and_recover_orphans(alive_ids=set())

    _, kwargs = mock_delay.call_args
    # Env-independent: same key inputs as the org-configured case.
    assert kwargs.get("idempotency_action") == "indexing"
    assert kwargs.get("idempotency_parts") == ["row-1"]


def test_find_and_recover_orphans_skips_rows_that_finished_between_select_and_update():
    """If a row's status changes from 'indexing' to 'indexed' between SELECT and UPDATE,
    the UPDATE must NOT clobber it back to 'pending'.

    We can't simulate the race directly in a unit test, but we can verify the SQL
    UPDATE statement includes the status filter that protects against it."""
    from agentic_project_service.tasks import watchdog

    captured_sql = []

    mock_session = MagicMock()
    select_result = MagicMock()
    select_result.fetchall.return_value = [
        MagicMock(id="row-1", source_id="src-1", knowledge_base_id="kb-1"),
    ]

    def capture_execute(sql, params=None):
        captured_sql.append(str(sql))
        return select_result

    mock_session.execute.side_effect = capture_execute

    mock_db = MagicMock()
    mock_db.session = mock_session

    with (
        patch.object(watchdog, "db", mock_db),
        patch("agentic_project_service.tasks.indexing.index_source.delay"),
        patch.object(watchdog, "get_all_user_provider_keys", return_value={}),
    ):
        watchdog._find_and_recover_orphans(alive_ids=set())

    update_sql = next((s for s in captured_sql if "UPDATE" in s.upper() and "ai" in s), None)
    assert update_sql is not None, "Expected an UPDATE statement to be issued"
    assert (
        "index_status = 'indexing'" in update_sql
    ), f"UPDATE must filter by current status to prevent TOCTOU clobber, got: {update_sql}"
