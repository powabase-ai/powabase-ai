"""Tests for _finish_run_in_background (module-level function in routes/agents.py).

Also pins the billing behavior of this module's background charge site: on
successful completion it posts an ``agent_run`` charge through the billing
port (services/billing_port.py) via the ``recording_billing`` fixture
(tests/conftest.py + tests/support/billing.py).
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestFinishRunInBackground:
    def _make_generator(self, chunks, output=None, error=None):
        """Create a generator that yields chunks then returns output via StopIteration."""

        def gen():
            for chunk in chunks:
                yield chunk
            if error:
                raise error
            return output

        return gen()

    def test_successful_completion(self, recording_billing):
        from agentic_project_service.routes.agents import _finish_run_in_background

        output = SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            error=None,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            messages=[],
            tool_calls=[],
            reasoning_artifact=None,
        )
        llm_gen = self._make_generator(["Hello ", "world"], output=output)

        flask_app = MagicMock()
        flask_app.app_context.return_value.__enter__ = MagicMock(return_value=None)
        flask_app.app_context.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("agentic_project_service.routes.agents.update_agent_run") as mock_update,
            patch("agentic_project_service.routes.agents.db") as mock_db,
        ):
            _finish_run_in_background(
                flask_app=flask_app,
                run_id="run_abc123",
                llm_gen=llm_gen,
                content_chunks=["Prior "],
                message="test",
                query_enrichment=None,
                retrieved_context_for_db=None,
                context_handler_id=None,
                started_at=datetime.now(UTC),
                reasoning_requested=False,
            )

            mock_update.assert_called_once()
            call_kwargs = mock_update.call_args[1]
            assert call_kwargs["run_id"] == "run_abc123"
            assert call_kwargs["status"].value == "completed"
            assert call_kwargs["content"] == "Prior Hello world"
            assert call_kwargs["error"] is None
            mock_db.session.commit.assert_called_once()

        # Billing: a completed background finish posts exactly one agent_run
        # charge through the port, keyed on run_id (not a fresh uuid4) so a
        # foreground charge that already fired before disconnect dedupes
        # against it (spec line 132).
        assert len(recording_billing.charges) == 1
        charge = recording_billing.charges[0]
        assert charge["action"] == "agent_run"
        assert charge["ref_type"] == "agent_run"
        assert charge["ref_id"] == "run_abc123"
        assert charge["idempotency_parts"] == ("run_abc123",)
        assert charge["metadata"] == {
            "streaming": True,
            "react_loop": False,
            "finished_in_background": True,
        }

    def test_generator_exception_marks_failed(self, recording_billing):
        from agentic_project_service.routes.agents import _finish_run_in_background

        llm_gen = self._make_generator([], error=RuntimeError("LLM broke"))

        flask_app = MagicMock()
        flask_app.app_context.return_value.__enter__ = MagicMock(return_value=None)
        flask_app.app_context.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("agentic_project_service.routes.agents.update_agent_run") as mock_update,
            patch("agentic_project_service.routes.agents.db"),
        ):
            _finish_run_in_background(
                flask_app=flask_app,
                run_id="run_fail",
                llm_gen=llm_gen,
                content_chunks=[],
                message="test",
                query_enrichment=None,
                retrieved_context_for_db=None,
                context_handler_id=None,
                started_at=datetime.now(UTC),
                reasoning_requested=False,
            )

            call_kwargs = mock_update.call_args[1]
            assert call_kwargs["status"].value == "failed"
            assert "LLM broke" in call_kwargs["error"]

        # Failed runs are not charged.
        assert recording_billing.charges == []

    def test_outer_exception_logged(self, recording_billing):
        """If update_agent_run itself raises, no unhandled exception escapes."""
        from agentic_project_service.routes.agents import _finish_run_in_background

        output = SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            error=None,
            usage=None,
            messages=[],
            tool_calls=[],
            reasoning_artifact=None,
        )
        llm_gen = self._make_generator(["ok"], output=output)

        flask_app = MagicMock()
        flask_app.app_context.return_value.__enter__ = MagicMock(return_value=None)
        flask_app.app_context.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "agentic_project_service.routes.agents.update_agent_run",
                side_effect=RuntimeError("DB down"),
            ),
            patch("agentic_project_service.routes.agents.db"),
        ):
            # Should not raise
            _finish_run_in_background(
                flask_app=flask_app,
                run_id="run_crash",
                llm_gen=llm_gen,
                content_chunks=[],
                message="test",
                query_enrichment=None,
                retrieved_context_for_db=None,
                context_handler_id=None,
                started_at=datetime.now(UTC),
                reasoning_requested=False,
            )

        # update_agent_run raised before the charge block ever ran.
        assert recording_billing.charges == []

    def test_run_id_bound_in_real_thread_for_post_charge(self, recording_billing):
        """Exercises the actual `threading.Thread` path: spawn the function in
        a real worker thread (mirroring the production call site in
        run_agent_stream) and assert that the inner billing.charge() call
        sees the bound run_id rather than uuid4 fallback.

        Regression guard: the prior version of this test invoked
        _finish_run_in_background directly on the test thread, so the inner
        ``set_run_id(run_id)`` could be silently deleted and the test would
        still pass (contextvars are local to the test thread anyway). This
        version runs the function in a freshly-spawned daemon Thread, just
        like routes/agents.py:_run_in_background. If the set_run_id call
        inside _finish_run_in_background is removed, the spy below sees
        ``None`` for the captured run_id.
        """
        import threading

        from agentic_project_service.routes.agents import _finish_run_in_background
        from agentic_project_service.services import run_context

        output = SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            error=None,
            usage=None,
            messages=[],
            tool_calls=[],
            reasoning_artifact=None,
        )
        llm_gen = self._make_generator(["ok"], output=output)

        flask_app = MagicMock()
        flask_app.app_context.return_value.__enter__ = MagicMock(return_value=None)
        flask_app.app_context.return_value.__exit__ = MagicMock(return_value=False)

        captured_run_id: list[str | None] = []

        def _spy_update(*_args, **_kwargs):
            # Capture the run_id visible to the worker thread at the moment
            # update_agent_run is invoked — proxy for the inner
            # billing.charge() call that follows it.
            captured_run_id.append(run_context.get_run_id())

        # Pre-condition: the test thread's contextvar must NOT have any
        # run_id bound. If it did, the worker thread could see the right id
        # by accident under copy_context paths (raw Threads don't propagate,
        # but tightening to `is None` rules out any prior-test leak).
        assert run_context.get_run_id() is None

        with (
            patch(
                "agentic_project_service.routes.agents.update_agent_run",
                side_effect=_spy_update,
            ),
            patch("agentic_project_service.routes.agents.db"),
        ):
            worker = threading.Thread(
                target=_finish_run_in_background,
                kwargs=dict(
                    flask_app=flask_app,
                    run_id="run_thread_real",
                    llm_gen=llm_gen,
                    content_chunks=[],
                    message="test",
                    query_enrichment=None,
                    retrieved_context_for_db=None,
                    context_handler_id=None,
                    started_at=datetime.now(UTC),
                    reasoning_requested=False,
                ),
                daemon=True,
            )
            worker.start()
            worker.join(timeout=5)
            assert not worker.is_alive(), "worker did not finish in 5s"

        assert captured_run_id == [
            "run_thread_real"
        ], f"worker thread did not bind run_id; got {captured_run_id}"

        # Direct check on the real charge (not just the get_run_id() proxy
        # above): the worker thread's billing.charge() used the bound
        # run_id, not a uuid4 fallback.
        assert len(recording_billing.charges) == 1
        assert recording_billing.charges[0]["idempotency_parts"] == ("run_thread_real",)
