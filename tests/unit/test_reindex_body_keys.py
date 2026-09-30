"""POST /api/knowledge-bases/<kb_id>/reindex refuses a body key it does not read.

An empty body means "reindex every source in the knowledge base", and each
source is re-embedded and billed. A body whose only key the route does not
know (a misspelling, or a key another route takes) used to be read as that
empty body, turning a request meant to touch a few sources into a full,
billed reindex.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from agentic_project_service.routes import knowledge_bases as kb_route


def _app():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(kb_route.knowledge_bases_bp)
    return app


def _post(body):
    session = MagicMock()
    with (
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "service_role", "is_service_role": True},
        ),
        patch.object(kb_route.db, "session", session),
        patch.object(kb_route, "reindex_knowledge_base") as task,
    ):
        with _app().test_client() as client:
            resp = client.post(
                f"/api/knowledge-bases/{uuid.uuid4()}/reindex",
                json=body,
                headers={"Authorization": "Bearer x"},
            )
    return resp, session, task


@pytest.mark.parametrize("body", [{"retry_failed": True}, {"failed_only": True, "force": 1}])
def test_an_unknown_key_is_refused_before_anything_is_reset(body):
    resp, session, task = _post(body)

    assert resp.status_code == 400
    error = resp.get_json()["error"]
    assert "indexed_source_ids" in error and "failed_only" in error
    session.execute.assert_not_called()
    task.delay.assert_not_called()


def test_the_known_keys_are_read():
    """A malformed id is reported as such, not as an unknown key."""
    resp, _, _ = _post({"indexed_source_ids": ["not-a-uuid"], "failed_only": False})

    assert (resp.status_code, resp.get_json()) == (404, {"error": "Invalid indexed_source_id"})
