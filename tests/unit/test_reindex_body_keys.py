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


def _send(data, content_type):
    session = MagicMock()
    session.execute.return_value.scalar.return_value = 0  # an empty knowledge base
    with (
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "user-1", "role": "service_role", "is_service_role": True},
        ),
        patch.object(kb_route.db, "session", session),
        patch.object(kb_route, "reindex_knowledge_base") as task,
        patch.object(kb_route, "get_all_user_provider_keys", return_value={}),
    ):
        task.delay.return_value = MagicMock(id="task-1")
        with _app().test_client() as client:
            resp = client.post(
                f"/api/knowledge-bases/{uuid.uuid4()}/reindex",
                data=data,
                content_type=content_type,
                headers={"Authorization": "Bearer x"},
            )
    return resp, session, task


@pytest.mark.parametrize(
    "data,content_type",
    [
        # curl -d without -H 'Content-Type: application/json'
        (
            '{"indexed_source_ids": ["00000000-0000-0000-0000-000000000001"]}',
            "application/x-www-form-urlencoded",
        ),
        ('{"indexed_source_ids": ["00000000-0000-0000-0000-000000000001"]}', "text/plain"),
        ("not json", "application/json"),
        ("[1]", "application/json"),
        ('"x"', "application/json"),
        ('{"indexed_source_ids": []}', "application/json"),
        ('{"indexed_source_ids": "00000000-0000-0000-0000-000000000001"}', "application/json"),
    ],
)
def test_a_body_that_does_not_name_what_to_reindex_is_refused(data, content_type):
    """Each of these used to be read as an empty body (or raised a 500): the
    whole knowledge base reset, re-embedded and billed."""
    resp, session, task = _send(data, content_type)

    assert resp.status_code == 400, resp.get_json()
    session.execute.assert_not_called()
    task.delay.assert_not_called()


@pytest.mark.parametrize(
    "data,content_type", [(None, None), ("", "application/json"), ("{}", "application/json")]
)
def test_no_body_still_reindexes_everything(data, content_type):
    """Callers rely on it: no body (or an empty JSON object) is the whole KB."""
    resp, _, task = _send(data, content_type)

    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()["scope"] == "all"
    task.delay.assert_called_once()
