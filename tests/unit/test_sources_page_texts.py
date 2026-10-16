"""GET /api/sources/<id>/page-texts — the all-pages form.

Each page is its own storage object, fetched with one HTTP request. A long
document (1,500+ pages at ~0.1 s per request) took minutes when fetched one
after another, past the gateway's 60 s upstream timeout, so the call failed
every time. The downloads must overlap, and the response must keep page order.
"""

import threading
import time
from unittest.mock import MagicMock, patch

from flask import Flask

from agentic_project_service.routes import sources as sources_route


def _client():
    app = Flask(__name__)
    app.register_blueprint(sources_route.sources_bp)
    return app.test_client()


def _session_with_pages(n):
    # Stored out of order: the route sorts by page.
    derivs = [{"page": p, "storage_path": f"sources/src-1/page_{p}.md"} for p in range(n, 0, -1)]
    result = MagicMock()
    result.fetchone.return_value = ({"page_text": derivs},)
    session = MagicMock()
    session.execute.return_value = result
    return session


class _SlowStorage:
    """Fake storage: each download takes ``delay`` s and records concurrency."""

    def __init__(self, delay, fail_pages=()):
        self.delay = delay
        self.fail_pages = set(fail_pages)
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def download(self, bucket_id, path):
        page = int(path.rsplit("_", 1)[1].split(".")[0])
        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay)
            if page in self.fail_pages:
                raise RuntimeError("storage down")
            return f"text of page {page}".encode()
        finally:
            with self.lock:
                self.in_flight -= 1


def _get(storage, n):
    with (
        patch.object(sources_route.db, "session", _session_with_pages(n)),
        patch.object(sources_route, "get_storage", return_value=storage),
        patch(
            "agentic_project_service.auth.decode_jwt",
            return_value={"sub": "svc", "role": "service_role", "is_service_role": True},
        ),
    ):
        return _client().get(
            "/api/sources/src-1/page-texts", headers={"Authorization": "Bearer fake"}
        )


def test_all_pages_download_concurrently_in_page_order():
    storage = _SlowStorage(delay=0.05)
    started = time.monotonic()
    resp = _get(storage, 40)
    elapsed = time.monotonic() - started

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["count"] == 40
    assert body["page_texts"] == [f"text of page {p}" for p in range(1, 41)]
    assert storage.max_in_flight > 1
    # Sequential would be 40 x 0.05 = 2.0 s.
    assert elapsed < 1.0, f"took {elapsed:.2f}s"


def test_failed_page_is_skipped_and_the_rest_keep_order():
    storage = _SlowStorage(delay=0.0, fail_pages={3})
    body = _get(storage, 5).get_json()

    assert body["page_texts"] == [f"text of page {p}" for p in (1, 2, 4, 5)]
    assert body["count"] == 4
