"""The storage tools cannot leave the bucket the model names, nor name an internal one.

``bucket`` and ``path`` come from the model. Before these checks they went into
the object URL as-is, and httpx removes dot segments, so ``../sources/...`` or a
bucket of ``x/../sources`` reached the internal sources bucket with the service
key. The tests drive the handlers through the real ``SupabaseStorage`` client
and record the URL httpx would send.
"""

import json
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agentic_project_service.services import storage as storage_mod
from agentic_project_service.services.storage import StorageError, SupabaseStorage
from agentic_project_service.services.tool_caller import ToolCaller
from agentic_project_service.tools import builtin
from agentic_project_service.tools.builtin import storage_read_handler, storage_write_handler

SERVICE = ToolCaller.service()
USER = ToolCaller(
    claims={"sub": "user-1", "role": "authenticated", "exp": time.time() + 3600},
    token="user-jwt",
)


class _Recorder:
    """Stands in for httpx.request / httpx.stream and records each final URL."""

    def __init__(self):
        self.urls: list[httpx.URL] = []
        self.clients = 0  # storage clients the handler asked for
        self.binary = False  # download bodies that are not UTF-8

    def _response(self, method, url):
        request = httpx.Request(method, url)
        self.urls.append(request.url)
        if "/object/list/" in request.url.path:
            return httpx.Response(200, json=[{"name": "q3.csv"}], request=request)
        if method == "POST" and "/object/sign/" in request.url.path:
            return httpx.Response(
                200, json={"signedURL": "/object/sign/x?token=t"}, request=request
            )
        if method == "POST":
            return httpx.Response(200, json={"Key": "k"}, request=request)
        if self.binary:
            return httpx.Response(200, content=bytes([0xFF, 0xFE]), request=request)
        return httpx.Response(200, content=b"a,b\n1,2\n", request=request)

    def request(self, method, url, **kwargs):
        return self._response(method, url)

    def paths(self) -> list[str]:
        return [u.raw_path.decode() for u in self.urls]


@pytest.fixture
def real_storage(monkeypatch):
    """Both storage factories return a real client whose HTTP calls are recorded."""
    monkeypatch.delenv("STORAGE_URL", raising=False)
    recorder = _Recorder()
    monkeypatch.setattr(storage_mod.httpx, "request", recorder.request)

    def client(bearer=None):
        recorder.clients += 1
        return SupabaseStorage(url="http://storage.test", service_key="service-key", bearer=bearer)

    monkeypatch.setattr(builtin, "get_storage", client)
    monkeypatch.setattr(builtin, "get_storage_for_user", client)
    return recorder


def _read(caller, operation, bucket, path):
    return json.loads(
        storage_read_handler(
            {"operation": operation, "bucket": bucket, "path": path, "_caller": caller}, None
        )
    )


def _write(caller, bucket, path):
    return json.loads(
        storage_write_handler(
            {"bucket": bucket, "path": path, "content": "x", "_caller": caller}, None
        )
    )


# ---------------------------------------------------------------------------
# Bucket and path forms that would name another bucket, or an internal one
# ---------------------------------------------------------------------------


REJECTED_CALLS = [
    # (id, callable(caller) -> result)
    (
        "download path ../sources",
        lambda c: _read(c, "download", "public", "../sources/p/s1/original.pdf"),
    ),
    (
        "download bucket x/../sources",
        lambda c: _read(c, "download", "x/../sources", "p/s1/original.pdf"),
    ),
    (
        "download bucket sources/kb-123",
        lambda c: _read(c, "download", "sources/kb-123", "original.pdf"),
    ),
    ("write bucket sources/kb-123", lambda c: _write(c, "sources/kb-123", "injected.txt")),
    ("write path ../sources", lambda c: _write(c, "public", "../sources/p/injected.txt")),
    ("list bucket x/../sources", lambda c: _read(c, "list", "x/../sources", "")),
    ("list prefix ../sources", lambda c: _read(c, "list", "public", "../sources/")),
    ("list prefix a/../../sources", lambda c: _read(c, "list", "public", "a/../../sources/")),
    ("download bucket %73ources", lambda c: _read(c, "download", "%73ources", "a.pdf")),
    ("write bucket %73ources", lambda c: _write(c, "%73ources", "a.txt")),
    ("download bucket sources", lambda c: _read(c, "download", "sources", "a.pdf")),
    ("download bucket Sources", lambda c: _read(c, "download", "Sources", "a.pdf")),
    ("download bucket ..", lambda c: _read(c, "download", "..", "sources/a.pdf")),
    ("download bucket .", lambda c: _read(c, "download", ".", "a.pdf")),
    ("download bucket with backslash", lambda c: _read(c, "download", "x\\..\\sources", "a.pdf")),
    ("download bucket with space", lambda c: _read(c, "download", "my bucket", "a.pdf")),
    # "$" in a pattern also matches before a trailing newline.
    ("download bucket sources\\n", lambda c: _read(c, "download", "sources\n", "a.pdf")),
    ("list bucket sources\\n", lambda c: _read(c, "list", "sources\n", "")),
    ("write bucket sources\\n", lambda c: _write(c, "sources\n", "a.txt")),
    ("download bucket docs\\n", lambda c: _read(c, "download", "docs\n", "a.pdf")),
    ("download path %2e%2e", lambda c: _read(c, "download", "public", "%2e%2e/sources/a.pdf")),
    ("download path ./", lambda c: _read(c, "download", "public", "./a.pdf")),
    ("download path a/./b", lambda c: _read(c, "download", "public", "a/./b.pdf")),
    ("download path backslash ..", lambda c: _read(c, "download", "public", "..\\sources\\a.pdf")),
    ("download path ?", lambda c: _read(c, "download", "public", "a.pdf?x=../../sources")),
    ("download path ? alone", lambda c: _read(c, "download", "public", "report?v=2.csv")),
    ("write path ? alone", lambda c: _write(c, "public", "report?v=2.csv")),
    ("download path #", lambda c: _read(c, "download", "public", "a.pdf#frag")),
    ("download path newline", lambda c: _read(c, "download", "public", "a\n.pdf")),
    ("download path NUL", lambda c: _read(c, "download", "public", "a\x00.pdf")),
    ("download path leading slash", lambda c: _read(c, "download", "public", "/a.pdf")),
    ("download path empty segment", lambda c: _read(c, "download", "public", "a//b.pdf")),
    ("download path trailing slash", lambda c: _read(c, "download", "public", "reports/")),
    ("write path trailing slash", lambda c: _write(c, "public", "reports/")),
    ("write path %", lambda c: _write(c, "public", "a%2fb.txt")),
    ("list prefix empty segment", lambda c: _read(c, "list", "public", "a//")),
    ("list prefix leading slash", lambda c: _read(c, "list", "public", "/reports/")),
    ("list prefix %", lambda c: _read(c, "list", "public", "re%")),
]


@pytest.mark.parametrize("caller", [SERVICE, USER], ids=["service", "end-user"])
@pytest.mark.parametrize("call", [c for _, c in REJECTED_CALLS], ids=[i for i, _ in REJECTED_CALLS])
def test_the_form_is_refused_before_any_request(real_storage, caller, call):
    result = call(caller)
    assert "error" in result
    # Refused by the tool's own check, not by something failing further down.
    assert real_storage.clients == 0
    assert real_storage.urls == []


@pytest.mark.parametrize("bucket", [None, 7, ["public"]])
def test_a_non_string_bucket_is_refused(real_storage, bucket):
    assert "error" in _read(SERVICE, "download", bucket, "a.pdf")
    assert real_storage.clients == 0


@pytest.mark.parametrize("path", [7, ["a.pdf"], {"p": 1}])
def test_a_non_string_path_is_refused(real_storage, path):
    assert "error" in _read(SERVICE, "download", "public", path)
    assert "error" in _write(SERVICE, "public", path)
    assert real_storage.clients == 0


# ---------------------------------------------------------------------------
# Positive controls: ordinary buckets and nested paths still work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("caller", [SERVICE, USER], ids=["service", "end-user"])
class TestOrdinaryPathsWork:
    def test_list_a_nested_prefix(self, real_storage, caller):
        result = _read(caller, "list", "reports_2026-q3", "reports/2026/")
        assert result["objects"] == [{"name": "q3.csv"}]
        assert real_storage.paths() == ["/storage/v1/object/list/reports_2026-q3"]

    def test_list_the_bucket_root(self, real_storage, caller):
        result = _read(caller, "list", "public", "")
        assert result["objects"] == [{"name": "q3.csv"}]

    def test_list_a_prefix_without_trailing_slash(self, real_storage, caller):
        assert "objects" in _read(caller, "list", "public", "reports/2026")

    def test_download_a_nested_path(self, real_storage, caller):
        result = _read(caller, "download", "public", "reports/2026/q3.csv")
        assert result["content"] == "a,b\n1,2\n"
        assert result["path"] == "reports/2026/q3.csv"
        assert real_storage.paths() == ["/storage/v1/object/public/reports/2026/q3.csv"]

    def test_write_a_nested_path(self, real_storage, caller):
        result = _write(caller, "public", "reports/2026/q3.csv")
        assert result["path"] == "public/reports/2026/q3.csv"
        assert real_storage.paths() == ["/storage/v1/object/public/reports/2026/q3.csv"]

    def test_names_with_spaces_and_unicode_reach_the_same_object(self, real_storage, caller):
        result = _write(caller, "public", "Q3 résumé (final).txt")
        # The model is told the name it gave, not the encoded form.
        assert result["path"] == "public/Q3 résumé (final).txt"
        (url,) = real_storage.urls
        assert url.path == "/storage/v1/object/public/Q3 résumé (final).txt"

    def test_a_dotted_file_name_is_not_a_dot_segment(self, real_storage, caller):
        assert "content" in _read(caller, "download", "public", "a/..b/.hidden/v1.2..csv")


# ---------------------------------------------------------------------------
# Punctuation in file names reaches Storage as written
# ---------------------------------------------------------------------------

# storage-api signs the object key exactly as the request path spells it, so a
# signed URL for a key whose sub-delimiters went out percent-encoded does not
# verify when it is fetched. These characters are sent as they are, the way
# httpx and the Supabase clients send them; only what could be read as URL
# syntax (a space, a non-ASCII letter, ";") is encoded.
PUNCTUATED_NAMES = [
    ("Smith & Co.pdf", "Smith%20&%20Co.pdf"),
    ("Q1+Q2.png", "Q1+Q2.png"),
    ("a,b.pdf", "a,b.pdf"),
    ("x=y:z@w$.txt", "x=y:z@w$.txt"),
    ("it's (final)!*.txt", "it's%20(final)!*.txt"),
    ("a;b.txt", "a%3Bb.txt"),
]


@pytest.mark.parametrize("caller", [SERVICE, USER], ids=["service", "end-user"])
@pytest.mark.parametrize(("name", "sent"), PUNCTUATED_NAMES, ids=[n for n, _ in PUNCTUATED_NAMES])
class TestPunctuatedNames:
    def test_download(self, real_storage, caller, name, sent):
        result = _read(caller, "download", "public", f"reports/{name}")
        assert result["path"] == f"reports/{name}"
        assert real_storage.paths() == [f"/storage/v1/object/public/reports/{sent}"]

    def test_signed_url_for_a_binary_file(self, real_storage, caller, name, sent):
        real_storage.binary = True
        result = _read(caller, "download", "public", f"reports/{name}")
        assert result["encoding"] == "binary"
        assert real_storage.paths() == [
            f"/storage/v1/object/public/reports/{sent}",
            f"/storage/v1/object/sign/public/reports/{sent}",
        ]

    def test_write(self, real_storage, caller, name, sent):
        result = _write(caller, "public", f"reports/{name}")
        assert result["path"] == f"public/reports/{name}"
        assert real_storage.paths() == [f"/storage/v1/object/public/reports/{sent}"]


@pytest.mark.parametrize("name", ["Smith & Co%20.pdf", "Q1+Q2?.png", "a,b#.pdf", "a=b\\c.txt"])
def test_escapes_and_url_syntax_are_still_refused(real_storage, name):
    assert "Invalid path" in _read(SERVICE, "download", "public", name)["error"]
    assert "Invalid path" in _write(SERVICE, "public", name)["error"]
    assert real_storage.urls == []


# ---------------------------------------------------------------------------
# The storage client itself: defense in depth for every caller
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("STORAGE_URL", raising=False)
    recorder = _Recorder()
    monkeypatch.setattr(storage_mod.httpx, "request", recorder.request)
    storage = SupabaseStorage(url="http://storage.test", service_key="service-key")
    return storage, recorder


class TestStorageClientUrls:
    @pytest.mark.parametrize(
        "call",
        [
            lambda s: s.download("x/../sources", "a.pdf"),
            lambda s: s.download("..", "sources/a.pdf"),
            lambda s: s.download(".", "a.pdf"),
            lambda s: s.download("", "a.pdf"),
            lambda s: s.download("public", "../sources/a.pdf"),
            lambda s: s.download("public", "a/../../sources/a.pdf"),
            lambda s: s.download("public", "./a.pdf"),
            lambda s: s.download_from_path("public/../sources/a.pdf"),
            lambda s: s.upload("public", "../sources/a.txt", b"x"),
            lambda s: s.upload("sources/../x", "a.txt", b"x"),
            lambda s: s.create_signed_url("public", "../sources/a.pdf"),
            lambda s: s.object_size("public/../sources/a.pdf"),
            lambda s: s.delete("x/../sources", ["a.pdf"]),
            lambda s: s.get_bucket("x/../sources"),
        ],
    )
    def test_a_dot_segment_or_slash_in_the_bucket_never_goes_out(self, client, call):
        storage, recorder = client
        with pytest.raises(StorageError):
            call(storage)
        assert recorder.urls == []

    def test_streaming_download_refuses_a_dot_segment(self, client, monkeypatch):
        storage, _ = client
        stream = MagicMock()
        monkeypatch.setattr(storage_mod.httpx, "stream", stream)
        with pytest.raises(StorageError):
            next(storage.stream_download("public", "../sources/a.pdf"))
        stream.assert_not_called()

    def test_the_bucket_is_percent_encoded(self, client):
        storage, recorder = client
        storage.download("%73ources", "a.pdf")
        assert recorder.paths() == ["/storage/v1/object/%2573ources/a.pdf"]

    def test_an_ordinary_path_is_unchanged(self, client):
        storage, recorder = client
        storage.download("sources", "src-1/original/report_ab12cd34.pdf")
        storage.upload("sources", "src-1/derivatives/image/image_page1.png", b"x")
        assert recorder.paths() == [
            "/storage/v1/object/sources/src-1/original/report_ab12cd34.pdf",
            "/storage/v1/object/sources/src-1/derivatives/image/image_page1.png",
        ]

    def test_existing_object_keys_keep_their_request_form(self, client):
        """Paths are not re-encoded here: keys stored or passed by other callers
        (which may already hold an escape such as %20) reach the same object as
        before. The agent tools encode their own paths."""
        storage, recorder = client
        storage.download("user-uploads", "my file.pdf")
        storage.download("user-uploads", "pre%20encoded.pdf")
        assert recorder.paths() == [
            "/storage/v1/object/user-uploads/my%20file.pdf",
            "/storage/v1/object/user-uploads/pre%20encoded.pdf",
        ]


def test_the_handlers_list_request_names_only_the_validated_bucket(real_storage):
    _read(SERVICE, "list", "public", "reports/")
    assert real_storage.paths() == ["/storage/v1/object/list/public"]


def test_signed_url_for_binary_uses_the_encoded_path(monkeypatch):
    storage = MagicMock()
    storage.download_from_path.return_value = bytes([0xFF, 0xFE])
    storage.create_signed_url.return_value = "https://example.com/signed"
    with patch.object(builtin, "get_storage", return_value=storage):
        result = _read(SERVICE, "download", "public", "img dir/a b.png")
    storage.download_from_path.assert_called_once_with("public/img%20dir/a%20b.png")
    storage.create_signed_url.assert_called_once_with("public", "img%20dir/a%20b.png")
    assert result["path"] == "img dir/a b.png"
