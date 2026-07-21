"""Supabase Storage service for the project service.

Handles file uploads, downloads, and bucket management via the project's
Supabase Storage API.
"""

import hashlib
import logging
import os
import re

from anyascii import anyascii
import httpx

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Raised when a storage operation fails."""

    pass


class SupabaseStorage:
    """Client for Supabase Storage API."""

    def __init__(
        self,
        url: str | None = None,
        service_key: str | None = None,
        public_url: str | None = None,
    ):
        """Initialize the storage client."""
        self.service_key = service_key or os.getenv("SERVICE_ROLE_KEY")

        if not self.service_key:
            raise ValueError("SERVICE_ROLE_KEY is required for storage operations")

        # STORAGE_URL bypasses Kong for direct storage-api access (K8s mode)
        storage_url_direct = os.getenv("STORAGE_URL")
        if storage_url_direct:
            self.url = storage_url_direct.rstrip("/")
            self.storage_url = self.url  # Already points to storage-api root
        else:
            self.url = (url or os.getenv("SUPABASE_URL", "http://kong:8000")).rstrip("/")
            self.storage_url = f"{self.url}/storage/v1"
        self.headers = {
            "Authorization": f"Bearer {self.service_key}",
            "apikey": self.service_key,
        }

        # Public URL for externally-reachable signed URLs (e.g. for LLM providers)
        _public = public_url or os.getenv("STORAGE_PUBLIC_URL", "")
        self.public_storage_url: str | None = (
            f"{_public.rstrip('/')}/storage/v1" if _public else None
        )

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make a request to the Storage API."""
        url = f"{self.storage_url}{path}"
        headers = {**self.headers, **kwargs.pop("headers", {})}

        try:
            response = httpx.request(method, url, headers=headers, timeout=30, **kwargs)
            return response
        except httpx.RequestError as e:
            logger.error(f"Storage request failed: {e}")
            raise StorageError(f"Storage request failed: {e}") from e

    def list_buckets(self) -> list[dict]:
        """List all buckets."""
        response = self._request("GET", "/bucket")
        if response.status_code != 200:
            raise StorageError(f"Failed to list buckets: {response.text}")
        return response.json()

    def get_bucket(self, bucket_id: str) -> dict | None:
        """Get bucket info, or None if not found."""
        response = self._request("GET", f"/bucket/{bucket_id}")
        if response.status_code == 404:
            return None
        if response.status_code == 400:
            try:
                data = response.json()
                if (
                    data.get("statusCode") == "404"
                    or "not found" in data.get("message", "").lower()
                ):
                    return None
            except Exception:
                pass
            raise StorageError(f"Failed to get bucket: {response.text}")
        if response.status_code != 200:
            raise StorageError(f"Failed to get bucket: {response.text}")
        return response.json()

    def create_bucket(
        self,
        bucket_id: str,
        public: bool = False,
        file_size_limit: int | None = None,
        allowed_mime_types: list[str] | None = None,
    ) -> dict:
        """Create a new bucket."""
        payload = {"id": bucket_id, "name": bucket_id, "public": public}
        if file_size_limit:
            payload["file_size_limit"] = file_size_limit
        if allowed_mime_types:
            payload["allowed_mime_types"] = allowed_mime_types

        response = self._request("POST", "/bucket", json=payload)

        if response.status_code == 400 and "already exists" in response.text.lower():
            return self.get_bucket(bucket_id)

        if response.status_code not in (200, 201):
            raise StorageError(f"Failed to create bucket: {response.text}")

        return response.json()

    def ensure_bucket(self, bucket_id: str, **kwargs) -> dict:
        """Ensure a bucket exists, creating it if necessary."""
        bucket = self.get_bucket(bucket_id)
        if bucket:
            return bucket
        return self.create_bucket(bucket_id, **kwargs)

    def upload(
        self,
        bucket_id: str,
        path: str,
        file_data: bytes,
        content_type: str = "application/octet-stream",
        upsert: bool = True,
    ) -> str:
        """Upload a file to storage."""
        path = path.lstrip("/")
        headers = {
            "Content-Type": content_type,
            "x-upsert": str(upsert).lower(),
        }

        response = self._request(
            "POST",
            f"/object/{bucket_id}/{path}",
            content=file_data,
            headers=headers,
        )

        if response.status_code not in (200, 201):
            raise StorageError(f"Failed to upload file: {response.text}")

        return f"{bucket_id}/{path}"

    def stream_download(self, bucket_id: str, path: str, chunk_size: int = 8192):
        """Stream a file download as a generator.

        Uses the first-yield-is-metadata pattern: the first yielded value is
        the Content-Length header (str or None).  Subsequent yields are byte
        chunks.  Callers should call ``next(gen)`` to eagerly trigger the HTTP
        connection and status validation *before* handing the generator to a
        streaming ``Response``.

        The httpx connection stays open until the generator is exhausted or
        garbage-collected.
        """
        path = path.lstrip("/")
        url = f"{self.storage_url}/object/{bucket_id}/{path}"
        headers = {**self.headers}

        try:
            with httpx.stream("GET", url, headers=headers, timeout=30) as response:
                if response.status_code == 404:
                    raise StorageError(f"File not found: {bucket_id}/{path}")
                if response.status_code != 200:
                    raise StorageError(f"Failed to download file: {response.status_code}")
                yield response.headers.get("content-length")  # metadata sentinel
                yield from response.iter_bytes(chunk_size=chunk_size)
        except StorageError:
            raise
        except httpx.RequestError as e:
            logger.error(f"Streaming download failed: {e}")
            raise StorageError(f"Streaming download failed: {e}") from e

    def stream_download_from_path(self, storage_path: str, chunk_size: int = 8192):
        """Stream download using the full storage path (bucket/path)."""
        parts = storage_path.split("/", 1)
        if len(parts) != 2:
            raise StorageError(f"Invalid storage path: {storage_path}")
        bucket_id, path = parts
        return self.stream_download(bucket_id, path, chunk_size)

    def download(self, bucket_id: str, path: str) -> bytes:
        """Download a file from storage."""
        path = path.lstrip("/")
        response = self._request("GET", f"/object/{bucket_id}/{path}")

        if response.status_code == 404:
            raise StorageError(f"File not found: {bucket_id}/{path}")
        if response.status_code != 200:
            raise StorageError(f"Failed to download file: {response.text}")

        return response.content

    def download_from_path(self, storage_path: str) -> bytes:
        """Download a file using the full storage path."""
        parts = storage_path.split("/", 1)
        if len(parts) != 2:
            raise StorageError(f"Invalid storage path: {storage_path}")
        bucket_id, path = parts
        return self.download(bucket_id, path)

    def delete(self, bucket_id: str, paths: list[str]) -> None:
        """Delete files from storage."""
        response = self._request(
            "DELETE",
            f"/object/{bucket_id}",
            json={"prefixes": paths},
        )
        if response.status_code not in (200, 204):
            raise StorageError(f"Failed to delete files: {response.text}")

    def has_public_url(self) -> bool:
        """Return True if a public storage URL is configured."""
        return self.public_storage_url is not None

    def create_signed_url(
        self,
        bucket_id: str,
        path: str,
        expires_in: int = 3600,
        public: bool = False,
    ) -> str:
        """Create a signed URL for private file access.

        Args:
            bucket_id: Storage bucket identifier.
            path: Object path within the bucket.
            expires_in: Expiry time in seconds (default 3600).
            public: When True and a public storage URL is configured, the
                returned URL uses the public base so external services (e.g.
                LLM providers) can reach it.
        """
        path = path.lstrip("/")
        response = self._request(
            "POST",
            f"/object/sign/{bucket_id}/{path}",
            json={"expiresIn": expires_in},
        )

        if response.status_code != 200:
            raise StorageError(f"Failed to create signed URL: {response.text}")

        data = response.json()
        signed_url_path = data.get("signedURL")
        if not signed_url_path:
            raise StorageError(f"Unexpected signed URL response (missing 'signedURL'): {data}")
        base_url = (
            self.public_storage_url if public and self.public_storage_url else self.storage_url
        )
        return f"{base_url}{signed_url_path}"


def _sanitize_filename_for_storage(filename: str) -> str:
    """Sanitize a filename for Supabase Storage object keys.

    Uses anyascii to transliterate non-ASCII characters to their closest
    ASCII equivalents, then strips any remaining disallowed characters.
    The original filename is preserved in the source record (name column
    and auto_metadata.origin_path).
    """
    base, ext = os.path.splitext(filename)
    # Transliterate to ASCII (e.g. 测试 → Ce Shi, résumé → resume)
    ascii_base = anyascii(base)
    # Replace any non-word/non-hyphen chars with underscore
    safe_base = re.sub(r"[^\w\-]", "_", ascii_base)
    safe_base = re.sub(r"_+", "_", safe_base).strip("_")
    if not safe_base:
        safe_base = "file"
    # Append short hash of original name to prevent transliteration collisions
    name_hash = hashlib.sha256(filename.encode()).hexdigest()[:8]
    safe_base = f"{safe_base}_{name_hash}"
    safe_ext = re.sub(r"[^\w.]", "", ext, flags=re.ASCII)
    return safe_base + safe_ext


def get_source_storage_path(source_id: str, filename: str) -> str:
    """Get the storage path for a source's original file."""
    return f"{source_id}/original/{_sanitize_filename_for_storage(filename)}"


def get_derivative_storage_path(source_id: str, derivative_type: str, filename: str) -> str:
    """Get the storage path for a source's derivative."""
    return f"{source_id}/derivatives/{derivative_type}/{_sanitize_filename_for_storage(filename)}"


# Default bucket for sources
SOURCES_BUCKET = "sources"


# Singleton instance
_storage_client: SupabaseStorage | None = None


def get_storage() -> SupabaseStorage:
    """Get the storage client singleton."""
    global _storage_client
    if _storage_client is None:
        _storage_client = SupabaseStorage()
    return _storage_client
