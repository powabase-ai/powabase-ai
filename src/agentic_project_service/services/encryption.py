"""Symmetric encryption for API keys using Fernet."""

import logging
import os

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)
_fernet: Fernet | None = None


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is not None:
        return _fernet

    key = os.getenv("API_KEY_ENCRYPTION_KEY")
    if key:
        # Wrap the Fernet ctor: it natively raises ValueError on a malformed
        # key (wrong length, not base64, truncated). Re-raise as RuntimeError
        # so the resolver's narrow ``except ValueError`` does NOT catch it
        # and treat every row as a per-row decrypt failure — which would
        # wipe the table on any pod with a typo'd env var.
        try:
            _fernet = Fernet(key.encode() if isinstance(key, str) else key)
        except ValueError as exc:
            raise RuntimeError(
                f"API_KEY_ENCRYPTION_KEY is malformed ({exc}). It must be a "
                f"32-byte url-safe base64-encoded key. Run 'make gen-keys' to "
                f"generate a valid one."
            ) from exc
        return _fernet

    # No env var. Fail hard by default — the previous implementation silently
    # generated a process-local temp key, which produced rows that became
    # permanently undecryptable on the next pod restart (issue #246). Local
    # dev that genuinely needs ephemeral encryption can opt in with
    # ALLOW_TEMP_ENCRYPTION_KEY=1.
    if not _is_truthy(os.getenv("ALLOW_TEMP_ENCRYPTION_KEY")):
        raise RuntimeError(
            "API_KEY_ENCRYPTION_KEY is not set. Refusing to encrypt/decrypt: "
            "a temporary in-process key would silently produce undecryptable "
            "rows on the next restart. Run 'make gen-keys' (or set "
            "API_KEY_ENCRYPTION_KEY in .env). For ephemeral local-dev use "
            "only, set ALLOW_TEMP_ENCRYPTION_KEY=1 to opt in."
        )

    temp = Fernet.generate_key().decode()
    logger.warning(
        "ALLOW_TEMP_ENCRYPTION_KEY is set — using a process-local Fernet key. "
        "Saved API keys will be lost on restart. NEVER use this in production."
    )
    _fernet = Fernet(temp.encode())
    return _fernet


def encrypt_api_key(plaintext: str) -> str:
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_api_key(ciphertext: str) -> str:
    try:
        return _get_fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken:
        raise ValueError("Failed to decrypt API key — encryption key may have changed")


def mask_api_key(key: str) -> str:
    if len(key) <= 8:
        return "***"
    return f"{key[:3]}...{key[-4:]}"
