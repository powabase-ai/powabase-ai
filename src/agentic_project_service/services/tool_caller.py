"""Who an agent's tools act for.

Tools run on background threads, after the request that started the run has
moved on, so the caller is captured when the run's tools are loaded and
carried by each tool handler.
"""

from __future__ import annotations

from dataclasses import dataclass

from flask import g

from ..auth import get_token_from_header, is_service_role_request


@dataclass(frozen=True)
class ToolCaller:
    """The end user a run acts for, or the service role when ``claims`` is None."""

    claims: dict | None = None
    # The end user's own bearer, for services that verify it themselves
    # (Storage). Never logged, never put in the database session.
    token: str | None = None

    @property
    def is_end_user(self) -> bool:
        return self.claims is not None

    @classmethod
    def service(cls) -> ToolCaller:
        return cls()

    @classmethod
    def from_request(cls) -> ToolCaller:
        """The caller of the current request."""
        if is_service_role_request():
            return cls.service()
        return cls(
            claims=dict(getattr(g, "jwt_payload", None) or {}), token=get_token_from_header()
        )

    def __repr__(self) -> str:  # keep the token out of logs and tracebacks
        return f"ToolCaller(end_user={self.claims.get('sub') if self.claims else None!r})"
