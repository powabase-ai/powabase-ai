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
    """The end user a run acts for, or the service role when ``claims`` is None.

    An end user has both their verified JWT claims (with a non-empty ``sub``)
    and their bearer token; the service role has neither. Anything in between
    is refused, since Storage would otherwise fall back to the service key.
    """

    claims: dict | None = None
    # The end user's own bearer, for services that verify it themselves
    # (Storage). Never logged, never put in the database session.
    token: str | None = None

    def __post_init__(self) -> None:
        if (self.claims is None) != (self.token is None):
            raise ValueError("An end-user caller needs both its claims and its token")
        if self.claims is None:
            return
        if not self.token:
            raise ValueError("An end-user caller needs a non-empty token")
        sub = self.claims.get("sub")
        if not isinstance(sub, str) or not sub:
            raise ValueError("An end-user caller's claims need a non-empty 'sub'")
        # A copy, so a later change to the dict the claims came from (such as
        # the request's payload) cannot change who the run acts for. It stays a
        # plain dict because the claims are serialized with json.dumps.
        object.__setattr__(self, "claims", dict(self.claims))

    @property
    def is_end_user(self) -> bool:
        return self.claims is not None

    @classmethod
    def service(cls) -> ToolCaller:
        return cls()

    @classmethod
    def from_request(cls) -> ToolCaller:
        """The caller of the current request, which must have been authenticated.

        Raises RuntimeError when the request carries no verified end-user
        payload or no bearer token, rather than guessing who the run is for.
        """
        if is_service_role_request():
            return cls.service()
        payload = getattr(g, "jwt_payload", None)
        token = get_token_from_header()
        if not payload or not token:
            raise RuntimeError("The request has no authenticated end user to act for")
        return cls(claims=payload, token=token)

    def __repr__(self) -> str:  # keep the token out of logs and tracebacks
        return f"ToolCaller(end_user={self.claims.get('sub') if self.claims else None!r})"
