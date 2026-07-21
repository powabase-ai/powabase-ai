"""Shared query-param parser for paginated list endpoints."""


class ListParamsError(Exception):
    """Raised on invalid sort/order params. Carries an HTTP status code."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def escape_like(s: str) -> str:
    """Escape SQL LIKE wildcards in user input.

    Order matters: backslash first so the others' added escapes are not
    re-escaped on the second pass.
    """
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def parse_list_params(
    request,
    sort_allowed: set[str],
    *,
    default_unpaginated: bool = False,
) -> tuple[int | None, int | None, str | None, str, str]:
    """Parse and validate list query params from a Flask request.

    Returns ``(limit, offset, q, sort, order)``.

    When ``default_unpaginated=True``:
        If neither ``limit`` nor ``offset`` is in ``request.args``, returns
        ``(None, None, q, sort, order)``. The caller must omit LIMIT/OFFSET
        from SQL — this is the back-compat path for endpoints that returned
        every row before pagination was added (currently only
        ``/api/orchestrations``).
        If either param is sent explicitly, the normal clamping rules apply.

    Otherwise:
        ``limit`` defaults to 50, clamped to ``[1, 100]``.
        ``offset`` defaults to 0, clamped to ``>= 0``.

    ``q`` is trimmed; empty-after-trim becomes ``None``; length is capped
    at 200 chars.

    ``sort`` must be in ``sort_allowed``; defaults to ``"created_at"``.
    ``order`` must be ``"asc"`` or ``"desc"``; defaults to ``"desc"``.

    Raises ``ListParamsError`` (status 400) on invalid ``sort`` or ``order``.
    """
    raw_limit = request.args.get("limit")
    raw_offset = request.args.get("offset")

    if default_unpaginated and raw_limit is None and raw_offset is None:
        limit: int | None = None
        offset: int | None = None
    else:
        try:
            limit = int(raw_limit) if raw_limit is not None else 50
        except (TypeError, ValueError):
            limit = 50
        limit = max(1, min(limit, 100))

        try:
            offset = int(raw_offset) if raw_offset is not None else 0
        except (TypeError, ValueError):
            offset = 0
        offset = max(0, offset)

    q_raw = request.args.get("q")
    q: str | None
    if q_raw is None:
        q = None
    else:
        q_trimmed = q_raw.strip()
        if not q_trimmed:
            q = None
        else:
            q = q_trimmed[:200]

    assert "created_at" in sort_allowed, "sort_allowed must contain the default sort 'created_at'"
    sort = request.args.get("sort") or "created_at"
    if sort not in sort_allowed:
        allowed_list = ", ".join(sorted(sort_allowed))
        raise ListParamsError(
            f"Invalid sort '{sort}'. Allowed: {allowed_list}",
            status=400,
        )

    order = (request.args.get("order") or "desc").lower()
    if order not in ("asc", "desc"):
        raise ListParamsError(
            f"Invalid order '{order}'. Allowed: asc, desc",
            status=400,
        )

    return limit, offset, q, sort, order
