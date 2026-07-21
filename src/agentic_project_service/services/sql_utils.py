"""Shared SQL identifier validation utilities."""

import re

_FIELD_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")
_MAX_IDENTIFIER_LENGTH = 50


def validate_sql_identifier(name: str) -> str:
    """Validate that a name is a safe SQL identifier.

    Returns the name if valid, otherwise raises ValueError.
    """
    if not _FIELD_NAME_RE.match(name):
        raise ValueError(
            f"Invalid SQL identifier: {name!r}. "
            "Must be alphanumeric + underscores, starting with a letter."
        )
    if len(name) > _MAX_IDENTIFIER_LENGTH:
        raise ValueError(
            f"SQL identifier too long: {name!r} ({len(name)} chars). "
            f"Maximum length is {_MAX_IDENTIFIER_LENGTH}."
        )
    return name
