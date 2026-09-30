"""Remove text Postgres cannot store from every statement's parameters.

Postgres cannot hold some characters Python strings can:

* NUL (U+0000). ``text`` and ``varchar`` have no NUL -- psycopg refuses to
  send one -- and ``jsonb`` rejects the ``\\u0000`` escape that
  ``json.dumps`` writes for it.
* Unpaired surrogates (U+D800-U+DFFF on their own). They cannot be encoded
  as UTF-8 at all, and ``jsonb`` rejects their ``\\uXXXX`` escapes. A JSON
  request body can carry one, and Python decodes it into a ``str``.

Such characters arrive from outside: extracted documents (a PDF's text layer,
a spreadsheet cell), a user's message, a model's reply, a tool's result. One
of them used to fail the whole write -- indexing a source, recording a run --
and the dozens of places that write text cannot each be trusted to clean it.
So it is done once, on the engine, for every statement: NUL is dropped and
an unpaired surrogate becomes U+FFFD, the Unicode replacement character.

A string parameter holding one of those escapes is parsed as JSON; when it
is JSON (a document about to be cast to ``jsonb``, as the routes write them),
the characters are removed from its values and it is serialized again.
A string that is not JSON is left as it is: there ``\\u0000`` is six
ordinary characters, which ``text`` stores fine.
"""

from __future__ import annotations

import json
import re
from typing import Any

from psycopg.types.json import Json, Jsonb
from sqlalchemy import event
from sqlalchemy.engine.interfaces import ExecuteStyle

_UNPAIRED_SURROGATE = re.compile("[\ud800-\udfff]")

# What may be a \u escape jsonb refuses: NUL, or a surrogate. Only a cheap
# screen -- a surrogate pair is fine, and "\\u0000" is an escaped backslash
# followed by "u0000" -- so a match is parsed before anything is changed.
_REFUSED_JSON_ESCAPE = re.compile(r"\\u(?:0000|[dD][89a-fA-F][0-9a-fA-F]{2})")


def clean_text(value: str) -> str:
    """*value* without NUL, and with each unpaired surrogate replaced."""
    if "\x00" in value:
        value = value.replace("\x00", "")
    if not value.isascii() and _UNPAIRED_SURROGATE.search(value):
        value = _UNPAIRED_SURROGATE.sub("\ufffd", value)
    return value


def _rebuilt(sequence: list | tuple, items) -> list | tuple:
    """*items* in a sequence of *sequence*'s kind. A named tuple is rebuilt as
    itself; its constructor takes fields, not one iterable."""
    if isinstance(sequence, list):
        return list(items)
    if hasattr(sequence, "_make"):
        return type(sequence)._make(items)
    return tuple(items)


def _clean_object(value: Any) -> Any:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, dict):
        return {_clean_object(k): _clean_object(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return _rebuilt(value, (_clean_object(v) for v in value))
    return value


def _clean_json_document(value: str) -> str:
    try:
        document = json.loads(value)
    except ValueError:
        return value
    cleaned = _clean_object(document)
    if cleaned == document:
        return value
    return json.dumps(cleaned)


def clean_parameter(value: Any) -> Any:
    """One statement parameter, as Postgres can store it."""
    if isinstance(value, str):
        value = clean_text(value)
        if "\\u" in value and _REFUSED_JSON_ESCAPE.search(value):
            value = _clean_json_document(value)
        return value
    if isinstance(value, (Json, Jsonb)):
        # How SQLAlchemy binds a JSON/JSONB column's Python value.
        return type(value)(_clean_object(value.obj), value.dumps)
    if isinstance(value, (list, tuple)):
        return _rebuilt(value, (clean_parameter(v) for v in value))
    return value


def _clean_parameter_set(parameters: Any) -> Any:
    if isinstance(parameters, dict):
        return {k: clean_parameter(v) for k, v in parameters.items()}
    if isinstance(parameters, (list, tuple)):
        return _rebuilt(parameters, (clean_parameter(v) for v in parameters))
    return parameters


def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    # ``executemany`` is also true for an "insertmanyvalues" batch -- a
    # multi-row INSERT from an ORM flush or ``insert().returning()`` -- whose
    # parameters are ONE flattened mapping, not a list of them. Only a driver
    # executemany passes a list.
    if executemany and (context is None or context.execute_style is ExecuteStyle.EXECUTEMANY):
        parameters = [_clean_parameter_set(p) for p in parameters]
    else:
        parameters = _clean_parameter_set(parameters)
    return statement, parameters


def install_pg_text_guard(engine) -> None:
    """Clean the parameters of every statement *engine* executes. Installing
    it again on the same engine adds nothing: SQLAlchemy keeps one listener."""
    event.listen(engine, "before_cursor_execute", _before_cursor_execute, retval=True)
