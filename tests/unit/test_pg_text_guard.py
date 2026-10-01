"""What the statement-parameter guard removes, and what it must leave alone.

The database-level behaviour (that each of these writes now succeeds) is
pinned against a real Postgres in tests/test_pg_text_guard_store.py.
"""

import json
from collections import namedtuple
from types import SimpleNamespace

from psycopg.types.json import Jsonb
from sqlalchemy import create_engine, event
from sqlalchemy.engine.interfaces import ExecuteStyle

from agentic_project_service._pg_text_guard import (
    _before_cursor_execute,
    clean_parameter,
    clean_text,
    install_pg_text_guard,
)


def test_nul_is_removed():
    assert clean_text("a\x00b\x00") == "ab"


def test_ordinary_text_is_returned_as_it_is():
    value = "plain text, ünïcödé and \U0001f600"
    assert clean_text(value) is value


def test_an_unpaired_surrogate_is_replaced():
    assert clean_text("a\ud800b\udfffc") == "a\ufffdb\ufffdc"


def test_a_json_nul_escape_is_removed():
    doc = json.dumps([{"role": "user", "content": "be\x00fore"}])
    assert "\\u0000" in doc

    cleaned = clean_parameter(doc)

    assert json.loads(cleaned) == [{"role": "user", "content": "before"}]


def test_a_json_unpaired_surrogate_escape_is_replaced():
    doc = json.dumps({"content": "x\udc00y"})

    assert json.loads(clean_parameter(doc)) == {"content": "x\ufffdy"}


def test_json_with_a_surrogate_pair_is_left_as_it_is():
    doc = json.dumps({"content": "\U0001f600"})
    assert "\\ud83d\\ude00" in doc

    assert clean_parameter(doc) is doc


def test_an_escaped_backslash_before_u0000_is_text_not_an_escape():
    # The JSON for the six characters \u0000: an escaped backslash, then u0000.
    doc = json.dumps({"content": "write \\u0000 for NUL"})
    assert "\\\\u0000" in doc

    assert clean_parameter(doc) is doc


def test_text_that_is_not_json_keeps_a_literal_escape():
    value = "in C, write \\u0000 or \\0"
    assert clean_parameter(value) is value


def test_a_jsonb_bound_value_is_cleaned_and_keeps_its_serializer():
    def dumps(obj):
        return json.dumps(obj)

    cleaned = clean_parameter(Jsonb({"k\x00": ["v\x00", {"n": 1}]}, dumps))

    assert isinstance(cleaned, Jsonb)
    assert cleaned.obj == {"k": ["v", {"n": 1}]}
    assert cleaned.dumps is dumps


def test_array_elements_are_cleaned():
    assert clean_parameter(["a\x00", "b"]) == ["a", "b"]


def test_other_values_pass_through():
    raw = b"\x00binary\x00"
    assert clean_parameter(raw) is raw
    assert clean_parameter(7) == 7
    assert clean_parameter(None) is None


def test_every_parameter_set_of_an_executemany_is_cleaned():
    statement, params = _before_cursor_execute(
        None, None, "INSERT ...", [{"t": "a\x00"}, {"t": "b\x00"}], None, True
    )
    assert statement == "INSERT ..."
    assert params == [{"t": "a"}, {"t": "b"}]


def test_an_insertmanyvalues_batch_is_one_parameter_set():
    """SQLAlchemy runs a multi-row INSERT (an ORM flush of several objects,
    ``insert().returning()`` over a list) as batches of one statement each:
    ``executemany`` is true, yet the parameters are one flattened mapping."""
    context = SimpleNamespace(execute_style=ExecuteStyle.INSERTMANYVALUES)

    _, params = _before_cursor_execute(
        None, None, "INSERT ...", {"t__0": "a\x00", "t__1": "b"}, context, True
    )

    assert params == {"t__0": "a", "t__1": "b"}


def test_a_driver_executemany_cleans_each_parameter_set():
    context = SimpleNamespace(execute_style=ExecuteStyle.EXECUTEMANY)

    _, params = _before_cursor_execute(
        None, None, "UPDATE ...", [{"t": "a\x00"}, {"t": "b\x00"}], context, True
    )

    assert params == [{"t": "a"}, {"t": "b"}]


def test_a_named_tuple_keeps_its_type():
    Pair = namedtuple("Pair", "name n")

    cleaned = clean_parameter(Pair("a\x00", 1))
    inside = clean_parameter(Jsonb({"p": Pair("b\x00", 2)})).obj["p"]

    assert cleaned == Pair("a", 1) and type(cleaned) is Pair
    assert inside == Pair("b", 2) and type(inside) is Pair


def test_positional_parameters_are_cleaned():
    _, params = _before_cursor_execute(None, None, "SELECT", ("a\x00", 1), None, False)
    assert params == ("a", 1)


def test_installing_twice_registers_one_listener():
    engine = create_engine("postgresql+psycopg://u:p@localhost/db")

    install_pg_text_guard(engine)
    install_pg_text_guard(engine)

    assert event.contains(engine, "before_cursor_execute", _before_cursor_execute)
    assert len(engine.dispatch.before_cursor_execute) == 1
