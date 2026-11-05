import ast
import pathlib

import pytest

from standing_orders.engine.clock import FakeClock
from standing_orders.engine.program import OPERATORS, parse_program, validate_program
from standing_orders.engine.types import Tri
from standing_orders.storage.db import connect

ENGINE = pathlib.Path(__file__).parent.parent / "standing_orders" / "engine"


def test_schema_creates_all_tables(tmp_path):
    conn = connect(str(tmp_path / "so.db"))
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"orders", "events", "examples", "verifications", "counters"} <= names


def test_fake_clock_advances():
    c = FakeClock(10)
    c.advance(2.5)
    assert c.now() == 12.5


def test_unknown_is_distinct():
    assert Tri.UNKNOWN not in (Tri.TRUE, Tri.FALSE)
    assert "held_for" in OPERATORS


def test_engine_has_no_hermes_imports():
    for path in ENGINE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                   [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            for m in mods:
                assert not m.split(".")[0] in ("gateway", "hermes_cli", "agent", "tools", "hermes_constants"), (path, m)


def test_stubs_not_implemented():
    with pytest.raises(NotImplementedError):
        parse_program({})
    with pytest.raises(NotImplementedError):
        validate_program(None)
