import json
import pathlib
import re
from types import SimpleNamespace

import cv2
import pytest

from standing_orders.engine.types import Event, Frame
from standing_orders.hermes import tools as T
from standing_orders.hermes.paths import Layout
from standing_orders.storage.ringbuffer import RingBuffer
from tests.helpers import scene
from tests.test_compiler import GOOD, FakeLlm

ROOT = pathlib.Path(__file__).parent.parent


@pytest.fixture
def layout(tmp_path):
    return Layout(tmp_path / "data")


def with_frame(layout, ts=1000.0):
    RingBuffer(str(layout.ring)).add(Frame(scene(True), ts, "cam0"))


def make(layout, *answers):
    t = [5000.0]
    tools = T.StandingOrdersTools(FakeLlm(*answers), layout, time_fn=lambda: t[0])
    tools.t = t
    return tools


def j(s):
    return json.loads(s)


def test_create_returns_plan_and_saves_draft(layout):
    with_frame(layout)
    tools = make(layout, GOOD)
    out = j(tools.create_order({"text": "mug"}))
    assert out["status"] == "draft" and "at most 20 vision calls per hour" in out["plan"]
    assert tools.repo().status(out["order_id"]) == "draft" and tools.repo().armed() == []
    img = [b for b in tools.llm.calls[0]["input"] if b["type"] == "image"][0]
    assert img["data"][:2] == b"\xff\xd8"  # the ring buffer frame, as JPEG


def test_create_errors_are_clear(layout):
    tools = make(layout, GOOD)
    assert "text is required" in j(tools.create_order({}))["error"]
    assert "no reference frame" in j(tools.create_order({"text": "x"}))["error"]
    with_frame(layout)
    bad = make(layout, {**GOOD, "rule": {"exec": 1}}, {**GOOD, "rule": {"exec": 1}})
    assert "could not compile" in j(bad.create_order({"text": "x"}))["error"]
    assert bad.repo().list() == []


def test_arm_disarm_list(layout):
    with_frame(layout)
    tools = make(layout, GOOD)
    oid = j(tools.create_order({"text": "mug"}))["order_id"]
    assert j(tools.arm({"order_id": oid}))["status"] == "armed"
    assert [p.id for p in tools.repo().armed()] == [oid]
    assert j(tools.list_orders({}))["orders"][0]["status"] == "armed"
    assert j(tools.disarm({"order_id": oid}))["status"] == "paused"
    assert tools.repo().armed() == []
    assert "no such order" in j(tools.arm({"order_id": "nope"}))["error"]


def test_status_reports_camera_counters_budget(layout):
    with_frame(layout)
    tools = make(layout, GOOD)
    assert j(tools.status({}))["camera"]["state"] == "not started"
    oid = j(tools.create_order({"text": "mug"}))["order_id"]
    layout.write_status("unavailable", "no camera permission")
    tools.repo().add_counters(oid, frames=5, t2_calls=3, now=tools.t[0])
    s = j(tools.status({}))
    assert s["camera"]["state"] == "unavailable" and "permission" in s["camera"]["detail"]
    assert s["orders"][0]["vision_calls_last_hour"] == 3 and s["orders"][0]["vision_budget_per_hour"] == 20
    assert s["counters_last_hour"]["frames"] == 5


def test_backtest_gated_when_engine_function_missing(layout, monkeypatch):
    with_frame(layout)
    tools = make(layout, GOOD)
    oid = j(tools.create_order({"text": "mug"}))["order_id"]
    monkeypatch.setattr(T, "_engine_fn", lambda *a, **k: None)
    assert "not available" in j(tools.backtest({"order_id": oid}))["error"]


def test_backtest_wraps_engine_function(layout, monkeypatch):
    with_frame(layout)
    tools = make(layout, GOOD)
    oid = j(tools.create_order({"text": "mug"}))["order_id"]
    seen = {}

    def fake_backtest(program, ring, verifier, minutes):
        seen.update(pid=program.id, minutes=minutes)
        ev = Event(program.id, 3.0, "backtest", "mug gone", 0.9, "/c.mp4", f"{program.id}@3")
        return SimpleNamespace(events=[ev], frames_checked=40, unresolved=2)

    monkeypatch.setattr(T, "_engine_fn", lambda *a, **k: fake_backtest)
    out = j(tools.backtest({"order_id": oid, "minutes": 5}))
    assert seen == {"pid": oid, "minutes": 5.0}
    assert out["frames_checked"] == 40 and out["unresolved"] == 2
    assert out["would_have_fired"][0]["clip"] == "/c.mp4"


def test_backtest_unimplemented_stub_gives_clear_error(layout, monkeypatch):
    with_frame(layout)
    tools = make(layout, GOOD)
    oid = j(tools.create_order({"text": "mug"}))["order_id"]

    def stub(program, ring, verifier):
        raise NotImplementedError("backtest lands later")

    monkeypatch.setattr(T, "_engine_fn", lambda *a, **k: stub)
    assert "not available yet" in j(tools.backtest({"order_id": oid}))["error"]


def test_correct_labels_event_and_calls_engine(layout, monkeypatch):
    with_frame(layout)
    tools = make(layout, GOOD)
    oid = j(tools.create_order({"text": "mug"}))["order_id"]
    tools.repo().add_event(Event(oid, 1.0, "live", "r", 0.9, None, f"{oid}@1"))
    calls = []
    monkeypatch.setattr(T, "_engine_fn", lambda *a, **k: (lambda event_id, label: calls.append((event_id, label))))
    out = j(tools.correct({"event_id": f"{oid}@1", "label": "wrong", "note": "plate"}))
    assert calls == [(f"{oid}@1", "wrong")] and "applied" in out["result"]
    assert tools.repo().get_event(f"{oid}@1")["label"] == "wrong"
    assert "label must be" in j(tools.correct({"event_id": f"{oid}@1", "label": "meh"}))["error"]
    assert "no such event" in j(tools.correct({"event_id": "zz", "label": "wrong"}))["error"]


def test_registered_tools_match_plugin_yaml_exactly():
    registered = []
    ctx = SimpleNamespace(llm=None, register_tool=lambda **kw: registered.append(kw))
    T.register_tools(ctx)
    yaml_text = (ROOT / "plugin.yaml").read_text()
    declared = re.findall(r"^  - (so_\w+)$", yaml_text, re.M)
    assert sorted(declared) == sorted(r["name"] for r in registered) == sorted(T.TOOL_NAMES)
    for r in registered:
        assert r["schema"]["name"] == r["name"] and callable(r["handler"])
