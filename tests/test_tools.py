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
    tools = T.StandingOrdersTools(FakeLlm(*answers), layout, time_fn=lambda: t[0], snapshot_wait=0)
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


def test_create_ignores_model_supplied_paths(layout, tmp_path):
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"\xff\xd8secret")
    tools = make(layout, GOOD)
    assert "no reference frame" in j(tools.create_order({"text": "mug", "reference_frame_path": str(secret)}))["error"]
    assert tools.llm.calls == []


def test_create_asks_idle_engine_for_one_unrecorded_snapshot(layout):
    import threading, time

    def engine():  # stands in for the idle engine thread
        for _ in range(100):
            if layout.snapshot_request.exists():
                layout.snapshot_request.unlink()
                layout.snapshot.write_bytes(cv2.imencode(".jpg", scene(True))[1].tobytes())
                return
            time.sleep(0.05)

    t = threading.Thread(target=engine)
    t.start()
    tools = T.StandingOrdersTools(FakeLlm(GOOD), layout, snapshot_wait=5)
    out = j(tools.create_order({"text": "mug"}))
    t.join()
    assert out["status"] == "draft"
    assert not layout.snapshot.exists() and not any(layout.ring.glob("*.jpg"))  # used once, never recorded


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


def test_backtest_runs_engine_on_ring_footage(layout, monkeypatch):
    with_frame(layout)
    tools = make(layout, GOOD)
    oid = j(tools.create_order({"text": "mug"}))["order_id"]
    seen = {}

    def fake_backtest(program, source, verifier, **kw):
        seen.update(pid=program.id, cap=kw["max_t2_calls"], frames=len(source._frames))
        ev = Event(program.id, 3.0, "backtest", "mug gone", 0.9, "/c.mp4", f"{program.id}@3")
        return SimpleNamespace(events=[ev], frames_checked=40, t2_calls=3, unresolved_frames=2)

    monkeypatch.setattr(T, "run_backtest", fake_backtest)
    monkeypatch.setattr(T, "maybe_embedder", lambda: None)
    out = j(tools.backtest({"order_id": oid, "minutes": 5}))
    assert seen["pid"] == oid and seen["cap"] == 10
    assert out["frames_checked"] == 40 and out["unresolved_frames"] == 2
    assert out["would_have_fired"][0]["clip"] == "/c.mp4"
    assert [e["kind"] for e in tools.repo().events(oid)] == ["backtest"]
    assert "no such order" in j(tools.backtest({"order_id": "zz"}))["error"]


def test_correct_calls_engine_and_reports_proof(layout, monkeypatch):
    with_frame(layout)
    tools = make(layout, GOOD)
    oid = j(tools.create_order({"text": "mug"}))["order_id"]
    tools.repo().add_event(Event(oid, 1.0, "live", "r", 0.9, None, f"{oid}@1"))
    calls = []

    def fake(conn, order_id, event_id, label, predicate, note, **kw):
        calls.append((order_id, event_id, label, predicate))
        return SimpleNamespace(accepted=True, flipped=True, broken=[], message="ok")

    monkeypatch.setattr(T, "apply_correction", fake)
    monkeypatch.setattr(T, "maybe_embedder", lambda: None)
    out = j(tools.correct({"event_id": f"{oid}@1", "label": "wrong", "predicate": "mug"}))
    assert calls == [(oid, f"{oid}@1", "wrong", "mug")] and out["accepted"] and out["flipped"]
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


def test_register_tools_does_not_touch_ctx_llm():
    class Probe:   # like the catalog validator's capability probe: no `llm` attribute at all
        def register_tool(self, **kw):
            pass
    T.register_tools(Probe())
