import json

import cv2

from standing_orders.hermes import commands, pane_data
from standing_orders.hermes.paths import Layout
from standing_orders.storage.db import connect
from standing_orders.storage.orders import OrderRepo
from standing_orders.engine.program import parse_program
from tests.test_program import MUG
from tests.helpers import scene


def _layout(tmp_path):
    return Layout(tmp_path / "data")


def test_empty_snapshot_and_summary(tmp_path):
    lay = _layout(tmp_path)
    s = pane_data.snapshot(lay)
    assert s["orders"] == [] and s["events"] == [] and s["camera"]["state"] == "not started"
    assert "No orders yet" in pane_data.summary_text(lay)


def test_orders_status_and_preview(tmp_path):
    lay = _layout(tmp_path)
    OrderRepo(connect(lay.db)).create(parse_program(MUG), "draft")
    assert pane_data.set_order_status(lay, MUG["id"], "armed")["status"] == "armed"
    assert "error" in pane_data.set_order_status(lay, "nope", "armed")
    assert "error" in pane_data.set_order_status(lay, MUG["id"], "draft")
    snap = pane_data.snapshot(lay)
    assert snap["orders"][0]["status"] == "armed" and snap["orders"][0]["zones"] == ["desk"]
    assert MUG["id"] in commands.handle_orders.__globals__["summary_text"](lay)
    assert pane_data.preview(lay) == {"image": None}
    lay.ring.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(lay.ring / "0001.jpg"), scene(True, False))
    assert pane_data.preview(lay)["image"].startswith("data:image/jpeg;base64,")


def test_label_unknown_event(tmp_path):
    assert "error" in pane_data.label_event(_layout(tmp_path), "ghost", "wrong")


def test_plugin_api_routes(tmp_path, monkeypatch):
    import importlib.util
    import pathlib

    from fastapi.testclient import TestClient
    from fastapi import FastAPI

    lay = _layout(tmp_path)
    spec = importlib.util.spec_from_file_location("so_plugin_api", pathlib.Path(__file__).parent.parent / "dashboard" / "plugin_api.py")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    monkeypatch.setattr(api, "data_dir", lambda: lay.root)
    app = FastAPI()
    app.include_router(api.router, prefix="/api/plugins/standing-orders")
    c = TestClient(app)
    base = "/api/plugins/standing-orders"
    assert c.get(f"{base}/snapshot").json()["orders"] == []
    assert c.get(f"{base}/preview").json() == {"image": None}
    assert "error" in c.post(f"{base}/events/x/label", json={"label": "wrong"}).json()
