"""Backend routes for the Desktop pane: /api/plugins/standing-orders/... Thin wrappers over pane_data."""

from __future__ import annotations
import sys
from pathlib import Path

from fastapi import APIRouter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for the standing_orders package

from standing_orders.hermes import pane_data  # noqa: E402
from standing_orders.hermes.paths import Layout, data_dir  # noqa: E402

router = APIRouter()


@router.get("/snapshot")
async def snapshot():
    return pane_data.snapshot(Layout(data_dir()))


@router.get("/preview")
async def preview():
    return pane_data.preview(Layout(data_dir()))


@router.post("/orders/{order_id}/status")
async def order_status(order_id: str, body: dict):
    return pane_data.set_order_status(Layout(data_dir()), order_id, str(body.get("status", "")))


@router.post("/events/{event_id}/label")
async def event_label(event_id: str, body: dict):
    return pane_data.label_event(Layout(data_dir()), event_id, str(body.get("label", "")), str(body.get("note", "")))
