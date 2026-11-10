"""Data behind the Desktop pane and /orders. Plain functions over the shared SQLite file and data dir:
no FastAPI, no Hermes imports, so they run in pytest. The pane never talks to the engine thread."""

from __future__ import annotations
import base64
import time

import cv2

from ..engine.corrections import apply_correction
from ..engine.embedder import maybe_embedder
from ..llm.compiler import plan_text
from ..llm.verifier import CachedVerifier
from ..storage.db import connect
from ..storage.orders import OrderRepo
from .paths import Layout

PREVIEW_WIDTH = 480


class _CacheOnly:
    """Inner verifier for pane corrections: the pane has no model access, so a cache miss is UNKNOWN."""

    async def check(self, crops, predicates):
        from ..engine.types import Tri, Verdict
        return {n: Verdict(Tri.UNKNOWN, 0.0, "not in the verification cache") for n in predicates}


def snapshot(layout: Layout, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    repo = OrderRepo(connect(layout.db))
    since = now - 3600
    orders = []
    for o in repo.list():
        program = repo.get(o["id"])
        c = repo.counters(o["id"], since=since)
        orders.append({
            **o, "plan": plan_text(program) if program else "", "zones": sorted(program.zones) if program else [],
            "vision_calls_last_hour": c["t2_calls"],
            "vision_budget_per_hour": program.vision_budget_per_hour if program else None,
            "fires_last_hour": c["fires"]})
    return {"camera": layout.read_status(), "orders": orders, "funnel_last_hour": repo.counters(since=since),
            "events": repo.events(limit=30), "list_items": repo.list_items(10), "now": now}


def preview(layout: Layout) -> dict:
    """Newest recorded frame as a small JPEG data URL, or {"image": None}."""
    path = layout.newest_frame()
    img = cv2.imread(str(path)) if path else None
    if img is None:
        return {"image": None}
    h, w = img.shape[:2]
    img = cv2.resize(img, (PREVIEW_WIDTH, max(1, round(h * PREVIEW_WIDTH / w))))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if not ok:
        return {"image": None}
    return {"image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(), "taken_at": path.stat().st_mtime}


def set_order_status(layout: Layout, order_id: str, status: str) -> dict:
    repo = OrderRepo(connect(layout.db))
    if status not in ("armed", "paused"):
        return {"error": "status must be armed or paused"}
    if repo.status(order_id) is None:
        return {"error": f"no such order: {order_id}"}
    repo.set_status(order_id, status)
    return {"order_id": order_id, "status": status}


def label_event(layout: Layout, event_id: str, label: str, note: str = "") -> dict:
    """Right/Wrong buttons. Runs the replay proof from the verification cache only; anything the cache
    cannot answer stays UNKNOWN, so a model-backed replay still needs `so_correct` in chat."""
    conn = connect(layout.db)
    event = OrderRepo(conn).get_event(event_id)
    if event is None:
        return {"error": f"no such event: {event_id}"}
    try:
        res = apply_correction(
            conn, event["order_id"], event_id, label, None, note, embedder=maybe_embedder(),
            verifier_factory=lambda clock: CachedVerifier(_CacheOnly(), conn), clip_dir=str(layout.clips))
    except Exception as e:
        return {"error": f"correction failed: {type(e).__name__}: {e}"}
    return {"event_id": event_id, "label": label, "accepted": res.accepted, "flipped": res.flipped,
            "broken_events": res.broken, "result": res.message}


def summary_text(layout: Layout, now: float | None = None) -> str:
    """One chat-sized status block for /orders."""
    s = snapshot(layout, now)
    cam = s["camera"]
    lines = [f"Camera: {cam.get('state')}" + (f" ({cam['detail']})" if cam.get("detail") else "")]
    f = s["funnel_last_hour"]
    lines.append(f"Last hour: {f['frames']} frames, {f['t0_passed']} changed, {f['t1_decided']} settled locally, "
                 f"{f['t2_calls']} vision calls, {f['fires']} firings")
    if not s["orders"]:
        lines.append("No orders yet. Describe one in chat, e.g. \"tell me if I leave my mug on the desk\".")
    for o in s["orders"]:
        lines.append(f"- {o['id']} [{o['status']}] {o['text']} "
                     f"({o['vision_calls_last_hour']}/{o['vision_budget_per_hour']} vision calls/h)")
    if s["list_items"]:
        lines.append("List: " + "; ".join(i["text"] for i in s["list_items"]))
    return "\n".join(lines)
