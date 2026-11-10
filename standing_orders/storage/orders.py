"""OrderRepo: orders, events and counters in SQLite. The bus between engine, tools and the pane.

Plain sqlite3, no Hermes imports. Implements the Runner's ProgramStore (armed()) and sink (emit())."""

from __future__ import annotations
import dataclasses
import json
import sqlite3
import threading
import time

from ..engine.program import WatchProgram, parse_program
from ..engine.types import Event
from .db import EVENT_KINDS, EVENT_LABELS, ORDER_STATUSES

_EVENT_COLS = ("id", "order_id", "fired_at", "kind", "reason", "confidence", "clip_path", "label", "label_note")


def program_to_dict(program: WatchProgram) -> dict:
    d = dataclasses.asdict(program)
    d["zones"] = {n: [[x, y] for x, y in pts] for n, pts in program.zones.items()}
    return d


def program_from_dict(data: dict) -> WatchProgram:
    return parse_program(data)


LIST_ITEM_MAX = 200


class OrderRepo:
    def __init__(self, conn: sqlite3.Connection, time_fn=time.time):
        self.conn, self._time = conn, time_fn
        self._lock = threading.Lock()
        self._parsed: dict[str, tuple[float, WatchProgram]] = {}  # id -> (updated_at, program)

    # orders
    def create(self, program: WatchProgram, status: str = "draft") -> None:
        self._check_status(status)
        now = self._time()
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO orders (id, text, program_json, status, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                (program.id, program.text, json.dumps(program_to_dict(program)), status, now, now))

    def get(self, order_id: str) -> WatchProgram | None:
        row = self.conn.execute("SELECT program_json FROM orders WHERE id=?", (order_id,)).fetchone()
        return program_from_dict(json.loads(row[0])) if row else None

    def status(self, order_id: str) -> str | None:
        row = self.conn.execute("SELECT status FROM orders WHERE id=?", (order_id,)).fetchone()
        return row[0] if row else None

    def list(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, text, status, created_at, updated_at FROM orders ORDER BY created_at, id").fetchall()
        return [dict(zip(("id", "text", "status", "created_at", "updated_at"), r)) for r in rows]

    def armed(self) -> list[WatchProgram]:
        rows = self.conn.execute(
            "SELECT id, program_json, updated_at FROM orders WHERE status='armed' ORDER BY created_at, id").fetchall()
        out = []
        for oid, pj, updated in rows:
            cached = self._parsed.get(oid)
            if cached is None or cached[0] != updated:
                try:
                    cached = (updated, program_from_dict(json.loads(pj)))
                except Exception:
                    continue  # a corrupt row must not stop the engine
                self._parsed[oid] = cached
            out.append(cached[1])
        return out

    def set_status(self, order_id: str, status: str) -> None:
        self._check_status(status)
        with self._lock, self.conn:
            cur = self.conn.execute("UPDATE orders SET status=?, updated_at=? WHERE id=?",
                                    (status, self._time(), order_id))
        if cur.rowcount == 0:
            raise KeyError(f"no such order: {order_id}")

    def update_program(self, program: WatchProgram) -> None:
        with self._lock, self.conn:
            cur = self.conn.execute("UPDATE orders SET text=?, program_json=?, updated_at=? WHERE id=?",
                                    (program.text, json.dumps(program_to_dict(program)), self._time(), program.id))
        if cur.rowcount == 0:
            raise KeyError(f"no such order: {program.id}")

    @staticmethod
    def _check_status(status: str) -> None:
        if status not in ORDER_STATUSES:
            raise ValueError(f"status must be one of {ORDER_STATUSES}, got {status!r}")

    # events
    def emit(self, event: Event) -> None:
        """Runner sink: persist a fired event and bump the fire counter."""
        self.add_event(event)
        if event.kind == "live":
            self.add_counters(event.order_id, fires=1)

    def add_event(self, event: Event) -> None:
        if event.kind not in EVENT_KINDS:
            raise ValueError(f"kind must be one of {EVENT_KINDS}, got {event.kind!r}")
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO events (id, order_id, fired_at, kind, reason, confidence, clip_path) "
                "VALUES (?,?,?,?,?,?,?)",
                (event.id, event.order_id, event.fired_at, event.kind, event.reason, event.confidence,
                 event.clip_path))

    def events(self, order_id: str | None = None, kind: str | None = None, limit: int = 50) -> list[dict]:
        filters = [(c, v) for c, v in (("order_id=?", order_id), ("kind=?", kind)) if v is not None]
        sql = f"SELECT {', '.join(_EVENT_COLS)} FROM events"
        if filters:
            sql += " WHERE " + " AND ".join(c for c, _ in filters)
        sql += " ORDER BY fired_at DESC, id LIMIT ?"
        rows = self.conn.execute(sql, [v for _, v in filters] + [limit]).fetchall()
        return [dict(zip(_EVENT_COLS, r)) for r in rows]

    def get_event(self, event_id: str) -> dict | None:
        row = self.conn.execute(f"SELECT {', '.join(_EVENT_COLS)} FROM events WHERE id=?", (event_id,)).fetchone()
        return dict(zip(_EVENT_COLS, row)) if row else None

    def label_event(self, event_id: str, label: str, note: str = "") -> None:
        if label not in EVENT_LABELS:
            raise ValueError(f"label must be one of {EVENT_LABELS}, got {label!r}")
        with self._lock, self.conn:
            cur = self.conn.execute("UPDATE events SET label=?, label_note=? WHERE id=?", (label, note, event_id))
        if cur.rowcount == 0:
            raise KeyError(f"no such event: {event_id}")

    def delete_backtest_events(self, order_id: str) -> None:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM events WHERE order_id=? AND kind='backtest'", (order_id,))

    # counters (hourly buckets)
    def add_counters(self, order_id: str, *, frames: int = 0, t0_passed: int = 0, t1_decided: int = 0,
                     t2_calls: int = 0, fires: int = 0, now: float | None = None) -> None:
        now = self._time() if now is None else now
        bucket = now - now % 3600
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO counters (bucket_start, order_id, frames, t0_passed, t1_decided, t2_calls, fires) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(bucket_start, order_id) DO UPDATE SET "
                "frames=frames+excluded.frames, t0_passed=t0_passed+excluded.t0_passed, "
                "t1_decided=t1_decided+excluded.t1_decided, t2_calls=t2_calls+excluded.t2_calls, "
                "fires=fires+excluded.fires",
                (bucket, order_id, frames, t0_passed, t1_decided, t2_calls, fires))

    def counters(self, order_id: str | None = None, since: float = 0.0) -> dict[str, int]:
        sql = ("SELECT COALESCE(SUM(frames),0), COALESCE(SUM(t0_passed),0), COALESCE(SUM(t1_decided),0), "
               "COALESCE(SUM(t2_calls),0), COALESCE(SUM(fires),0) FROM counters WHERE bucket_start >= ?")
        args: list = [since - since % 3600]
        if order_id is not None:
            sql, args = sql + " AND order_id=?", args + [order_id]
        row = self.conn.execute(sql, args).fetchone()
        return dict(zip(("frames", "t0_passed", "t1_decided", "t2_calls", "fires"), row))

    # the user's list: the only thing an add_list_item order is allowed to change
    def add_list_item(self, text: str, source_order: str | None = None) -> str:
        text = " ".join(text.split())[:LIST_ITEM_MAX]
        if not text:
            raise ValueError("list item is empty")
        with self._lock, self.conn:
            self.conn.execute("INSERT INTO list_items (text, created_at, source_order) VALUES (?,?,?)",
                              (text, self._time(), source_order))
        return text

    def list_items(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute("SELECT id, text, created_at, source_order FROM list_items "
                                 "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(zip(("id", "text", "created_at", "source_order"), r)) for r in rows]
