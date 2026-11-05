"""SQLite schema, the bus between engine, tools and the pane. Plain sqlite3: no Hermes imports."""
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
  id TEXT PRIMARY KEY, text TEXT NOT NULL, program_json TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders(id),
  fired_at REAL NOT NULL, kind TEXT NOT NULL,
  reason TEXT, confidence REAL, clip_path TEXT,
  label TEXT, label_note TEXT
);
CREATE TABLE IF NOT EXISTS examples (
  id TEXT PRIMARY KEY, order_id TEXT NOT NULL, predicate TEXT NOT NULL,
  polarity INTEGER NOT NULL, embedding BLOB NOT NULL, crop_path TEXT,
  source_event TEXT
);
CREATE TABLE IF NOT EXISTS verifications (
  frame_hash TEXT, predicate_hash TEXT, verdict_json TEXT, created_at REAL,
  PRIMARY KEY (frame_hash, predicate_hash)
);
CREATE TABLE IF NOT EXISTS counters (
  bucket_start REAL, order_id TEXT, frames INTEGER, t0_passed INTEGER,
  t1_decided INTEGER, t2_calls INTEGER, fires INTEGER,
  PRIMARY KEY (bucket_start, order_id)
);
"""

ORDER_STATUSES = ("draft", "armed", "paused")
EVENT_KINDS = ("live", "backtest")
EVENT_LABELS = ("correct", "wrong")


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")  # tools, engine and pane are separate processes
    conn.executescript(SCHEMA)
    return conn
