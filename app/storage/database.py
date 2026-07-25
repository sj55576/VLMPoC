"""SQLite bootstrap; SQLAlchemy remains a supported dependency for production migrations."""
from __future__ import annotations

import contextlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, sop_id TEXT, source_type TEXT, source_name TEXT, started_at TEXT, ended_at TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS step_results (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, step_id TEXT, status TEXT, started_at TEXT, completed_at TEXT, confidence REAL, reason TEXT);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, event_type TEXT, severity TEXT, timestamp TEXT, step_id TEXT, message TEXT, confidence REAL, evidence_json TEXT, frame_path TEXT);
CREATE TABLE IF NOT EXISTS vlm_results (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, timestamp TEXT, provider TEXT, model TEXT, request_json TEXT, response_json TEXT, latency_ms REAL, success INTEGER, error_message TEXT);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS ix_events_session_id ON events(session_id);
CREATE INDEX IF NOT EXISTS ix_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS ix_events_event_type ON events(event_type);
CREATE INDEX IF NOT EXISTS ix_vlm_results_session_id ON vlm_results(session_id);
CREATE INDEX IF NOT EXISTS ix_vlm_results_timestamp ON vlm_results(timestamp);
CREATE INDEX IF NOT EXISTS ix_step_results_session_id ON step_results(session_id);
CREATE INDEX IF NOT EXISTS ix_step_results_started_at ON step_results(started_at);
"""


def utc_now_iso() -> str:
    """Single source of truth for stored timestamps; keeps ISO lexicographic comparisons correct."""
    return datetime.now(UTC).isoformat()


def sqlite_path(url: str) -> Path:
    if not url.startswith("sqlite:///"): raise ValueError("This local build supports sqlite:/// URLs only")
    return Path(url.removeprefix("sqlite:///"))


def _migrate(connection: sqlite3.Connection) -> None:
    """Idempotently bring pre-existing databases up to the current schema."""
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(events)").fetchall()}
    if "frame_path" not in columns:
        connection.execute("ALTER TABLE events ADD COLUMN frame_path TEXT")


def connect(url: str) -> sqlite3.Connection:
    path = sqlite_path(url); path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False); connection.row_factory = sqlite3.Row
    for pragma in ("PRAGMA journal_mode=WAL", "PRAGMA synchronous=NORMAL", "PRAGMA busy_timeout=5000"):
        # Some URLs/filesystems (e.g. read-only, :memory:) refuse pragmas; degrade gracefully.
        with contextlib.suppress(sqlite3.Error):
            connection.execute(pragma)
    connection.executescript(SCHEMA)
    _migrate(connection)
    connection.executescript(INDEXES)
    connection.commit()
    return connection
