"""SQLite bootstrap; SQLAlchemy remains a supported dependency for production migrations."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, sop_id TEXT, source_type TEXT, source_name TEXT, started_at TEXT, ended_at TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS step_results (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, step_id TEXT, status TEXT, started_at TEXT, completed_at TEXT, confidence REAL, reason TEXT);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, event_type TEXT, severity TEXT, timestamp TEXT, step_id TEXT, message TEXT, confidence REAL, evidence_json TEXT);
CREATE TABLE IF NOT EXISTS vlm_results (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, timestamp TEXT, provider TEXT, model TEXT, request_json TEXT, response_json TEXT, latency_ms REAL, success INTEGER, error_message TEXT);
"""


def sqlite_path(url: str) -> Path:
    if not url.startswith("sqlite:///"): raise ValueError("This local build supports sqlite:/// URLs only")
    return Path(url.removeprefix("sqlite:///"))


def connect(url: str) -> sqlite3.Connection:
    path = sqlite_path(url); path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False); connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA); connection.commit(); return connection
