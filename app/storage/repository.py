"""Thread-safe, parameterized SQLite event repository."""
from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

from .database import connect, utc_now_iso
from .frames import FrameStore


class Repository:
    def __init__(self, url: str, frame_store: FrameStore | None = None) -> None:
        self.db, self.lock = connect(url), threading.Lock()
        self.frame_store = frame_store

    def create_session(self, session: dict[str, Any]) -> None:
        with self.lock:
            self.db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, NULL, ?)", (session["id"], session["sop_id"], session["source_type"], session["source_name"], session["started_at"], session["status"])); self.db.commit()

    def stop_session(self, session_id: str, ended_at: str) -> None:
        with self.lock: self.db.execute("UPDATE sessions SET ended_at=?, status='STOPPED' WHERE id=?", (ended_at, session_id)); self.db.commit()

    def save_step(self, session_id: str, step: Any) -> None:
        with self.lock:
            self.db.execute("INSERT INTO step_results(session_id,step_id,status,started_at,completed_at,confidence,reason) VALUES(?,?,?,?,?,?,?)", (session_id, step.step_id, step.status.value, _iso(step.started_at), _iso(step.completed_at), step.confidence, step.reason)); self.db.commit()

    def save_event(self, session_id: str, event_type: str, step_id: str | None, message: str, confidence: float, evidence: dict[str, Any], severity: str = "INFO", frame_path: str | None = None) -> int:
        with self.lock:
            cursor = self.db.execute("INSERT INTO events(session_id,event_type,severity,timestamp,step_id,message,confidence,evidence_json,frame_path) VALUES(?,?,?,?,?,?,?,?,?)", (session_id,event_type,severity,utc_now_iso(),step_id,message,confidence,json.dumps(evidence, ensure_ascii=False),frame_path)); self.db.commit(); return int(cursor.lastrowid)

    def set_event_frame(self, event_id: int, frame_path: str) -> None:
        """Backfill the frame path once the caller has both the event id and the saved JPEG."""
        with self.lock:
            self.db.execute("UPDATE events SET frame_path=? WHERE id=?", (frame_path, event_id)); self.db.commit()

    def save_vlm(self, session_id: str, provider: str, model: str, request: dict[str, Any], response: dict[str, Any], latency_ms: float, success: bool, error: str | None = None) -> None:
        with self.lock:
            timestamp = utc_now_iso()
            self.db.execute("INSERT INTO vlm_results(session_id,timestamp,provider,model,request_json,response_json,latency_ms,success,error_message) VALUES(?,?,?,?,?,?,?,?,?)", (session_id,timestamp,provider,model,json.dumps(request),json.dumps(response),latency_ms,int(success),error)); self.db.commit()

    def events(self, session_id: str | None = None, *, limit: int = 200, offset: int = 0, event_type: str | None = None, severity: str | None = None, since: str | None = None) -> list[dict[str, Any]]:
        query, args = "SELECT * FROM events", []
        query, args = _apply_event_filters(query, args, session_id, event_type, severity, since)
        query += " ORDER BY id DESC"
        if limit > 0:
            query += " LIMIT ? OFFSET ?"; args += [limit, offset]
        with self.lock: return [dict(row) for row in self.db.execute(query, args).fetchall()]

    def count_events(self, session_id: str | None = None, *, event_type: str | None = None, severity: str | None = None, since: str | None = None) -> int:
        query, args = "SELECT COUNT(*) c FROM events", []
        query, args = _apply_event_filters(query, args, session_id, event_type, severity, since)
        with self.lock: return int(self.db.execute(query, args).fetchone()["c"])

    def event(self, event_id: int) -> dict[str, Any] | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone(); return dict(row) if row else None

    def sessions(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        query = ("SELECT s.*, COALESCE(e.event_count, 0) AS event_count FROM sessions s "
                 "LEFT JOIN (SELECT session_id, COUNT(*) event_count FROM events GROUP BY session_id) e ON e.session_id = s.id "
                 "ORDER BY s.started_at DESC LIMIT ? OFFSET ?")
        with self.lock: return [dict(row) for row in self.db.execute(query, (limit, offset)).fetchall()]

    def session(self, session_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone(); return dict(row) if row else None

    def vlm_records(self, session_id: str | None = None, *, limit: int = 50) -> list[dict[str, Any]]:
        query, args = "SELECT * FROM vlm_results", []
        if session_id: query += " WHERE session_id=?"; args = [session_id]
        query += " ORDER BY id DESC LIMIT ?"; args = [*args, limit]
        with self.lock: return [dict(row) for row in self.db.execute(query, args).fetchall()]

    def stats(self) -> dict[str, Any]:
        """Aggregate counters for the Prometheus endpoint."""
        with self.lock:
            sessions_total = int(self.db.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"])
            events_total = int(self.db.execute("SELECT COUNT(*) c FROM events").fetchone()["c"])
            events_by_type = {row["event_type"]: row["c"] for row in self.db.execute("SELECT event_type, COUNT(*) c FROM events GROUP BY event_type").fetchall()}
            events_by_severity = {row["severity"]: row["c"] for row in self.db.execute("SELECT severity, COUNT(*) c FROM events GROUP BY severity").fetchall()}
            vlm_total = int(self.db.execute("SELECT COUNT(*) c FROM vlm_results").fetchone()["c"])
            vlm_failures = int(self.db.execute("SELECT COUNT(*) c FROM vlm_results WHERE success=0").fetchone()["c"])
            avg_row = self.db.execute("SELECT AVG(latency_ms) a FROM vlm_results WHERE success=1").fetchone()
            vlm_latency_ms_avg = float(avg_row["a"]) if avg_row["a"] is not None else 0.0
        return {
            "sessions_total": sessions_total,
            "events_total": events_total,
            "events_by_type": events_by_type,
            "events_by_severity": events_by_severity,
            "vlm_total": vlm_total,
            "vlm_failures": vlm_failures,
            "vlm_latency_ms_avg": vlm_latency_ms_avg,
        }

    def purge_older_than(self, retention_days: int) -> dict[str, int]:
        """Delete rows older than retention_days; leaves RUNNING sessions untouched."""
        if retention_days <= 0:
            return {}
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        with self.lock:
            deleted = {}
            deleted["events"] = self.db.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,)).rowcount
            deleted["vlm_results"] = self.db.execute("DELETE FROM vlm_results WHERE timestamp < ?", (cutoff,)).rowcount
            deleted["step_results"] = self.db.execute("DELETE FROM step_results WHERE started_at < ?", (cutoff,)).rowcount
            deleted["sessions"] = self.db.execute("DELETE FROM sessions WHERE started_at < ? AND status != 'RUNNING'", (cutoff,)).rowcount
            self.db.commit()
        if self.frame_store is not None:
            deleted["frames"] = self.frame_store.purge_older_than(retention_days)
        return deleted


def _apply_event_filters(query: str, args: list[Any], session_id: str | None, event_type: str | None, severity: str | None, since: str | None) -> tuple[str, list[Any]]:
    clauses = []
    if session_id: clauses.append("session_id=?"); args.append(session_id)
    if event_type: clauses.append("event_type=?"); args.append(event_type)
    if severity: clauses.append("severity=?"); args.append(severity)
    if since: clauses.append("timestamp>=?"); args.append(since)
    if clauses: query += " WHERE " + " AND ".join(clauses)
    return query, args


def _iso(value: datetime | None) -> str | None: return value.isoformat() if value else None
