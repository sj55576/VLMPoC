"""Thread-safe, parameterized SQLite event repository."""
from __future__ import annotations
import json, threading
from datetime import datetime
from typing import Any
from .database import connect


class Repository:
    def __init__(self, url: str) -> None:
        self.db, self.lock = connect(url), threading.Lock()

    def create_session(self, session: dict[str, Any]) -> None:
        with self.lock:
            self.db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, NULL, ?)", (session["id"], session["sop_id"], session["source_type"], session["source_name"], session["started_at"], session["status"])); self.db.commit()

    def stop_session(self, session_id: str, ended_at: str) -> None:
        with self.lock: self.db.execute("UPDATE sessions SET ended_at=?, status='STOPPED' WHERE id=?", (ended_at, session_id)); self.db.commit()

    def save_step(self, session_id: str, step: Any) -> None:
        with self.lock:
            self.db.execute("INSERT INTO step_results(session_id,step_id,status,started_at,completed_at,confidence,reason) VALUES(?,?,?,?,?,?,?)", (session_id, step.step_id, step.status.value, _iso(step.started_at), _iso(step.completed_at), step.confidence, step.reason)); self.db.commit()

    def save_event(self, session_id: str, event_type: str, step_id: str | None, message: str, confidence: float, evidence: dict[str, Any], severity: str = "INFO") -> int:
        with self.lock:
            cursor = self.db.execute("INSERT INTO events(session_id,event_type,severity,timestamp,step_id,message,confidence,evidence_json) VALUES(?,?,?,?,?,?,?,?)", (session_id,event_type,severity,datetime.now().astimezone().isoformat(),step_id,message,confidence,json.dumps(evidence, ensure_ascii=False))); self.db.commit(); return int(cursor.lastrowid)

    def save_vlm(self, session_id: str, provider: str, model: str, request: dict[str, Any], response: dict[str, Any], latency_ms: float, success: bool, error: str | None = None) -> None:
        with self.lock:
            timestamp = datetime.now().astimezone().isoformat()
            self.db.execute("INSERT INTO vlm_results(session_id,timestamp,provider,model,request_json,response_json,latency_ms,success,error_message) VALUES(?,?,?,?,?,?,?,?,?)", (session_id,timestamp,provider,model,json.dumps(request),json.dumps(response),latency_ms,int(success),error)); self.db.commit()

    def events(self, session_id: str | None = None) -> list[dict[str, Any]]:
        query, args = "SELECT * FROM events", []
        if session_id: query += " WHERE session_id=?"; args = [session_id]
        with self.lock: return [dict(row) for row in self.db.execute(query+" ORDER BY id DESC", args).fetchall()]

    def event(self, event_id: int) -> dict[str, Any] | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone(); return dict(row) if row else None


def _iso(value: datetime | None) -> str | None: return value.isoformat() if value else None
