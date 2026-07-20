"""Tests for Repository.purge_older_than data-retention enforcement."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from app.storage.repository import Repository


def _repo(tmp_path) -> Repository:
    return Repository(f"sqlite:///{tmp_path}/r.db")


def test_purge_older_than_removes_old_rows_keeps_recent(tmp_path) -> None:
    repo = _repo(tmp_path)
    old = (datetime.now().astimezone() - timedelta(days=100)).isoformat()
    recent = datetime.now().astimezone().isoformat()

    repo.db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, NULL, ?)", ("old-session", "sop", "mock", "src", old, "STOPPED"))
    repo.db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, NULL, ?)", ("recent-session", "sop", "mock", "src", recent, "STOPPED"))
    repo.db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, NULL, ?)", ("old-running-session", "sop", "mock", "src", old, "RUNNING"))
    repo.db.execute("INSERT INTO events(session_id,event_type,severity,timestamp,step_id,message,confidence,evidence_json) VALUES(?,?,?,?,?,?,?,?)",
                     ("old-session", "step_completed", "INFO", old, "one", "old event", 0.9, json.dumps({})))
    repo.db.execute("INSERT INTO events(session_id,event_type,severity,timestamp,step_id,message,confidence,evidence_json) VALUES(?,?,?,?,?,?,?,?)",
                     ("recent-session", "step_completed", "INFO", recent, "one", "recent event", 0.9, json.dumps({})))
    repo.db.execute("INSERT INTO vlm_results(session_id,timestamp,provider,model,request_json,response_json,latency_ms,success,error_message) VALUES(?,?,?,?,?,?,?,?,?)",
                     ("old-session", old, "mock", "m", "{}", "{}", 10.0, 1, None))
    repo.db.execute("INSERT INTO vlm_results(session_id,timestamp,provider,model,request_json,response_json,latency_ms,success,error_message) VALUES(?,?,?,?,?,?,?,?,?)",
                     ("recent-session", recent, "mock", "m", "{}", "{}", 10.0, 1, None))
    repo.db.execute("INSERT INTO step_results(session_id,step_id,status,started_at,completed_at,confidence,reason) VALUES(?,?,?,?,?,?,?)",
                     ("old-session", "one", "COMPLETED", old, old, 0.9, "done"))
    repo.db.execute("INSERT INTO step_results(session_id,step_id,status,started_at,completed_at,confidence,reason) VALUES(?,?,?,?,?,?,?)",
                     ("recent-session", "one", "COMPLETED", recent, recent, 0.9, "done"))
    repo.db.commit()

    deleted = repo.purge_older_than(30)

    assert deleted == {"events": 1, "vlm_results": 1, "step_results": 1, "sessions": 1}

    session_ids = {row["id"] for row in repo.db.execute("SELECT id FROM sessions").fetchall()}
    assert session_ids == {"recent-session", "old-running-session"}
    event_messages = {row["message"] for row in repo.db.execute("SELECT message FROM events").fetchall()}
    assert event_messages == {"recent event"}
    assert repo.db.execute("SELECT COUNT(*) c FROM vlm_results").fetchone()["c"] == 1
    assert repo.db.execute("SELECT COUNT(*) c FROM step_results").fetchone()["c"] == 1


def test_purge_older_than_zero_is_noop(tmp_path) -> None:
    repo = _repo(tmp_path)
    old = (datetime.now().astimezone() - timedelta(days=100)).isoformat()
    repo.db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, NULL, ?)", ("old-session", "sop", "mock", "src", old, "STOPPED"))
    repo.db.commit()

    result = repo.purge_older_than(0)

    assert result == {}
    assert repo.db.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"] == 1
