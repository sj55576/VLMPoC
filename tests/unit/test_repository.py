"""Tests for Repository: filtering/pagination, aggregates, frame integration, and schema migration."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from app.storage.database import connect, utc_now_iso
from app.storage.frames import FrameStore
from app.storage.repository import Repository


def _repo(tmp_path, frame_store: FrameStore | None = None) -> Repository:
    return Repository(f"sqlite:///{tmp_path}/r.db", frame_store=frame_store)


def _session(repo: Repository, session_id: str, status: str = "STOPPED") -> None:
    repo.create_session({"id": session_id, "sop_id": "sop", "source_type": "mock", "source_name": "src", "started_at": utc_now_iso(), "status": status})


def test_events_pagination_and_offset(tmp_path) -> None:
    repo = _repo(tmp_path)
    _session(repo, "s1")
    ids = [repo.save_event("s1", "step_completed", "one", f"msg{i}", 0.9, {}) for i in range(5)]

    page1 = repo.events("s1", limit=2, offset=0)
    page2 = repo.events("s1", limit=2, offset=2)

    assert [row["id"] for row in page1] == list(reversed(ids))[0:2]
    assert [row["id"] for row in page2] == list(reversed(ids))[2:4]


def test_events_limit_zero_means_unbounded(tmp_path) -> None:
    repo = _repo(tmp_path)
    _session(repo, "s1")
    for i in range(3):
        repo.save_event("s1", "step_completed", "one", f"msg{i}", 0.9, {})

    rows = repo.events("s1", limit=0)
    assert len(rows) == 3


def test_events_filters_by_event_type(tmp_path) -> None:
    repo = _repo(tmp_path)
    _session(repo, "s1")
    repo.save_event("s1", "step_completed", "one", "a", 0.9, {})
    repo.save_event("s1", "safety_violation", "one", "b", 0.9, {}, severity="CRITICAL")

    rows = repo.events("s1", event_type="safety_violation")
    assert len(rows) == 1
    assert rows[0]["message"] == "b"


def test_events_filters_by_severity(tmp_path) -> None:
    repo = _repo(tmp_path)
    _session(repo, "s1")
    repo.save_event("s1", "step_completed", "one", "a", 0.9, {}, severity="INFO")
    repo.save_event("s1", "safety_violation", "one", "b", 0.9, {}, severity="CRITICAL")

    rows = repo.events("s1", severity="CRITICAL")
    assert len(rows) == 1
    assert rows[0]["message"] == "b"


def test_events_filters_by_since(tmp_path) -> None:
    repo = _repo(tmp_path)
    _session(repo, "s1")
    repo.save_event("s1", "step_completed", "one", "old", 0.9, {})
    cutoff = utc_now_iso()
    repo.save_event("s1", "step_completed", "one", "new", 0.9, {})

    rows = repo.events("s1", since=cutoff)
    assert {row["message"] for row in rows} == {"new"}


def test_events_filters_combine_and_are_parameterized(tmp_path) -> None:
    repo = _repo(tmp_path)
    _session(repo, "s1")
    repo.save_event("s1", "safety_violation", "one", "a'; DROP TABLE events; --", 0.9, {}, severity="CRITICAL")

    rows = repo.events("s1", event_type="safety_violation", severity="CRITICAL")
    assert len(rows) == 1
    # table must still exist and be queryable
    assert repo.count_events("s1") == 1


def test_count_events_agrees_with_events(tmp_path) -> None:
    repo = _repo(tmp_path)
    _session(repo, "s1")
    for i in range(7):
        repo.save_event("s1", "step_completed", "one", f"m{i}", 0.9, {})

    assert repo.count_events("s1") == 7
    assert len(repo.events("s1", limit=0)) == repo.count_events("s1")
    assert repo.count_events("s1", event_type="step_completed") == 7
    assert repo.count_events("s1", event_type="safety_violation") == 0


def test_event_frame_backfill(tmp_path) -> None:
    repo = _repo(tmp_path)
    _session(repo, "s1")
    event_id = repo.save_event("s1", "step_completed", "one", "msg", 0.9, {})
    assert repo.event(event_id)["frame_path"] is None

    repo.set_event_frame(event_id, "s1/1.jpg")

    assert repo.event(event_id)["frame_path"] == "s1/1.jpg"


def test_save_event_accepts_frame_path_directly(tmp_path) -> None:
    repo = _repo(tmp_path)
    _session(repo, "s1")
    event_id = repo.save_event("s1", "step_completed", "one", "msg", 0.9, {}, frame_path="s1/1.jpg")
    assert repo.event(event_id)["frame_path"] == "s1/1.jpg"


def test_sessions_lists_with_event_count(tmp_path) -> None:
    repo = _repo(tmp_path)
    _session(repo, "s1")
    _session(repo, "s2")
    repo.save_event("s1", "step_completed", "one", "a", 0.9, {})
    repo.save_event("s1", "step_completed", "one", "b", 0.9, {})

    rows = {row["id"]: row for row in repo.sessions()}
    assert rows["s1"]["event_count"] == 2
    assert rows["s2"]["event_count"] == 0


def test_sessions_pagination(tmp_path) -> None:
    repo = _repo(tmp_path)
    for i in range(5):
        _session(repo, f"s{i}")

    rows = repo.sessions(limit=2, offset=0)
    assert len(rows) == 2


def test_session_lookup(tmp_path) -> None:
    repo = _repo(tmp_path)
    _session(repo, "s1")
    assert repo.session("s1") is not None
    assert repo.session("missing") is None


def test_vlm_records_ordered_desc_and_filtered(tmp_path) -> None:
    repo = _repo(tmp_path)
    _session(repo, "s1")
    _session(repo, "s2")
    repo.save_vlm("s1", "mock", "m", {}, {}, 10.0, True)
    repo.save_vlm("s2", "mock", "m", {}, {}, 20.0, True)
    repo.save_vlm("s1", "mock", "m", {}, {}, 30.0, True)

    rows = repo.vlm_records("s1")
    assert len(rows) == 2
    assert rows[0]["latency_ms"] == 30.0  # most recent first


def test_stats_aggregation(tmp_path) -> None:
    repo = _repo(tmp_path)
    _session(repo, "s1")
    repo.save_event("s1", "step_completed", "one", "a", 0.9, {}, severity="INFO")
    repo.save_event("s1", "safety_violation", "one", "b", 0.9, {}, severity="CRITICAL")
    repo.save_vlm("s1", "mock", "m", {}, {}, 10.0, True)
    repo.save_vlm("s1", "mock", "m", {}, {}, 30.0, True)
    repo.save_vlm("s1", "mock", "m", {}, {}, 0.0, False, error="boom")

    stats = repo.stats()

    assert stats["sessions_total"] == 1
    assert stats["events_total"] == 2
    assert stats["events_by_type"] == {"step_completed": 1, "safety_violation": 1}
    assert stats["events_by_severity"] == {"INFO": 1, "CRITICAL": 1}
    assert stats["vlm_total"] == 3
    assert stats["vlm_failures"] == 1
    assert stats["vlm_latency_ms_avg"] == pytest.approx(20.0)


def test_stats_empty_db_defaults(tmp_path) -> None:
    repo = _repo(tmp_path)
    stats = repo.stats()
    assert stats["sessions_total"] == 0
    assert stats["vlm_latency_ms_avg"] == 0.0
    assert stats["events_by_type"] == {}


def test_purge_older_than_also_purges_frames(tmp_path) -> None:
    store = FrameStore(tmp_path / "frames")
    repo = _repo(tmp_path, frame_store=store)
    _session(repo, "s1")
    event_id = repo.save_event("s1", "step_completed", "one", "old", 0.9, {})
    relative = store.save("s1", event_id, np.zeros((10, 10, 3), dtype=np.uint8))
    repo.set_event_frame(event_id, relative)

    # backdate the row and the frame file so both are eligible for purge
    old_ts = "2000-01-01T00:00:00+00:00"
    repo.db.execute("UPDATE events SET timestamp=? WHERE id=?", (old_ts, event_id)); repo.db.commit()
    import os
    import time
    old_mtime = time.time() - 999 * 86400
    os.utime(store.path(relative), (old_mtime, old_mtime))

    deleted = repo.purge_older_than(30)

    assert deleted["events"] == 1
    assert deleted["frames"] == 1
    assert repo.event(event_id) is None
    assert store.path(relative) is None


def test_purge_older_than_without_frame_store_has_no_frames_key(tmp_path) -> None:
    repo = _repo(tmp_path)
    _session(repo, "s1")
    result = repo.purge_older_than(30)
    assert "frames" not in result


def test_migration_adds_frame_path_to_legacy_database(tmp_path) -> None:
    """A pre-existing database without frame_path must be upgraded transparently by connect()."""
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, sop_id TEXT, source_type TEXT, source_name TEXT, started_at TEXT, ended_at TEXT, status TEXT);"
        "CREATE TABLE step_results (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, step_id TEXT, status TEXT, started_at TEXT, completed_at TEXT, confidence REAL, reason TEXT);"
        "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, event_type TEXT, severity TEXT, timestamp TEXT, step_id TEXT, message TEXT, confidence REAL, evidence_json TEXT);"
        "CREATE TABLE vlm_results (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, timestamp TEXT, provider TEXT, model TEXT, request_json TEXT, response_json TEXT, latency_ms REAL, success INTEGER, error_message TEXT);"
    )
    legacy.execute("INSERT INTO events(session_id,event_type,severity,timestamp,step_id,message,confidence,evidence_json) VALUES('s1','step_completed','INFO','ts','one','m',0.9,'{}')")
    legacy.commit()
    legacy.close()

    connection = connect(f"sqlite:///{db_path}")
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(events)").fetchall()}
    assert "frame_path" in columns

    row = connection.execute("SELECT * FROM events WHERE session_id='s1'").fetchone()
    assert row["frame_path"] is None

    connection.execute("INSERT INTO events(session_id,event_type,severity,timestamp,step_id,message,confidence,evidence_json,frame_path) VALUES('s2','step_completed','INFO','ts','one','m',0.9,'{}','p.jpg')")
    connection.commit()
    new_row = connection.execute("SELECT * FROM events WHERE session_id='s2'").fetchone()
    assert new_row["frame_path"] == "p.jpg"


def test_repository_default_frame_store_is_none(tmp_path) -> None:
    repo = Repository(f"sqlite:///{tmp_path}/r.db")
    assert repo.frame_store is None
    assert Path(f"{tmp_path}/r.db").exists()
