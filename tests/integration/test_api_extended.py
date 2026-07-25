"""API coverage for evidence frames, pagination, metrics, auth, and server-side sources."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import ROOT, create_app


def _drive(client: TestClient, frames: int = 30) -> None:
    client.post("/api/session/start", json={"source_type": "mock"})
    for _ in range(frames):
        client.post("/api/vlm/analyze")


def test_event_csv_download_is_reachable():
    """`/api/events/download` used to be shadowed by `/api/events/{event_id}` and 422."""
    with TestClient(create_app()) as client:
        _drive(client, frames=20)
        response = client.get("/api/events/download")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert response.text.splitlines()[0].startswith("id,event_type,severity")


def test_events_are_paginated_filtered_and_counted():
    with TestClient(create_app()) as client:
        _drive(client, frames=30)
        first = client.get("/api/events", params={"limit": 1})
        assert first.status_code == 200
        assert len(first.json()) == 1
        total = int(first.headers["X-Total-Count"])
        assert total >= 2
        second = client.get("/api/events", params={"limit": 1, "offset": 1})
        assert second.json()[0]["id"] != first.json()[0]["id"]
        completed = client.get("/api/events", params={"event_type": "step_completed"})
        assert completed.status_code == 200
        assert all(event["event_type"] == "step_completed" for event in completed.json())


def test_sessions_endpoints_expose_history():
    with TestClient(create_app()) as client:
        _drive(client, frames=20)
        session_id = client.get("/api/session/status").json()["session"]["id"]
        listing = client.get("/api/sessions").json()
        assert any(item["id"] == session_id for item in listing)
        assert listing[0]["event_count"] >= 1
        detail = client.get(f"/api/sessions/{session_id}").json()
        assert detail["id"] == session_id and detail["event_count"] >= 1
        assert client.get("/api/sessions/does-not-exist").status_code == 404
        events = client.get(f"/api/sessions/{session_id}/events")
        assert events.status_code == 200
        assert int(events.headers["X-Total-Count"]) >= 1


def test_events_store_and_serve_an_evidence_frame():
    with TestClient(create_app()) as client:
        client.post("/api/session/start", json={"source_type": "mock"})
        import cv2

        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        ok, buffer = cv2.imencode(".jpg", frame)
        assert ok
        for _ in range(30):
            client.post("/api/analyze/image", files={"file": ("f.jpg", buffer.tobytes(), "image/jpeg")})
        events = client.get("/api/events").json()
        assert events, "the mock timeline should have produced events"
        stored = [event for event in events if event.get("frame_path")]
        assert stored, "save_event_frames is enabled, so events must carry an evidence frame"
        image = client.get(f"/api/events/{stored[0]['id']}/frame")
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"
        assert image.content[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_event_frame_is_404_when_none_was_stored(monkeypatch):
    monkeypatch.setenv("SAVE_EVENT_FRAMES", "false")
    with TestClient(create_app()) as client:
        _drive(client, frames=20)
        events = client.get("/api/events").json()
        assert events
        assert all(not event.get("frame_path") for event in events)
        assert client.get(f"/api/events/{events[0]['id']}/frame").status_code == 404


def test_metrics_endpoint_exposes_prometheus_families():
    with TestClient(create_app()) as client:
        _drive(client, frames=20)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        body = response.text
        assert "# TYPE vlmsop_frames_processed_total counter" in body
        assert "vlmsop_build_info{" in body
        assert "vlmsop_session_active 1" in body


def test_api_key_protects_data_paths_but_not_health(monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200
        assert client.get("/api/config").status_code == 401
        assert client.get("/api/config", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/api/config", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert client.get("/api/config", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert client.get("/metrics", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert client.get("/metrics").status_code == 401
        # The dashboard shell must still load its own assets, or the page cannot render.
        assert client.get("/static/app.js").status_code == 200


def test_rate_limit_rejects_bursts(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    with TestClient(create_app()) as client:
        codes = [client.get("/api/sops").status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200]
        assert codes[-1] == 429
        assert client.get("/health").status_code == 200  # exempt paths stay reachable


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"source_type": "nonsense"}, "Unknown source_type"),
        ({"source_type": "file"}, "requires a video path"),
        ({"source_type": "file", "source_uri": "../../etc/passwd"}, "must live under"),
        ({"source_type": "file", "source_uri": "missing.mp4"}, "was not found"),
        ({"source_type": "rtsp", "source_uri": "file:///etc/passwd"}, "must start with"),
    ],
)
def test_start_rejects_unusable_sources(payload, fragment):
    with TestClient(create_app()) as client:
        response = client.post("/api/session/start", json=payload)
        assert response.status_code == 422
        assert fragment in response.json()["detail"]


def test_browser_source_types_stay_client_driven():
    """A cached dashboard sending `camera` must not make the server open a webcam."""
    with TestClient(create_app()) as client:
        status = client.post("/api/session/start", json={"source_type": "camera"}).json()
        assert status["session"]["source_type"] == "browser"
        assert status["source"]["type"] == "browser"


@pytest.fixture
def sample_clip(monkeypatch):
    """A short video inside ./data, the only directory a file source may read from."""
    monkeypatch.setenv("SOURCE_TARGET_FPS", "500")
    import cv2

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    clip = data_dir / "pytest_source_clip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for index in range(10):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:, :, index % 3] = 255
        writer.write(frame)
    writer.release()
    try:
        yield clip
    finally:
        Path(clip).unlink(missing_ok=True)


def _await_ingestion(client: TestClient, started: dict, expected: int) -> dict:
    deadline = time.time() + 10
    status = started
    while time.time() < deadline and status["frames_processed"] < expected:
        time.sleep(0.05)
        status = client.get("/api/session/status").json()
    return status


def test_server_side_file_source_ingests_frames(sample_clip):
    with TestClient(create_app()) as client:
        started = client.post("/api/session/start", json={"source_type": "file", "source_uri": sample_clip.name})
        assert started.status_code == 200, started.text
        assert started.json()["session"]["source_type"] == "file"
        status = _await_ingestion(client, started.json(), 10)
        assert status["frames_processed"] == 10
        assert status["source"]["frames_read"] == 10
        assert status["source"]["finished"] is True
        assert status["session"]["status"] == "STOPPED"  # the runner stops at end of file


def test_configured_source_is_used_when_the_request_omits_one(sample_clip, monkeypatch):
    """`source.type` in config is server-side, so it must not be aliased to the browser."""
    monkeypatch.setenv("SOURCE_TYPE", "file")
    monkeypatch.setenv("SOURCE_URI", sample_clip.name)
    with TestClient(create_app()) as client:
        started = client.post("/api/session/start", json={})
        assert started.status_code == 200, started.text
        assert started.json()["session"]["source_type"] == "file"
        assert _await_ingestion(client, started.json(), 10)["frames_processed"] == 10
