from fastapi.testclient import TestClient

from app.main import create_app


def test_health_and_full_mock_flow():
    app=create_app()
    with TestClient(app) as client:
        assert client.get("/health").json()["status"]=="ok"
        assert client.post("/api/session/start",json={"source_type":"mock"}).status_code==200
        for _ in range(60): client.post("/api/vlm/analyze")
        status=client.get("/api/session/status").json(); assert status["progress"]==1
        events=client.get("/api/events").json(); assert len(events)>=4
        assert client.post("/api/session/stop").status_code==200

def test_activity_present_after_processing_frames():
    app=create_app()
    with TestClient(app) as client:
        client.post("/api/session/start",json={"source_type":"mock"})
        payload=client.post("/api/vlm/analyze").json()
        assert "activity" in payload
        assert payload["activity"] is None or "label" in payload["activity"]
        response=client.get("/api/activity")
        assert response.status_code==200
        body=response.json()
        assert "current" in body and "history" in body


def test_daily_activity_mode_disables_sop(monkeypatch):
    monkeypatch.setenv("SOP_ENABLED", "false")
    app = create_app()
    with TestClient(app) as client:
        started = client.post("/api/session/start", json={"source_type": "mock"}).json()
        assert started["session"]["sop_id"] == "daily_activity"
        assert started["current_step"] is None
        assert started["steps"] == []
        frame = client.post("/api/vlm/analyze").json()
        assert frame["current_step"] is None
        assert frame["condition"] is None


def test_websocket_frame():
    app=create_app()
    with TestClient(app) as client:
        client.post("/api/session/start")
        with client.websocket_connect("/api/ws") as ws:
            result=ws.receive_json(); assert result["type"]=="frame_result"


def test_sop_catalog_and_invalid_selection():
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/api/sops").json()[0]["id"] == "example_assembly"
        response = client.post("/api/session/start", json={"sop_id": "missing"})
        assert response.status_code == 422
