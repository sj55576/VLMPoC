"""Tests for safety-violation alert events emitted from VLM records."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import load_settings
from app.services.session import SessionService
from app.storage.repository import Repository
from app.vlm.base import VLMProvider
from app.vlm.schemas import VLMResponse

ROOT = Path(__file__).resolve().parents[2]
FRAME = np.zeros((10, 10, 3), dtype=np.uint8)


class SafetyViolationVLM(VLMProvider):
    async def analyze(self, images: list[Any], observation: dict[str, Any], sop_context: dict[str, Any]) -> VLMResponse:
        return VLMResponse(scene_summary="unsafe scene", confidence=0.8, safety_violation=True, violations=["no_helmet"])


def _service(tmp_path) -> SessionService:
    settings = load_settings(ROOT)
    settings.storage.database_url = f"sqlite:///{tmp_path}/svc.db"
    return SessionService(settings, ROOT)


def test_safety_violation_emits_critical_event(tmp_path) -> None:
    service = _service(tmp_path)
    service.start(source_type="mock", source_name="synthetic")
    service.pipeline.vlm = SafetyViolationVLM()

    asyncio.run(service.process_frame(force_vlm=True))

    events = service.repository.events(service.session["id"])
    safety_events = [event for event in events if event["event_type"] == "safety_violation"]
    assert len(safety_events) == 1
    assert safety_events[0]["severity"] == "CRITICAL"
    assert "no_helmet" in safety_events[0]["message"]
    recent_safety = [event for event in service.recent_events if event["event_type"] == "safety_violation"]
    assert recent_safety and recent_safety[0]["severity"] == "CRITICAL"


def test_safety_violation_dedup_same_violation_set(tmp_path) -> None:
    service = _service(tmp_path)
    service.start(source_type="mock", source_name="synthetic")
    service.pipeline.vlm = SafetyViolationVLM()

    asyncio.run(service.process_frame(force_vlm=True))
    asyncio.run(service.process_frame(force_vlm=True))

    events = service.repository.events(service.session["id"])
    safety_events = [event for event in events if event["event_type"] == "safety_violation"]
    assert len(safety_events) == 1


def test_repository_save_event_round_trips_critical_severity(tmp_path) -> None:
    repo = Repository(f"sqlite:///{tmp_path}/r.db")
    repo.create_session({"id": "s1", "sop_id": "sop", "source_type": "mock", "source_name": "src", "started_at": "2026-01-01T00:00:00", "status": "RUNNING"})
    event_id = repo.save_event("s1", event_type="safety_violation", step_id="", message="safety violation: no_helmet", confidence=0.8, evidence={"violations": ["no_helmet"]}, severity="CRITICAL")
    assert repo.event(event_id)["severity"] == "CRITICAL"
