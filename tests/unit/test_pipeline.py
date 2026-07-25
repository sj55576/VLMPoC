"""Tests for VisionPipeline's non-blocking VLM triggering and daily-activity merge."""
from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from app.activity import ActivityEstimate
from app.core.config import VLMSettings
from app.sop.engine import SOPEngine
from app.sop.models import SOPDefinition, SOPMeta, StepDefinition
from app.vision.detector import Detector
from app.vision.models import Detection, Pose
from app.vision.pipeline import VisionPipeline
from app.vision.pose import PoseEstimator
from app.vision.tracker import IoUTracker
from app.vlm.base import VLMProvider
from app.vlm.schemas import VLMResponse

FRAME = np.zeros((10, 10, 3), dtype=np.uint8)


class ListDetector(Detector):
    def __init__(self, objects: list[Detection]) -> None:
        self.objects = objects

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        return self.objects


class NoPose(PoseEstimator):
    def estimate(self, frame: np.ndarray, frame_id: int) -> list[Pose]:
        return []


class SlowVLM(VLMProvider):
    """Simulates a slow VLM backend that should never block frame processing."""

    async def analyze(self, images: list[Any], observation: dict[str, Any], sop_context: dict[str, Any]) -> VLMResponse:
        await asyncio.sleep(0.2)
        return VLMResponse(scene_summary="slow result", detected_action="walking", confidence=0.9)


class CountingVLM(VLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def analyze(self, images: list[Any], observation: dict[str, Any], sop_context: dict[str, Any]) -> VLMResponse:
        self.calls += 1
        return VLMResponse(scene_summary="ok", detected_action=f"action{self.calls}", confidence=0.5)


class FailingVLM(VLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def analyze(self, images: list[Any], observation: dict[str, Any], sop_context: dict[str, Any]) -> VLMResponse:
        self.calls += 1
        return VLMResponse(scene_summary="fail", confidence=0.0, provider_success=False, error_message="boom")


def _trivial_engine() -> SOPEngine:
    sop = SOPDefinition(sop=SOPMeta(id="test", name="test", version="1"), steps=[
        StepDefinition(id="one", name="one", conditions={"type": "object_present", "object": "person"}),
    ])
    return SOPEngine(sop)


def make_pipeline(vlm: VLMProvider, vlm_settings: VLMSettings | None = None, sop_enabled: bool = True) -> tuple[VisionPipeline, ListDetector]:
    settings = vlm_settings or VLMSettings(provider="mock", interval_seconds=100.0, min_trigger_gap_seconds=1.0, failure_backoff_seconds=5.0)
    detector = ListDetector([Detection(class_name="person", confidence=0.9, bbox=(0, 0, 10, 10))])
    pipeline = VisionPipeline(
        detector,
        NoPose(),
        IoUTracker(0.25, 5.0),
        _trivial_engine(),
        vlm,
        settings,
        capture_vlm_images=False,
        activity_enabled=False,
        sop_enabled=sop_enabled,
    )
    return pipeline, detector


def test_process_does_not_block_on_slow_vlm() -> None:
    async def scenario() -> None:
        pipeline, _ = make_pipeline(SlowVLM())
        start = datetime(2026, 1, 1, tzinfo=UTC)

        t0 = time.perf_counter()
        obs, _, _ = await pipeline.process(FRAME, 0, now=start)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.15
        assert obs.vlm_result is None  # background call has not completed yet

        await asyncio.sleep(0.3)  # let the background task finish

        obs2, _, _ = await pipeline.process(FRAME, 1, now=start + timedelta(seconds=0.1))
        assert obs2.vlm_result is not None
        assert obs2.vlm_result["detected_action"] == "walking"

    asyncio.run(scenario())


def test_force_vlm_awaits_and_returns_fresh_result() -> None:
    async def scenario() -> None:
        vlm = CountingVLM()
        pipeline, _ = make_pipeline(vlm)
        start = datetime(2026, 1, 1, tzinfo=UTC)

        obs, _, _ = await pipeline.process(FRAME, 0, now=start, force_vlm=True)

        assert vlm.calls == 1
        assert obs.vlm_result["detected_action"] == "action1"

    asyncio.run(scenario())


def test_object_change_respects_min_trigger_gap() -> None:
    async def scenario() -> None:
        vlm = CountingVLM()
        settings = VLMSettings(provider="mock", interval_seconds=100.0, min_trigger_gap_seconds=1.0, failure_backoff_seconds=5.0)
        pipeline, detector = make_pipeline(vlm, vlm_settings=settings)
        start = datetime(2026, 1, 1, tzinfo=UTC)

        # Initial frame always triggers (last_vlm_at is None).
        await pipeline.process(FRAME, 0, now=start)
        await asyncio.sleep(0.01)
        assert vlm.calls == 1

        # Object set changes well within the gap -> suppressed.
        detector.objects = [Detection(class_name="cup", confidence=0.9, bbox=(0, 0, 5, 5))]
        await pipeline.process(FRAME, 1, now=start + timedelta(seconds=0.5))
        await asyncio.sleep(0.01)
        assert vlm.calls == 1

        # No further change, still within the gap -> still suppressed.
        await pipeline.process(FRAME, 2, now=start + timedelta(seconds=0.6))
        await asyncio.sleep(0.01)
        assert vlm.calls == 1

        # Object set changes again after the gap has elapsed -> triggers.
        detector.objects = [Detection(class_name="person", confidence=0.9, bbox=(0, 0, 10, 10))]
        await pipeline.process(FRAME, 3, now=start + timedelta(seconds=1.5))
        await asyncio.sleep(0.01)
        assert vlm.calls == 2

    asyncio.run(scenario())


def test_failure_backoff_suppresses_auto_trigger_but_not_force_vlm() -> None:
    async def scenario() -> None:
        vlm = FailingVLM()
        settings = VLMSettings(provider="mock", interval_seconds=100.0, min_trigger_gap_seconds=0.1, failure_backoff_seconds=5.0)
        pipeline, detector = make_pipeline(vlm, vlm_settings=settings)
        start = datetime(2026, 1, 1, tzinfo=UTC)

        await pipeline.process(FRAME, 0, now=start)
        await asyncio.sleep(0.01)
        assert vlm.calls == 1
        assert pipeline.last_vlm["provider_success"] is False

        # Object set changes after min_trigger_gap but well within the failure backoff window.
        detector.objects = [Detection(class_name="cup", confidence=0.9, bbox=(0, 0, 5, 5))]
        await pipeline.process(FRAME, 1, now=start + timedelta(seconds=1))
        await asyncio.sleep(0.01)
        assert vlm.calls == 1  # still suppressed by backoff

        # force_vlm bypasses the backoff entirely.
        obs, _, _ = await pipeline.process(FRAME, 2, now=start + timedelta(seconds=1.2), force_vlm=True)
        assert vlm.calls == 2
        assert obs.vlm_result["provider_success"] is False

    asyncio.run(scenario())


def test_daily_mode_merge_keeps_vlm_action_on_success() -> None:
    pipeline, _ = make_pipeline(CountingVLM(), sop_enabled=False)
    pipeline.last_activity = ActivityEstimate(label="drinking", confidence=0.7, person_id=1, since=None, duration_seconds=0.0, evidence={})
    response = VLMResponse(scene_summary="user drinking coffee", detected_action="drinking_coffee", confidence=0.9)

    merged = pipeline._daily_activity_response(response)

    assert merged.provider_success is True
    assert merged.detected_action == "drinking_coffee"
    assert any(entry.description.startswith("local estimate") for entry in merged.evidence)


def test_daily_mode_merge_falls_back_to_estimator_on_failure() -> None:
    pipeline, _ = make_pipeline(CountingVLM(), sop_enabled=False)
    pipeline.last_activity = ActivityEstimate(label="standing", confidence=0.6, person_id=1, since=None, duration_seconds=0.0, evidence={})
    response = VLMResponse(scene_summary="failed", confidence=0.0, provider_success=False, error_message="timeout")

    merged = pipeline._daily_activity_response(response)

    assert merged.detected_action == "standing"
    assert merged.provider_success is False
    assert merged.error_message == "timeout"
    assert "VLM result unavailable; used local activity estimate." in merged.uncertainties


def test_daily_mode_merge_falls_back_when_vlm_action_unknown() -> None:
    pipeline, _ = make_pipeline(CountingVLM(), sop_enabled=False)
    pipeline.last_activity = ActivityEstimate(label="sitting", confidence=0.5, person_id=1, since=None, duration_seconds=0.0, evidence={})
    response = VLMResponse(scene_summary="nothing clear", detected_action="UNKNOWN", confidence=0.4)

    merged = pipeline._daily_activity_response(response)

    assert merged.detected_action == "sitting"
