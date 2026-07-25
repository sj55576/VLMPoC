"""Frame pipeline that avoids VLM-per-frame and retains bounded temporal history."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from app.activity import ActivityEstimate, ActivityEstimator
from app.core.config import VLMSettings
from app.sop.engine import SOPEngine
from app.vlm.base import VLMProvider
from app.vlm.schemas import VLMEvidence, VLMResponse, unknown_response

from .detector import Detector
from .models import Observation
from .pose import PoseEstimator
from .tracker import IoUTracker


class VisionPipeline:
    def __init__(self, detector: Detector, pose: PoseEstimator, tracker: IoUTracker, engine: SOPEngine, vlm: VLMProvider, vlm_settings: VLMSettings, max_history: int = 120, capture_vlm_images: bool = False, activity_window_seconds: float = 2.0, activity_min_hold_seconds: float = 0.6, activity_enabled: bool = True, sop_enabled: bool = True) -> None:
        self.detector, self.pose, self.tracker, self.engine, self.vlm = detector, pose, tracker, engine, vlm
        self.vlm_settings = vlm_settings
        self.history, self.last_vlm_at, self.vlm_calls = deque(maxlen=max_history), None, 0
        self.capture_vlm_images = capture_vlm_images
        self.last_vlm_latency_ms: float = 0.0
        self.last_vlm: dict[str, Any] | None = None
        self.last_vlm_request: dict[str, Any] | None = None
        self.base_time = datetime.now(UTC)
        self.activity_enabled = activity_enabled
        self.sop_enabled = sop_enabled
        self.activity = ActivityEstimator(window_seconds=activity_window_seconds, min_hold_seconds=activity_min_hold_seconds)
        self.last_activity: ActivityEstimate | None = None
        self._vlm_task: asyncio.Task | None = None
        self._last_failure_at: datetime | None = None
        self.pending_vlm_records: list[dict[str, Any]] = []

    def _daily_activity_response(self, response: VLMResponse) -> VLMResponse:
        """In daily-activity mode, prefer the VLM's own read of the scene but fall back to
        the local pose/object heuristic when the VLM call failed or produced nothing usable."""
        activity = self.last_activity
        label = activity.label if activity else "unknown"
        confidence = activity.confidence if activity else 0.0
        if response.provider_success and response.detected_action not in ("", "UNKNOWN"):
            evidence = list(response.evidence)
            if activity is not None:
                evidence.append(VLMEvidence(type="activity_estimator", description=f"local estimate: {label} ({confidence:.2f})"))
            return response.model_copy(update={"evidence": evidence})
        description = f"姿勢・物体推定に基づく日常動作: {label}" if activity else "日常動作を判定できる姿勢情報がありません。"
        uncertainties = list(response.uncertainties)
        uncertainties.append("VLM result unavailable; used local activity estimate.")
        return VLMResponse(
            scene_summary=description,
            detected_action=label,
            current_step_id=None,
            step_status="UNKNOWN",
            confidence=confidence,
            safety_violation=False,
            violations=[],
            evidence=[{"type": "activity_estimator", "description": description}],
            uncertainties=uncertainties,
            provider_success=response.provider_success,
            error_message=response.error_message,
        )

    def _encode_image(self, frame: np.ndarray) -> str:
        import base64

        import cv2
        height, width = frame.shape[0], frame.shape[1]
        max_dim = self.vlm_settings.image_max_dim
        if max_dim > 0 and max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            frame = cv2.resize(frame, (max(1, int(width*scale)), max(1, int(height*scale))))
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.vlm_settings.jpeg_quality])
        return base64.b64encode(buffer).decode("ascii")

    def _build_request(self, obs: Observation, objects: list[Any], candidate: set[str], prior: set[str], now: datetime) -> dict[str, Any]:
        current = self.engine.state.current if self.sop_enabled else None
        return {
            "timestamp": now.isoformat(),
            "current_step": current.model_dump() if current else None,
            "objects": [{"class_name": x.class_name, "track_id": x.track_id, "confidence": x.confidence, "bbox_normalized": obs.normalized_bbox(x)} for x in objects],
            "poses": [x.model_dump() for x in obs.poses],
            "candidate_events": list(candidate - prior),
            "recent_steps": list(self.engine.state.completed_ids()) if self.sop_enabled else [],
            "current_activity": self.last_activity.model_dump(mode="json") if self.last_activity else None,
        }

    async def _call_vlm(self, frame: np.ndarray, request: dict[str, Any]) -> tuple[VLMResponse, float]:
        images: list[str] = []
        if self.capture_vlm_images:
            images = [self._encode_image(frame)]
        current = self.engine.state.current if self.sop_enabled else None
        started = time.perf_counter()
        response = await self.vlm.analyze(images, request, {"current_step": current.model_dump() if current else {}})
        if not self.sop_enabled:
            response = self._daily_activity_response(response)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return response, latency_ms

    def _apply_result(self, request: dict[str, Any], response: VLMResponse, latency_ms: float) -> None:
        self.last_vlm = response.model_dump()
        self.last_vlm_latency_ms = latency_ms
        self.last_vlm_request = request
        self.vlm_calls += 1
        if not response.provider_success:
            self._last_failure_at = self.last_vlm_at
        else:
            self._last_failure_at = None
        self.pending_vlm_records.append({
            "request": request,
            "response": response.model_dump(),
            "latency_ms": latency_ms,
            "success": response.provider_success,
            "error_message": response.error_message,
        })

    def _record_failure(self, request: dict[str, Any], error: Exception) -> None:
        response = unknown_response(f"VLM call raised {error.__class__.__name__}: {error}")
        if not self.sop_enabled:
            response = self._daily_activity_response(response)
        self._apply_result(request, response, 0.0)
        self._last_failure_at = self.last_vlm_at

    async def _run_vlm(self, frame: np.ndarray, request: dict[str, Any]) -> None:
        try:
            response, latency_ms = await self._call_vlm(frame, request)
            self._apply_result(request, response, latency_ms)
        except Exception as exc:
            self._record_failure(request, exc)

    def _should_trigger(self, candidate: set[str], prior: set[str], now: datetime) -> bool:
        if self._vlm_task is not None and not self._vlm_task.done():
            return False
        settings = self.vlm_settings
        if self._last_failure_at is not None and (now - self._last_failure_at).total_seconds() < settings.failure_backoff_seconds:
            return False
        if self.last_vlm_at is None:
            return True
        elapsed = (now - self.last_vlm_at).total_seconds()
        if elapsed >= settings.interval_seconds:
            return True
        return candidate != prior and elapsed >= settings.min_trigger_gap_seconds

    async def process(self, frame: np.ndarray, frame_id: int, now: datetime | None = None, force_vlm: bool = False) -> tuple[Observation, Any, str | None]:
        now = now or self.base_time + timedelta(seconds=frame_id/10)
        detections = self.detector.detect(frame, frame_id)
        # Factory-trained labels such as screwdriver/part_a are not meaningful in daily mode.
        if not self.sop_enabled:
            detections = [detection for detection in detections if detection.class_name == "person"]
        objects = self.tracker.update(detections, now)
        obs = Observation(timestamp=now, frame_id=frame_id, width=frame.shape[1], height=frame.shape[0], objects=objects, poses=self.pose.estimate(frame, frame_id), vlm_result=self.last_vlm)
        self.last_activity = self.activity.update(obs, now) if self.activity_enabled else None
        candidate = {x.class_name for x in objects}
        prior = {x.class_name for x in self.history[-1].objects} if self.history else set()

        if force_vlm:
            if self._vlm_task is not None and not self._vlm_task.done():
                await self._vlm_task
            request = self._build_request(obs, objects, candidate, prior, now)
            self.last_vlm_at = now
            try:
                response, latency_ms = await self._call_vlm(frame, request)
                self._apply_result(request, response, latency_ms)
            except Exception as exc:
                self._record_failure(request, exc)
            obs.vlm_result = self.last_vlm
        elif self._should_trigger(candidate, prior, now):
            request = self._build_request(obs, objects, candidate, prior, now)
            self.last_vlm_at = now
            self._vlm_task = asyncio.create_task(self._run_vlm(frame, request))

        result, event = self.engine.evaluate(obs, now) if self.sop_enabled else (None, None)
        self.history.append(obs)
        return obs, result, event
