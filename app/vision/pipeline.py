"""Frame pipeline that avoids VLM-per-frame and retains bounded temporal history."""
from __future__ import annotations
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any
import numpy as np
from app.activity import ActivityEstimate, ActivityEstimator
from app.sop.engine import SOPEngine
from app.vlm.base import VLMProvider
from .detector import Detector
from .models import Observation
from .pose import PoseEstimator
from .tracker import IoUTracker


class VisionPipeline:
    def __init__(self, detector: Detector, pose: PoseEstimator, tracker: IoUTracker, engine: SOPEngine, vlm: VLMProvider, interval_seconds: float, max_history: int = 120, max_images: int = 1, capture_vlm_images: bool = False, activity_window_seconds: float = 2.0, activity_min_hold_seconds: float = 0.6, activity_enabled: bool = True) -> None:
        self.detector, self.pose, self.tracker, self.engine, self.vlm = detector, pose, tracker, engine, vlm
        self.interval_seconds, self.history, self.last_vlm_at, self.vlm_calls = interval_seconds, deque(maxlen=max_history), None, 0
        self.max_images, self.capture_vlm_images = max_images, capture_vlm_images
        self.last_vlm_latency_ms: float = 0.0
        self.last_vlm: dict[str, Any] | None = None
        self.last_vlm_request: dict[str, Any] | None = None
        self.base_time = datetime.now(timezone.utc)
        self.activity_enabled = activity_enabled
        self.activity = ActivityEstimator(window_seconds=activity_window_seconds, min_hold_seconds=activity_min_hold_seconds)
        self.last_activity: ActivityEstimate | None = None

    async def process(self, frame: np.ndarray, frame_id: int, now: datetime | None = None, force_vlm: bool = False) -> tuple[Observation, Any, str | None]:
        now = now or self.base_time + timedelta(seconds=frame_id/10)
        objects = self.tracker.update(self.detector.detect(frame, frame_id), now)
        obs = Observation(timestamp=now, frame_id=frame_id, width=frame.shape[1], height=frame.shape[0], objects=objects, poses=self.pose.estimate(frame, frame_id), vlm_result=self.last_vlm)
        self.last_activity = self.activity.update(obs, now) if self.activity_enabled else None
        # Trigger only on interval, candidate transition, or manual request.
        candidate = {x.class_name for x in objects}
        prior = {x.class_name for x in self.history[-1].objects} if self.history else set()
        should_vlm = force_vlm or self.last_vlm_at is None or (now-self.last_vlm_at).total_seconds() >= self.interval_seconds or candidate != prior
        if should_vlm:
            current = self.engine.state.current
            request = {"timestamp": now.isoformat(), "current_step": current.model_dump() if current else None, "objects": [{"class_name":x.class_name,"track_id":x.track_id,"confidence":x.confidence,"bbox_normalized":obs.normalized_bbox(x)} for x in objects], "poses":[x.model_dump() for x in obs.poses], "candidate_events":list(candidate-prior), "recent_steps":list(self.engine.state.completed_ids()), "current_activity": self.last_activity.model_dump(mode="json") if self.last_activity else None}
            images: list[str] = []
            if self.capture_vlm_images and self.max_images > 0:
                import base64
                import cv2
                _, buffer = cv2.imencode(".jpg", frame)
                images = [base64.b64encode(buffer).decode("ascii")]
            started = time.perf_counter()
            response = await self.vlm.analyze(images, request, {"current_step": current.model_dump() if current else {}})
            self.last_vlm_latency_ms = round((time.perf_counter()-started)*1000, 2)
            self.last_vlm_request = request
            self.last_vlm, self.last_vlm_at, self.vlm_calls = response.model_dump(), now, self.vlm_calls+1; obs.vlm_result = self.last_vlm
        result, event = self.engine.evaluate(obs, now)
        self.history.append(obs)
        return obs, result, event
