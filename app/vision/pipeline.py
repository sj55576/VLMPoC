"""Frame pipeline that avoids VLM-per-frame and retains bounded temporal history."""
from __future__ import annotations
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any
import numpy as np
from app.sop.engine import SOPEngine
from app.vlm.base import VLMProvider
from .detector import Detector
from .models import Observation
from .pose import PoseEstimator
from .tracker import IoUTracker


class VisionPipeline:
    def __init__(self, detector: Detector, pose: PoseEstimator, tracker: IoUTracker, engine: SOPEngine, vlm: VLMProvider, interval_seconds: float, max_history: int = 120) -> None:
        self.detector, self.pose, self.tracker, self.engine, self.vlm = detector, pose, tracker, engine, vlm
        self.interval_seconds, self.history, self.last_vlm_at, self.vlm_calls = interval_seconds, deque(maxlen=max_history), None, 0
        self.last_vlm: dict[str, Any] | None = None
        self.base_time = datetime.now(timezone.utc)

    async def process(self, frame: np.ndarray, frame_id: int, now: datetime | None = None, force_vlm: bool = False) -> tuple[Observation, Any, str | None]:
        now = now or self.base_time + timedelta(seconds=frame_id/10)
        objects = self.tracker.update(self.detector.detect(frame, frame_id), now)
        obs = Observation(timestamp=now, frame_id=frame_id, width=frame.shape[1], height=frame.shape[0], objects=objects, poses=self.pose.estimate(frame, frame_id), vlm_result=self.last_vlm)
        # Trigger only on interval, candidate transition, or manual request.
        candidate = {x.class_name for x in objects}
        prior = {x.class_name for x in self.history[-1].objects} if self.history else set()
        should_vlm = force_vlm or self.last_vlm_at is None or (now-self.last_vlm_at).total_seconds() >= self.interval_seconds or candidate != prior
        if should_vlm:
            current = self.engine.state.current
            request = {"timestamp": now.isoformat(), "current_step": current.model_dump() if current else None, "objects": [{"class_name":x.class_name,"track_id":x.track_id,"confidence":x.confidence,"bbox_normalized":obs.normalized_bbox(x)} for x in objects], "poses":[x.model_dump() for x in obs.poses], "candidate_events":list(candidate-prior), "recent_steps":list(self.engine.state.completed_ids())}
            response = await self.vlm.analyze([], request, {"current_step": current.model_dump() if current else {}})
            self.last_vlm, self.last_vlm_at, self.vlm_calls = response.model_dump(), now, self.vlm_calls+1; obs.vlm_result = self.last_vlm
        result, event = self.engine.evaluate(obs, now)
        self.history.append(obs)
        return obs, result, event
