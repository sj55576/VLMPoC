"""Live session coordinator, event de-duplication, and websocket fan-out."""
from __future__ import annotations
import time
import uuid
from datetime import datetime, timezone
from typing import Any
import numpy as np
from app.core.config import Settings
from app.sop.engine import SOPEngine
from app.sop.loader import load_sop
from app.storage.repository import Repository
from app.vision.detector import create_detector
from app.vision.pipeline import VisionPipeline
from app.vision.pose import create_pose_estimator
from app.vision.tracker import IoUTracker
from app.vlm.base import create_vlm_provider


class SessionService:
    def __init__(self, settings: Settings, root: Any) -> None:
        self.settings, self.root, self.repository = settings, root, Repository(settings.storage.database_url)
        self.sop = load_sop(root / "sop" / "example_assembly.yaml"); self.session: dict[str, Any] | None = None
        self.pipeline: VisionPipeline | None = None; self.subscribers: set[Any] = set(); self.frame_id = 0; self.recent_events: list[dict[str, Any]] = []; self._event_keys: set[tuple[str, str]] = set()

    def start(self, source_type: str = "mock", source_name: str = "synthetic") -> dict[str, Any]:
        self.session = {"id":str(uuid.uuid4()),"sop_id":self.sop.sop.id,"source_type":source_type,"source_name":source_name,"started_at":datetime.now(timezone.utc).isoformat(),"status":"RUNNING"}
        self.repository.create_session(self.session); self.frame_id = 0; self._event_keys.clear(); self.recent_events.clear()
        engine = SOPEngine(self.sop); v = self.settings.vision; vlm = self.settings.vlm
        self.pipeline = VisionPipeline(create_detector(self.settings.app.mode, v.detection_model_path, v.detection_confidence, v.device), create_pose_estimator(self.settings.app.mode, v.pose_confidence), IoUTracker(v.tracker_iou_threshold, v.missing_tolerance_seconds), engine, create_vlm_provider(vlm.provider, vlm.model, vlm.base_url, vlm.api_key), vlm.interval_seconds)
        return self.status()

    def stop(self) -> dict[str, Any]:
        if self.session: self.repository.stop_session(self.session["id"], datetime.now(timezone.utc).isoformat()); self.session["status"] = "STOPPED"
        return self.status()

    def status(self) -> dict[str, Any]:
        state = self.pipeline.engine.state if self.pipeline else None
        return {"session":self.session,"current_step":state.current.model_dump() if state and state.current else None,"progress":state.progress() if state else 0,"steps":[x.model_dump(mode="json") for x in state.steps.values()] if state else [],"vlm_calls":self.pipeline.vlm_calls if self.pipeline else 0}

    async def process_mock_frame(self, force_vlm: bool = False, frame: np.ndarray | None = None) -> dict[str, Any]:
        if not self.pipeline: self.start()
        frame = frame if frame is not None else np.zeros((480,640,3), dtype=np.uint8); start = time.perf_counter()
        calls_before = self.pipeline.vlm_calls
        obs, condition, transition = await self.pipeline.process(frame, self.frame_id, force_vlm=force_vlm); self.frame_id += 1
        if transition and self.session:
            key = (transition, condition.reason)
            if key not in self._event_keys:
                self._event_keys.add(key); event = {"event_type":transition,"step_id":self.pipeline.engine.sop.steps[self.pipeline.engine.state.index-1].id if transition == "step_completed" else self.pipeline.engine.state.current.id,"message":condition.reason,"confidence":condition.confidence,"evidence":condition.evidence}
                event["id"] = self.repository.save_event(self.session["id"], **event); self.repository.save_step(self.session["id"], self.pipeline.engine.state.steps[event["step_id"]]); self.recent_events = ([event] + self.recent_events)[:30]
        if self.session and self.pipeline.vlm_calls != calls_before:
            self.repository.save_vlm(self.session["id"], self.settings.vlm.provider, self.settings.vlm.model, self.pipeline.last_vlm_request or {}, obs.vlm_result or {}, 0.0, True)
        payload = {"type":"frame_result","timestamp":obs.timestamp.isoformat(),"fps":round(1/max(time.perf_counter()-start,.0001),2),"objects":[x.model_dump() for x in obs.objects],"poses":[x.model_dump() for x in obs.poses],"current_step":self.pipeline.engine.state.current.model_dump() if self.pipeline.engine.state.current else None,"recent_events":self.recent_events,"vlm_result":obs.vlm_result,"condition":condition.model_dump() if condition else None,"inference_ms":round((time.perf_counter()-start)*1000,2)}
        await self.broadcast(payload); return payload

    async def broadcast(self, payload: dict[str, Any]) -> None:
        stale = []
        for ws in self.subscribers:
            try: await ws.send_json(payload)
            except Exception: stale.append(ws)
        for ws in stale: self.subscribers.discard(ws)
