"""Live session coordinator, event de-duplication, and websocket fan-out."""
from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from app.core.config import Settings
from app.sop.engine import SOPEngine
from app.sop.loader import load_sop
from app.sop.models import SOPDefinition
from app.storage.repository import Repository
from app.vision.detector import create_detector
from app.vision.pipeline import VisionPipeline
from app.vision.pose import create_pose_estimator
from app.vision.tracker import IoUTracker
from app.vlm.base import create_vlm_provider


class SessionService:
    def __init__(self, settings: Settings, root: Any) -> None:
        self.settings, self.root, self.repository = settings, root, Repository(settings.storage.database_url)
        self.repository.purge_older_than(settings.storage.retention_days)
        self.sops = self._load_sops()
        self.sop = self.sops["example_assembly"]
        self.session: dict[str, Any] | None = None
        self.pipeline: VisionPipeline | None = None; self.subscribers: set[Any] = set(); self.frame_id = 0; self.recent_events: list[dict[str, Any]] = []; self._event_keys: set[tuple[str, str]] = set()
        self._last_activity_label: str | None = None
        self.stream_subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._process_lock = asyncio.Lock()
        self._runner_task: asyncio.Task[None] | None = None
        self._source_started_at: datetime | None = None
        self._last_frame: np.ndarray | None = None

    def _load_sops(self) -> dict[str, SOPDefinition]:
        """Load every validated SOP YAML from the application SOP directory."""
        loaded = {sop.sop.id: sop for path in sorted((self.root / "sop").glob("*.yaml")) for sop in [load_sop(path)]}
        if not loaded:
            raise ValueError("No valid SOP YAML files were found")
        return loaded

    def list_sops(self) -> list[dict[str, str]]:
        return [{"id": item.sop.id, "name": item.sop.name, "version": item.sop.version} for item in self.sops.values()]

    def reload_sops(self) -> list[dict[str, str]]:
        """Reload SOP definitions; an active session retains its original state machine."""
        self.sops = self._load_sops()
        if not self.session:
            self.sop = self.sops.get(self.sop.sop.id, next(iter(self.sops.values())))
        return self.list_sops()

    def start(self, source_type: str = "mock", source_name: str = "synthetic", sop_id: str | None = None) -> dict[str, Any]:
        if self.session and self.session["status"] == "RUNNING":
            self.stop()
        if sop_id:
            try:
                self.sop = self.sops[sop_id]
            except KeyError as exc:
                raise ValueError(f"Unknown SOP id: {sop_id}") from exc
        self.session = {"id":str(uuid.uuid4()),"sop_id":self.sop.sop.id if self.settings.sop.enabled else "daily_activity","source_type":source_type,"source_name":source_name,"started_at":datetime.now(UTC).isoformat(),"status":"RUNNING"}
        self.repository.create_session(self.session); self.frame_id = 0; self._event_keys.clear(); self.recent_events.clear(); self._source_started_at = datetime.now(UTC); self._last_activity_label = None; self._last_frame = None
        engine = SOPEngine(self.sop); v = self.settings.vision; vlm = self.settings.vlm; activity = self.settings.activity
        self.pipeline = VisionPipeline(
            create_detector(self.settings.app.mode, v.detection_model_path, v.detection_confidence, v.device, getattr(v, "class_aliases", {})),
            create_pose_estimator(self.settings.app.mode, v.pose_confidence, v.pose_model_path),
            IoUTracker(v.tracker_iou_threshold, v.missing_tolerance_seconds),
            engine,
            create_vlm_provider(vlm),
            vlm_settings=vlm,
            capture_vlm_images=vlm.provider != "mock",
            activity_window_seconds=activity.window_seconds,
            activity_min_hold_seconds=activity.min_hold_seconds,
            activity_enabled=activity.enabled,
            sop_enabled=self.settings.sop.enabled,
        )
        return self.status()

    def stop(self) -> dict[str, Any]:
        if self.session and self.session["status"] == "RUNNING":
            self.repository.stop_session(self.session["id"], datetime.now(UTC).isoformat()); self.session["status"] = "STOPPED"
        if self._runner_task and not self._runner_task.done():
            self._runner_task.cancel()
        self._runner_task = None
        return self.status()

    def status(self) -> dict[str, Any]:
        state = self.pipeline.engine.state if self.pipeline and self.settings.sop.enabled else None
        return {"session":self.session,"current_step":state.current.model_dump() if state and state.current else None,"progress":state.progress() if state else 0,"steps":[x.model_dump(mode="json") for x in state.steps.values()] if state else [],"vlm_calls":self.pipeline.vlm_calls if self.pipeline else 0}

    def ensure_runner(self) -> None:
        """Start exactly one mock producer; clients only consume its broadcasts."""
        if not self.session or self.session["status"] != "RUNNING" or self.session["source_type"] != "mock":
            return
        if self._runner_task is None or self._runner_task.done():
            self._runner_task = asyncio.create_task(self._run_mock_source())

    async def _run_mock_source(self) -> None:
        while self.session and self.session["status"] == "RUNNING":
            await self.process_mock_frame()
            await asyncio.sleep(0.1)

    def subscribe_stream(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=2)
        self.stream_subscribers.add(queue)
        return queue

    def unsubscribe_stream(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self.stream_subscribers.discard(queue)

    async def process_mock_frame(self, force_vlm: bool = False, frame: np.ndarray | None = None) -> dict[str, Any]:
        if not self.pipeline:
            self.start()
        if not self.session or self.session["status"] != "RUNNING":
            raise RuntimeError("No active session. Start a session before processing frames.")
        async with self._process_lock:
            return await self._process_frame(force_vlm, frame)

    async def _process_frame(self, force_vlm: bool, frame: np.ndarray | None) -> dict[str, Any]:
        if frame is not None:
            self._last_frame = frame.copy()
        frame = frame if frame is not None else self._last_frame if self._last_frame is not None else np.zeros((480,640,3), dtype=np.uint8)
        start = time.perf_counter()
        simulated_now = self._source_started_at + timedelta(seconds=self.frame_id / 10) if self.settings.app.mode == "mock" and self._source_started_at else None
        obs, condition, transition = await self.pipeline.process(frame, self.frame_id, now=simulated_now, force_vlm=force_vlm); self.frame_id += 1
        if transition and self.session:
            key = (transition, condition.reason)
            if key not in self._event_keys:
                self._event_keys.add(key); event = {"event_type":transition,"step_id":self.pipeline.engine.sop.steps[self.pipeline.engine.state.index-1].id if transition == "step_completed" else self.pipeline.engine.state.current.id,"message":condition.reason,"confidence":condition.confidence,"evidence":condition.evidence}
                event["id"] = self.repository.save_event(self.session["id"], **event); self.repository.save_step(self.session["id"], self.pipeline.engine.state.steps[event["step_id"]]); self.recent_events = [event, *self.recent_events][:30]
        if self.session and self.pipeline.pending_vlm_records:
            records, self.pipeline.pending_vlm_records = self.pipeline.pending_vlm_records, []
            for record in records:
                self.repository.save_vlm(self.session["id"], self.settings.vlm.provider, self.settings.vlm.model, record["request"], record["response"], record["latency_ms"], record["success"], record["error_message"])
                response = record["response"]
                if record["success"] and response.get("safety_violation"):
                    key = ("safety_violation", "|".join(sorted(response.get("violations") or [])))
                    if key not in self._event_keys:
                        self._event_keys.add(key)
                        message = "safety violation: " + (", ".join(response.get("violations") or []) or "unspecified")
                        step_id = self.pipeline.engine.state.current.id if self.settings.sop.enabled and self.pipeline.engine.state.current else ""
                        confidence = float(response.get("confidence") or 0.0)
                        evidence = {"violations": response.get("violations") or [], "scene_summary": response.get("scene_summary")}
                        event = {"event_type":"safety_violation","severity":"CRITICAL","step_id":step_id,"message":message,"confidence":confidence,"evidence":evidence}
                        event["id"] = self.repository.save_event(self.session["id"], event_type="safety_violation", step_id=step_id, message=message, confidence=confidence, evidence=evidence, severity="CRITICAL")
                        self.recent_events = [event, *self.recent_events][:30]
        activity = self.pipeline.last_activity
        if self.session and activity and activity.label != self._last_activity_label:
            message = f"activity: {self._last_activity_label or 'unknown'} -> {activity.label}"
            event = {"event_type":"activity_changed","step_id":self.pipeline.engine.state.current.id if self.settings.sop.enabled and self.pipeline.engine.state.current else "","message":message,"confidence":activity.confidence,"evidence":activity.evidence}
            event["id"] = self.repository.save_event(self.session["id"], **event); self.recent_events = [event, *self.recent_events][:30]
            self._last_activity_label = activity.label
        payload = {"type":"frame_result","timestamp":obs.timestamp.isoformat(),"fps":round(1/max(time.perf_counter()-start,.0001),2),"objects":[x.model_dump() for x in obs.objects],"poses":[x.model_dump() for x in obs.poses],"current_step":self.pipeline.engine.state.current.model_dump() if self.settings.sop.enabled and self.pipeline.engine.state.current else None,"recent_events":self.recent_events,"vlm_result":obs.vlm_result,"condition":condition.model_dump() if condition else None,"inference_ms":round((time.perf_counter()-start)*1000,2),"vlm_calls":self.pipeline.vlm_calls,"activity":activity.model_dump(mode="json") if activity else None}
        await self.broadcast(payload); return payload

    async def broadcast(self, payload: dict[str, Any]) -> None:
        stale = []
        for ws in self.subscribers:
            try: await ws.send_json(payload)
            except Exception: stale.append(ws)
        for ws in stale: self.subscribers.discard(ws)
        for queue in self.stream_subscribers:
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty): queue.get_nowait()
            queue.put_nowait(payload)
