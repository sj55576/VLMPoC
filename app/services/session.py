"""Live session coordinator, event de-duplication, and websocket fan-out."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from app import __version__
from app.core.config import Settings
from app.sop.engine import SOPEngine
from app.sop.loader import load_sop
from app.sop.models import SOPDefinition
from app.storage.frames import FrameStore
from app.storage.repository import Repository
from app.vision.detector import create_detector
from app.vision.pipeline import VisionPipeline
from app.vision.pose import create_pose_estimator
from app.vision.source import (
    FrameSource,
    FrameSourceError,
    SourceSpec,
    create_frame_source,
    resolve_source_uri,
)
from app.vision.tracker import IoUTracker
from app.vlm.base import create_vlm_provider

LOGGER = logging.getLogger(__name__)

# Frames pushed by the browser keep the original request values; server-side ingestion
# uses its own names so a cached dashboard can never make the server open a webcam.
BROWSER_SOURCE_ALIASES = {"camera": "browser", "video": "browser", "browser": "browser"}
SERVER_SOURCE_TYPES = {"server_camera": "camera", "file": "file", "rtsp": "rtsp"}
# config `source.type` is server-side by definition, so its "camera" is the host webcam.
CONFIGURED_SOURCE_NAMES = {"mock": "mock", "camera": "server_camera", "file": "file", "rtsp": "rtsp"}


class SessionService:
    def __init__(self, settings: Settings, root: Any) -> None:
        self.settings, self.root = settings, root
        self.frame_store = FrameStore(settings.storage.frame_dir, enabled=settings.storage.save_event_frames, jpeg_quality=settings.storage.frame_jpeg_quality, max_dim=settings.storage.frame_max_dim)
        self.repository = Repository(settings.storage.database_url, self.frame_store)
        self.repository.purge_older_than(settings.storage.retention_days)
        self.sops = self._load_sops()
        self.sop = self.sops["example_assembly"]
        self.session: dict[str, Any] | None = None
        self.pipeline: VisionPipeline | None = None; self.subscribers: set[Any] = set(); self.frame_id = 0; self.recent_events: list[dict[str, Any]] = []; self._event_keys: set[tuple[str, str]] = set()
        self._last_activity_label: str | None = None
        self.stream_subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._process_lock = asyncio.Lock()
        self._runner_task: asyncio.Task[None] | None = None
        self._retention_task: asyncio.Task[None] | None = None
        self._source_started_at: datetime | None = None
        self._last_frame: np.ndarray | None = None
        self._source: FrameSource | None = None
        self._source_spec: SourceSpec | None = None
        self.frames_processed = 0
        self.source_error: str | None = None
        self._closing_tasks: set[asyncio.Task[None]] = set()

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

    def _resolve_source(self, source_type: str | None, source_uri: str | None) -> tuple[str, SourceSpec | None]:
        """Map a requested source onto a capture spec.

        Returns the canonical session source type and the spec to capture from, or
        ``None`` when frames arrive from the browser instead of the server.
        """
        if source_type:
            requested = source_type.strip()
            kind = BROWSER_SOURCE_ALIASES.get(requested, requested)
        else:
            # No explicit request: fall back to the configured server-side default.
            kind = CONFIGURED_SOURCE_NAMES.get((self.settings.source.type or "mock").strip(), "mock")
            requested = kind
        if kind == "browser":
            return kind, None
        spec_type = "mock" if kind == "mock" else SERVER_SOURCE_TYPES.get(kind)
        if spec_type is None:
            raise ValueError(f"Unknown source_type: {requested}. Use one of mock, browser, {', '.join(sorted(SERVER_SOURCE_TYPES))}.")
        source = self.settings.source
        try:
            uri = resolve_source_uri(spec_type, source_uri if source_uri is not None else source.uri, self.root / "data")
        except FrameSourceError as exc:
            raise ValueError(str(exc)) from exc
        return kind, SourceSpec(type=spec_type, uri=uri, target_fps=source.target_fps, queue_size=source.queue_size, reconnect_seconds=source.reconnect_seconds, max_reconnect_attempts=source.max_reconnect_attempts, loop_file=source.loop_file)

    def start(self, source_type: str | None = None, source_name: str = "synthetic", sop_id: str | None = None, source_uri: str | None = None) -> dict[str, Any]:
        if self.session and self.session["status"] == "RUNNING":
            self.stop()
        if sop_id:
            try:
                self.sop = self.sops[sop_id]
            except KeyError as exc:
                raise ValueError(f"Unknown SOP id: {sop_id}") from exc
        kind, spec = self._resolve_source(source_type, source_uri)
        self._source_spec, self.source_error = spec, None
        self.session = {"id":str(uuid.uuid4()),"sop_id":self.sop.sop.id if self.settings.sop.enabled else "daily_activity","source_type":kind,"source_name":source_name,"source_uri":spec.uri if spec else "","started_at":datetime.now(UTC).isoformat(),"status":"RUNNING"}
        self.repository.create_session(self.session); self.frame_id = 0; self._event_keys.clear(); self.recent_events.clear(); self._source_started_at = datetime.now(UTC); self._last_activity_label = None; self._last_frame = None
        engine = SOPEngine(self.sop, self.settings.vlm.max_result_age_seconds); v = self.settings.vision; vlm = self.settings.vlm; activity = self.settings.activity
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
        self._detach_source()
        return self.status()

    def _detach_source(self) -> None:
        """Release the capture source from synchronous code without blocking the event loop."""
        source, self._source = self._source, None
        if source is None:
            return
        source.abort()
        try:
            task = asyncio.get_running_loop().create_task(source.close())
        except RuntimeError:  # no running loop: the daemon thread exits on the abort flag
            return
        self._closing_tasks.add(task)
        task.add_done_callback(self._closing_tasks.discard)

    async def shutdown(self) -> None:
        """Stop background work and join the capture thread; used by the app lifespan."""
        for task in (self._runner_task, self._retention_task):
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._runner_task = self._retention_task = None
        source, self._source = self._source, None
        if source is not None:
            await source.close()
        if self._closing_tasks:
            await asyncio.gather(*list(self._closing_tasks), return_exceptions=True)

    def status(self) -> dict[str, Any]:
        state = self.pipeline.engine.state if self.pipeline and self.settings.sop.enabled else None
        source = {"type": self._source_spec.type if self._source_spec else "browser", "uri": self._source_spec.uri if self._source_spec else "", "error": self.source_error, **(self._source_spec.stats.as_dict() if self._source_spec else {})}
        return {"session":self.session,"current_step":state.current.model_dump() if state and state.current else None,"progress":state.progress() if state else 0,"steps":[x.model_dump(mode="json") for x in state.steps.values()] if state else [],"vlm_calls":self.pipeline.vlm_calls if self.pipeline else 0,"frames_processed":self.frames_processed,"source":source}

    @property
    def has_subscribers(self) -> bool:
        return bool(self.subscribers or self.stream_subscribers)

    async def ensure_runner(self) -> None:
        """Start exactly one producer for the active session; clients only consume broadcasts."""
        if not self.session or self.session["status"] != "RUNNING" or self._source_spec is None:
            return
        if self._runner_task is not None and not self._runner_task.done():
            return
        if self._source is None:
            source = create_frame_source(self._source_spec)
            try:
                await source.open()
            except FrameSourceError as exc:
                self.source_error = str(exc)
                raise ValueError(str(exc)) from exc
            self._source = source
        # Mock frames only pace the loop; the mock detector synthesises objects from frame ids
        # and an uploaded still image must stay the analysed frame.
        self._runner_task = asyncio.create_task(self._run_source(self._source, deliver_frames=self._source_spec.type != "mock"))

    async def _run_source(self, source: FrameSource, deliver_frames: bool) -> None:
        try:
            while self.session and self.session["status"] == "RUNNING":
                frame = await source.read()
                if frame is None:
                    LOGGER.info("Frame source finished; stopping session %s", self.session["id"] if self.session else "-")
                    self.stop()
                    return
                await self.process_frame(frame=frame if deliver_frames else None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a runner crash must not take the process down silently
            LOGGER.exception("Frame runner failed")
            self.source_error = f"{exc.__class__.__name__}: {exc}"

    async def run_retention(self) -> None:
        """Purge expired rows and evidence frames on an interval, not only at startup."""
        interval = self.settings.storage.retention_interval_minutes
        if interval <= 0 or self.settings.storage.retention_days <= 0:
            return
        while True:
            await asyncio.sleep(interval * 60)
            try:
                deleted = await asyncio.to_thread(self.repository.purge_older_than, self.settings.storage.retention_days)
                if any(deleted.values()):
                    LOGGER.info("Retention purge removed %s", deleted)
            except Exception:  # a purge failure must never end the loop
                LOGGER.exception("Retention purge failed")

    def metrics_snapshot(self) -> dict[str, Any]:
        """Flatten runtime counters and stored aggregates for the Prometheus endpoint."""
        state = self.pipeline.engine.state if self.pipeline and self.settings.sop.enabled else None
        stats = self._source_spec.stats if self._source_spec else None
        return {
            "version": __version__, "mode": self.settings.app.mode, "vlm_provider": self.settings.vlm.provider,
            "session_active": bool(self.session and self.session["status"] == "RUNNING"),
            "frames_processed": self.frames_processed,
            "vlm_calls": self.pipeline.vlm_calls if self.pipeline else 0,
            "vlm_latency_ms_last": self.pipeline.last_vlm_latency_ms if self.pipeline else 0.0,
            "websocket_subscribers": len(self.subscribers),
            "stream_subscribers": len(self.stream_subscribers),
            "source_reconnects": stats.reconnects if stats else 0,
            "sop_progress": state.progress() if state else 0.0,
            **self.repository.stats(),
        }

    def start_retention_task(self) -> None:
        if self._retention_task is None or self._retention_task.done():
            self._retention_task = asyncio.create_task(self.run_retention())

    def subscribe_stream(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=2)
        self.stream_subscribers.add(queue)
        return queue

    def unsubscribe_stream(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self.stream_subscribers.discard(queue)

    async def process_frame(self, force_vlm: bool = False, frame: np.ndarray | None = None) -> dict[str, Any]:
        if not self.pipeline:
            self.start()
        if not self.session or self.session["status"] != "RUNNING":
            raise RuntimeError("No active session. Start a session before processing frames.")
        async with self._process_lock:
            return await self._process_frame(force_vlm, frame)

    def _record_event(self, event: dict[str, Any], frame: np.ndarray | None) -> dict[str, Any]:
        """Persist an event, attach an evidence frame when enabled, and publish it."""
        assert self.session is not None
        event["id"] = self.repository.save_event(self.session["id"], event_type=event["event_type"], step_id=event.get("step_id"), message=event["message"], confidence=event["confidence"], evidence=event.get("evidence") or {}, severity=event.get("severity", "INFO"))
        if frame is not None and self.frame_store.enabled:
            saved = self.frame_store.save(self.session["id"], event["id"], frame)
            if saved:
                self.repository.set_event_frame(event["id"], saved); event["frame_path"] = saved
        self.recent_events = [event, *self.recent_events][:30]
        return event

    async def _process_frame(self, force_vlm: bool, frame: np.ndarray | None) -> dict[str, Any]:
        if frame is not None:
            self._last_frame = frame.copy()
        frame = frame if frame is not None else self._last_frame if self._last_frame is not None else np.zeros((480,640,3), dtype=np.uint8)
        start = time.perf_counter()
        simulated_now = self._source_started_at + timedelta(seconds=self.frame_id / 10) if self.settings.app.mode == "mock" and self._source_started_at else None
        obs, condition, transition = await self.pipeline.process(frame, self.frame_id, now=simulated_now, force_vlm=force_vlm); self.frame_id += 1; self.frames_processed += 1
        if transition and self.session:
            key = (transition, condition.reason)
            if key not in self._event_keys:
                self._event_keys.add(key); event = {"event_type":transition,"step_id":self.pipeline.engine.sop.steps[self.pipeline.engine.state.index-1].id if transition == "step_completed" else self.pipeline.engine.state.current.id,"message":condition.reason,"confidence":condition.confidence,"evidence":condition.evidence}
                self._record_event(event, frame); self.repository.save_step(self.session["id"], self.pipeline.engine.state.steps[event["step_id"]])
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
                        self._record_event({"event_type":"safety_violation","severity":"CRITICAL","step_id":step_id,"message":message,"confidence":confidence,"evidence":evidence}, frame)
        activity = self.pipeline.last_activity
        if self.session and activity and activity.label != self._last_activity_label:
            message = f"activity: {self._last_activity_label or 'unknown'} -> {activity.label}"
            self._record_event({"event_type":"activity_changed","step_id":self.pipeline.engine.state.current.id if self.settings.sop.enabled and self.pipeline.engine.state.current else "","message":message,"confidence":activity.confidence,"evidence":activity.evidence}, frame)
            self._last_activity_label = activity.label
        payload = {"type":"frame_result","timestamp":obs.timestamp.isoformat(),"fps":round(1/max(time.perf_counter()-start,.0001),2),"objects":[x.model_dump() for x in obs.objects],"poses":[x.model_dump() for x in obs.poses],"current_step":self.pipeline.engine.state.current.model_dump() if self.settings.sop.enabled and self.pipeline.engine.state.current else None,"recent_events":self.recent_events,"vlm_result":obs.vlm_result,"condition":condition.model_dump() if condition else None,"inference_ms":round((time.perf_counter()-start)*1000,2),"vlm_calls":self.pipeline.vlm_calls,"activity":activity.model_dump(mode="json") if activity else None}
        await self.broadcast(payload); return payload

    async def broadcast(self, payload: dict[str, Any]) -> None:
        # Snapshot both sets: a client disconnecting mid-broadcast mutates them.
        stale = []
        for ws in list(self.subscribers):
            try: await ws.send_json(payload)
            except Exception: stale.append(ws)
        for ws in stale: self.subscribers.discard(ws)
        for queue in list(self.stream_subscribers):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty): queue.get_nowait()
            queue.put_nowait(payload)
