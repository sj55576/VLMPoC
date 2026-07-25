"""FastAPI entrypoint for the local SOP monitor."""
from __future__ import annotations

import csv
import io
import json
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.core.config import load_settings
from app.core.logging import configure_logging
from app.core.metrics import render_metrics
from app.core.security import ApiKeyGuard, RateLimiter
from app.services.session import SERVER_SOURCE_TYPES, SessionService

ROOT = Path(__file__).resolve().parents[1]
# The dashboard shell stays reachable so a browser can prompt for credentials elsewhere;
# every data path is guarded.
PUBLIC_PATHS = ("/health", "/", "/static", "/favicon.ico")


def create_app() -> FastAPI:
    configure_logging(); settings = load_settings(ROOT); service = SessionService(settings, ROOT)
    guard = ApiKeyGuard(settings.security.api_key, PUBLIC_PATHS)
    limiter = RateLimiter(settings.security.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        service.start_retention_task()
        try:
            yield
        finally:
            await service.shutdown()

    app = FastAPI(title=settings.app.name, version=__version__, lifespan=lifespan)
    app.state.service = service; app.state.guard = guard; app.state.rate_limiter = limiter
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.middleware("http")
    async def enforce_access(request: Request, call_next):
        path = request.url.path
        if not guard.authorize(path, request.headers.get("x-api-key"), request.headers.get("authorization")):
            return Response(status_code=401, content='{"detail":"invalid or missing API key"}', media_type="application/json")
        if limiter.enabled and not guard.is_exempt(path) and not limiter.allow(request.client.host if request.client else "unknown"):
            return Response(status_code=429, content='{"detail":"rate limit exceeded"}', media_type="application/json")
        return await call_next(request)

    @app.get("/health")
    async def health(): return {"status":"ok","mode":settings.app.mode,"version":__version__}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics(): return PlainTextResponse(render_metrics(service.metrics_snapshot()), media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/api/config")
    async def config():
        safe = settings.model_dump()
        safe["vlm"]["api_key"] = "***" if safe["vlm"]["api_key"] else ""
        safe["security"]["api_key"] = "***" if safe["security"]["api_key"] else ""
        return safe

    @app.get("/api/sops")
    async def sops(): return service.list_sops()

    @app.post("/api/sops/reload")
    async def reload_sop(): return {"status":"reloaded","sops":service.reload_sops()}

    @app.post("/api/session/start")
    async def start(payload: dict | None = None):
        payload = payload or {}
        try:
            service.start(payload.get("source_type"), payload.get("source_name", "synthetic"), payload.get("sop_id"), payload.get("source_uri"))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        # Server-side ingestion begins with the session, so an unusable camera or stream is
        # reported here instead of silently producing nothing. The mock producer stays lazy —
        # it must not burn CPU for nobody — but a client that connected before this session
        # existed would otherwise wait forever, so start it when someone is already attached.
        kind = service.session["source_type"] if service.session else None
        if kind in SERVER_SOURCE_TYPES or (kind == "mock" and service.has_subscribers):
            try:
                await service.ensure_runner()
            except ValueError as exc:
                service.stop()
                raise HTTPException(422, str(exc)) from exc
        return service.status()

    @app.post("/api/session/stop")
    async def stop(): return service.stop()

    @app.get("/api/session/status")
    async def status(): return service.status()

    @app.get("/api/sessions")
    async def sessions(limit: int = 50, offset: int = 0):
        return service.repository.sessions(limit=max(1, min(limit, 500)), offset=max(0, offset))

    @app.get("/api/sessions/{session_id}")
    async def session_detail(session_id: str):
        record = service.repository.session(session_id)
        if not record: raise HTTPException(404, "session not found")
        return {**record, "event_count": service.repository.count_events(session_id), "vlm_results": service.repository.vlm_records(session_id, limit=20)}

    @app.get("/api/sessions/{session_id}/events")
    async def session_events(session_id: str, response: Response, limit: int = 100, offset: int = 0, event_type: str | None = None, severity: str | None = None):
        response.headers["X-Total-Count"] = str(service.repository.count_events(session_id, event_type=event_type, severity=severity))
        return service.repository.events(session_id, limit=max(1, min(limit, 1000)), offset=max(0, offset), event_type=event_type, severity=severity)

    @app.get("/api/events")
    async def events(response: Response, limit: int = 100, offset: int = 0, event_type: str | None = None, severity: str | None = None, since: str | None = None, session_id: str | None = None, all_sessions: bool = False):
        scope = None if all_sessions else session_id or (service.session["id"] if service.session else None)
        response.headers["X-Total-Count"] = str(service.repository.count_events(scope, event_type=event_type, severity=severity, since=since))
        return service.repository.events(scope, limit=max(1, min(limit, 1000)), offset=max(0, offset), event_type=event_type, severity=severity, since=since)

    @app.get("/api/events/download")
    async def download():
        rows = service.repository.events(service.session["id"] if service.session else None, limit=0); buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["id","event_type","severity","timestamp","step_id","message","confidence","evidence_json","frame_path"], extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
        return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv", headers={"Content-Disposition":"attachment; filename=events.csv"})

    @app.get("/api/events/{event_id}")
    async def event(event_id: int):
        value = service.repository.event(event_id)
        if not value: raise HTTPException(404, "event not found")
        return value

    @app.get("/api/events/{event_id}/frame")
    async def event_frame(event_id: int):
        value = service.repository.event(event_id)
        if not value: raise HTTPException(404, "event not found")
        relative = value.get("frame_path")
        path = service.frame_store.path(relative) if relative else None
        if not path: raise HTTPException(404, "no evidence frame stored for this event")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/activity")
    async def activity(limit: int = 100):
        current = service.pipeline.last_activity.model_dump(mode="json") if service.pipeline and service.pipeline.last_activity else None
        history = service.repository.events(service.session["id"] if service.session else None, limit=max(1, min(limit, 1000)), event_type="activity_changed")
        return {"current": current, "history": history}

    @app.get("/api/steps")
    async def steps(): return service.status()["steps"]

    @app.post("/api/analyze/image")
    async def analyze_image(file: UploadFile = File(...)):
        if file.size and file.size > settings.security.max_upload_mb * 1024 * 1024: raise HTTPException(413, "file too large")
        if not (file.content_type or "").startswith("image/"): raise HTTPException(415, "only image uploads are accepted")
        import cv2
        import numpy as np
        raw = await file.read()
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None: raise HTTPException(422, "invalid image data")
        frame = cv2.resize(frame, (640, 480))
        return await service.process_frame(frame=frame)

    @app.post("/api/analyze/video")
    async def analyze_video(file: UploadFile = File(...)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in settings.security.allowed_video_extensions: raise HTTPException(415, "unsupported video format")
        data = await file.read()
        if len(data) > settings.security.max_upload_mb * 1024 * 1024: raise HTTPException(413, "file too large")
        import cv2
        # A randomized temporary path prevents path traversal and avoids trusting the upload name.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temporary:
            temporary.write(data); temporary.flush()
            capture = cv2.VideoCapture(temporary.name)
            results = []
            while len(results) < 600:
                ok, frame = capture.read()
                if not ok: break
                results.append(await service.process_frame(frame=cv2.resize(frame, (640, 480))))
            capture.release()
        if not results: raise HTTPException(422, "unable to decode video")
        return {"frames_processed":len(results),"last":results[-1]}

    @app.post("/api/vlm/analyze")
    async def vlm_analyze(): return await service.process_frame(force_vlm=True)

    @app.get("/api/stream")
    async def stream():
        async def events():
            queue = service.subscribe_stream()
            await service.ensure_runner()
            try:
                while True:
                    yield f"data: {json.dumps(await queue.get(), ensure_ascii=False)}\n\n"
            finally:
                service.unsubscribe_stream(queue)
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.websocket("/api/ws")
    async def websocket(ws: WebSocket):
        # HTTP middleware does not cover websockets, so the same key is checked here.
        if not guard.authorize("/api/ws", ws.headers.get("x-api-key") or ws.query_params.get("api_key"), ws.headers.get("authorization")):
            await ws.close(code=1008)
            return
        await ws.accept(); service.subscribers.add(ws)
        try:
            await service.ensure_runner()
            while True:
                await ws.receive_text()
        except (WebSocketDisconnect, ValueError):
            pass
        finally:
            service.subscribers.discard(ws)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request): return HTMLResponse((ROOT / "templates" / "index.html").read_text(encoding="utf-8"))
    return app


app = create_app()
