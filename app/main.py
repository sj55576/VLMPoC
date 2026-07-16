"""FastAPI entrypoint for the local SOP monitor."""
from __future__ import annotations
import asyncio, csv, io
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from app.core.config import load_settings
from app.core.logging import configure_logging
from app.services.session import SessionService
from app.sop.loader import load_sop

ROOT = Path(__file__).resolve().parents[1]


def create_app() -> FastAPI:
    configure_logging(); settings = load_settings(ROOT); service = SessionService(settings, ROOT)
    app = FastAPI(title=settings.app.name); app.state.service = service
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.get("/health")
    async def health(): return {"status":"ok","mode":settings.app.mode,"version":"0.1.0"}

    @app.get("/api/config")
    async def config():
        safe = settings.model_dump(); safe["vlm"]["api_key"] = "***" if safe["vlm"]["api_key"] else ""; return safe

    @app.get("/api/sops")
    async def sops(): return [{"id":service.sop.sop.id,"name":service.sop.sop.name,"version":service.sop.sop.version}]

    @app.post("/api/sops/reload")
    async def reload_sop(): service.sop = load_sop(ROOT / "sop" / "example_assembly.yaml"); return {"status":"reloaded","id":service.sop.sop.id}

    @app.post("/api/session/start")
    async def start(payload: dict | None = None):
        payload = payload or {}; return service.start(payload.get("source_type","mock"), payload.get("source_name","synthetic"))

    @app.post("/api/session/stop")
    async def stop(): return service.stop()

    @app.get("/api/session/status")
    async def status(): return service.status()

    @app.get("/api/events")
    async def events(): return service.repository.events(service.session["id"] if service.session else None)

    @app.get("/api/events/{event_id}")
    async def event(event_id: int):
        value = service.repository.event(event_id)
        if not value: raise HTTPException(404, "event not found")
        return value

    @app.get("/api/steps")
    async def steps(): return service.status()["steps"]

    @app.post("/api/analyze/image")
    async def analyze_image(file: UploadFile = File(...)):
        if file.size and file.size > settings.security.max_upload_mb * 1024 * 1024: raise HTTPException(413, "file too large")
        if not (file.content_type or "").startswith("image/"): raise HTTPException(415, "only image uploads are accepted")
        await file.read()  # bytes intentionally not logged or persisted in the mock path
        return await service.process_mock_frame()

    @app.post("/api/analyze/video")
    async def analyze_video(file: UploadFile = File(...)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in settings.security.allowed_video_extensions: raise HTTPException(415, "unsupported video format")
        data = await file.read()
        if len(data) > settings.security.max_upload_mb * 1024 * 1024: raise HTTPException(413, "file too large")
        # Keep uploads out of paths; mock mode processes a deterministic timeline.
        results = [await service.process_mock_frame() for _ in range(60)]
        return {"frames_processed":len(results),"last":results[-1]}

    @app.post("/api/vlm/analyze")
    async def vlm_analyze(): return await service.process_mock_frame(force_vlm=True)

    @app.get("/api/stream")
    async def stream():
        async def events():
            while True:
                payload = await service.process_mock_frame(); yield f"data: {payload}\n\n"; await asyncio.sleep(.1)
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/api/events/download")
    async def download():
        rows = service.repository.events(service.session["id"] if service.session else None); buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["id","event_type","severity","timestamp","step_id","message","confidence","evidence_json"], extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
        return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv", headers={"Content-Disposition":"attachment; filename=events.csv"})

    @app.websocket("/api/ws")
    async def websocket(ws: WebSocket):
        await ws.accept(); service.subscribers.add(ws)
        try:
            while True:
                try: await asyncio.wait_for(ws.receive_text(), timeout=.2)
                except TimeoutError: await service.process_mock_frame()
        except WebSocketDisconnect: service.subscribers.discard(ws)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request): return HTMLResponse((ROOT / "templates" / "index.html").read_text(encoding="utf-8"))
    return app


app = create_app()
