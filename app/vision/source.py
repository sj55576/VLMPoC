"""Server-side frame sources.

Until now the only producer was a synthetic loop in ``SessionService``: a camera or
video file could only be analysed by a browser pushing JPEGs to ``/api/analyze/image``.
These sources let the server itself ingest a webcam, a video file, or an RTSP stream.

``cv2.VideoCapture.read`` blocks, so capture runs on a daemon thread and hands frames to
the asyncio consumer through a bounded drop-oldest buffer: when analysis is slower than
the camera, old frames are discarded instead of building an ever-growing latency debt.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
SOURCE_TYPES = ("mock", "camera", "file", "rtsp")


class FrameSourceError(RuntimeError):
    """Raised when a source cannot be opened or its URI is not acceptable."""


@dataclass
class SourceStats:
    """Counters surfaced through /api/session/status and /metrics."""

    frames_read: int = 0
    frames_dropped: int = 0
    reconnects: int = 0
    opened: bool = False
    finished: bool = False
    last_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"frames_read": self.frames_read, "frames_dropped": self.frames_dropped, "reconnects": self.reconnects, "opened": self.opened, "finished": self.finished, "last_error": self.last_error}


@dataclass
class SourceSpec:
    """Resolved, validated description of what to capture."""

    type: str
    uri: str = ""
    target_fps: float = 10.0
    queue_size: int = 2
    reconnect_seconds: float = 3.0
    max_reconnect_attempts: int = 0
    loop_file: bool = False
    stats: SourceStats = field(default_factory=SourceStats)


def resolve_source_uri(source_type: str, uri: str, allowed_root: Path) -> str:
    """Validate a caller-supplied URI.

    ``source_type`` and ``uri`` can arrive from an HTTP request, so a file source is
    confined to ``allowed_root``; otherwise the endpoint would read any path on the host.
    """
    if source_type not in SOURCE_TYPES:
        raise FrameSourceError(f"source_type must be one of {', '.join(SOURCE_TYPES)}")
    uri = (uri or "").strip()
    if source_type == "mock":
        return ""
    if source_type == "camera":
        if uri and not uri.isdigit():
            raise FrameSourceError("A camera source URI must be a device index such as '0'.")
        return uri or "0"
    if source_type == "rtsp":
        if not uri.startswith(("rtsp://", "rtsps://", "http://", "https://")):
            raise FrameSourceError("An rtsp source URI must start with rtsp://, rtsps://, http:// or https://.")
        return uri
    if not uri:
        raise FrameSourceError("A file source requires a video path.")
    root = allowed_root.resolve()
    candidate = (root / uri).resolve() if not Path(uri).is_absolute() else Path(uri).resolve()
    if not candidate.is_relative_to(root):
        raise FrameSourceError(f"A file source must live under {root}.")
    if not candidate.is_file():
        raise FrameSourceError(f"Video file was not found: {candidate}")
    return str(candidate)


class FrameSource(ABC):
    """Async frame producer. ``read`` returns ``None`` once the source is exhausted."""

    def __init__(self, spec: SourceSpec) -> None:
        self.spec = spec
        self.stats = spec.stats

    @abstractmethod
    async def open(self) -> None: ...

    @abstractmethod
    async def read(self) -> np.ndarray | None: ...

    @abstractmethod
    async def close(self) -> None: ...

    def abort(self) -> None:
        """Signal the source to stop from synchronous code; ``close`` still does the joining."""
        return None


class MockFrameSource(FrameSource):
    """Paced blank frames; the mock detector synthesises its own objects from frame ids."""

    def __init__(self, spec: SourceSpec) -> None:
        super().__init__(spec)
        self._interval = 1.0 / spec.target_fps if spec.target_fps > 0 else 0.1
        self._next_at = 0.0

    async def open(self) -> None:
        self.stats.opened = True
        self._next_at = time.monotonic()

    async def read(self) -> np.ndarray | None:
        delay = self._next_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        self._next_at = max(self._next_at + self._interval, time.monotonic())
        self.stats.frames_read += 1
        return np.zeros((480, 640, 3), dtype=np.uint8)

    async def close(self) -> None:
        self.stats.opened = False


class OpenCVFrameSource(FrameSource):
    """Threaded ``cv2.VideoCapture`` reader with drop-oldest buffering and reconnect."""

    def __init__(self, spec: SourceSpec) -> None:
        super().__init__(spec)
        self._buffer: deque[np.ndarray] = deque(maxlen=max(1, spec.queue_size))
        self._buffer_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._available = asyncio.Event()
        self._finished = threading.Event()
        self._opened = threading.Event()
        self._open_error: str | None = None
        self._interval = 1.0 / spec.target_fps if spec.target_fps > 0 else 0.0

    # -- capture thread ----------------------------------------------------

    def _target(self) -> Any:
        """cv2 wants an int for a device index and a string for a path or URL."""
        return int(self.spec.uri) if self.spec.type == "camera" else self.spec.uri

    def _open_capture(self) -> Any:
        import cv2

        capture = cv2.VideoCapture(self._target())
        if not capture.isOpened():
            capture.release()
            return None
        # A small driver-side buffer keeps live sources close to real time.
        if self.spec.type in {"camera", "rtsp"}:
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:  # unsupported by some capture backends, never fatal
                LOGGER.debug("Capture backend rejected CAP_PROP_BUFFERSIZE")
        return capture

    def _publish(self, frame: np.ndarray) -> None:
        with self._buffer_lock:
            if len(self._buffer) == self._buffer.maxlen:
                self.stats.frames_dropped += 1
            self._buffer.append(frame)
        self.stats.frames_read += 1
        self._wake()

    def _wake(self) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._available.set)
        except RuntimeError:  # loop shut down between the check and the call
            LOGGER.debug("Event loop closed while waking the frame consumer")

    def _run(self) -> None:
        attempts = 0
        capture = None
        try:
            while not self._stop.is_set():
                if capture is None:
                    capture = self._open_capture()
                    if capture is None:
                        attempts += 1
                        limit = self.spec.max_reconnect_attempts
                        self._open_error = f"Unable to open {self.spec.type} source: {self.spec.uri or 'default'}"
                        self.stats.last_error = self._open_error
                        if not self._opened.is_set() or (limit and attempts >= limit):
                            # Never opened, or the retry budget is spent: give up and let read() end.
                            break
                        self.stats.reconnects += 1
                        if self._stop.wait(self.spec.reconnect_seconds):
                            break
                        continue
                    attempts = 0
                    self._open_error = None
                    self._opened.set()
                    self.stats.opened = True
                started = time.monotonic()
                ok, frame = capture.read()
                if not ok or frame is None:
                    capture.release()
                    capture = None
                    self.stats.opened = False
                    if self.spec.type == "file" and not self.spec.loop_file:
                        break
                    self.stats.reconnects += 1
                    if self._stop.wait(self.spec.reconnect_seconds if self.spec.type != "file" else 0.0):
                        break
                    continue
                self._publish(frame)
                if self._interval:
                    remaining = self._interval - (time.monotonic() - started)
                    if remaining > 0 and self._stop.wait(remaining):
                        break
        except Exception as exc:  # a capture thread must never die silently
            LOGGER.exception("Frame capture thread failed")
            self.stats.last_error = f"{exc.__class__.__name__}: {exc}"
        finally:
            if capture is not None:
                capture.release()
            self.stats.opened = False
            self.stats.finished = True
            self._finished.set()
            self._wake()

    # -- async interface ---------------------------------------------------

    async def open(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"frame-source-{self.spec.type}", daemon=True)
        self._thread.start()
        # Surface an unusable camera/file as a start-up error instead of an empty stream.
        opened = await asyncio.to_thread(self._wait_first_open)
        if not opened:
            await self.close()
            raise FrameSourceError(self._open_error or f"Unable to open {self.spec.type} source: {self.spec.uri or 'default'}")

    def _wait_first_open(self) -> bool:
        while not self._opened.is_set() and not self._finished.is_set():
            if self._opened.wait(0.1):
                break
        return self._opened.is_set()

    async def read(self) -> np.ndarray | None:
        while True:
            with self._buffer_lock:
                if self._buffer:
                    return self._buffer.popleft()
                self._available.clear()
                exhausted = self._finished.is_set()
            if exhausted:
                return None
            await self._available.wait()

    def abort(self) -> None:
        self._stop.set()
        self._wake()

    async def close(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, 5.0)
        self.stats.opened = False
        with self._buffer_lock:
            self._buffer.clear()


def create_frame_source(spec: SourceSpec) -> FrameSource:
    """Build the source described by ``spec``."""
    if spec.type == "mock":
        return MockFrameSource(spec)
    if spec.type in {"camera", "file", "rtsp"}:
        return OpenCVFrameSource(spec)
    raise FrameSourceError(f"source_type must be one of {', '.join(SOURCE_TYPES)}")
