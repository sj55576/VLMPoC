"""Evidence frame store: writes JPEG snapshots for events and serves them back safely."""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def _sanitize(value: str) -> str:
    """Strip anything but [A-Za-z0-9_-] so a hostile id cannot escape the base directory."""
    return _UNSAFE.sub("", str(value))


class FrameStore:
    """Stores evidence frames on disk as `<directory>/<session_id>/<event_id>.jpg`."""

    def __init__(self, directory: str | Path, enabled: bool = True, jpeg_quality: int = 80, max_dim: int = 1280) -> None:
        self.directory = Path(directory).resolve()
        self._enabled = enabled
        self.jpeg_quality = jpeg_quality
        self.max_dim = max_dim
        if self._enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def save(self, session_id: str, event_id: int, frame: Any) -> str | None:
        """Downscale, JPEG-encode, and write a frame; returns a path relative to `directory`, or None."""
        if not self._enabled or frame is None:
            return None
        try:
            import cv2

            safe_session = _sanitize(session_id) or "session"
            safe_event = _sanitize(str(event_id)) or "0"
            session_dir = self.directory / safe_session
            session_dir.mkdir(parents=True, exist_ok=True)

            height, width = frame.shape[:2]
            longest = max(height, width)
            if longest > self.max_dim:
                scale = self.max_dim / longest
                frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))

            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                logger.warning("Failed to JPEG-encode frame for session=%s event=%s", session_id, event_id)
                return None

            target = session_dir / f"{safe_event}.jpg"
            target.write_bytes(buffer.tobytes())
            return f"{safe_session}/{safe_event}.jpg"
        except Exception:
            logger.warning("Failed to save evidence frame for session=%s event=%s", session_id, event_id, exc_info=True)
            return None

    def path(self, relative: str) -> Path | None:
        """Traversal-safe read side: only returns a path inside `directory` that actually exists."""
        try:
            candidate = (self.directory / relative).resolve()
        except (OSError, ValueError):
            return None
        if candidate != self.directory and self.directory not in candidate.parents:
            return None
        if not candidate.is_file():
            return None
        return candidate

    def purge_older_than(self, days: int) -> int:
        """Delete .jpg files older than the cutoff and remove directories left empty."""
        if days <= 0 or not self.directory.exists():
            return 0
        cutoff = time.time() - days * 86400
        deleted = 0
        for session_dir in self.directory.iterdir():
            if not session_dir.is_dir():
                continue
            for jpg in session_dir.glob("*.jpg"):
                try:
                    if jpg.stat().st_mtime < cutoff:
                        jpg.unlink()
                        deleted += 1
                except OSError:
                    continue
            try:
                if not any(session_dir.iterdir()):
                    session_dir.rmdir()
            except OSError:
                pass
        return deleted

    def purge_session(self, session_id: str) -> int:
        """Delete all frames belonging to a session; returns the count deleted."""
        safe_session = _sanitize(session_id)
        if not safe_session:
            return 0
        session_dir = self.directory / safe_session
        if not session_dir.is_dir():
            return 0
        deleted = 0
        for jpg in session_dir.glob("*.jpg"):
            try:
                jpg.unlink()
                deleted += 1
            except OSError:
                continue
        try:
            if not any(session_dir.iterdir()):
                session_dir.rmdir()
        except OSError:
            pass
        return deleted
