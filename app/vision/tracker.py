"""IoU tracker fallback retaining identity and basic motion metadata."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from .geometry import iou
from .models import Detection


@dataclass
class Track:
    id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    history: list[tuple[float, float, float, float]] = field(default_factory=list)
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    velocity: tuple[float, float] = (0.0, 0.0)


class IoUTracker:
    def __init__(self, threshold: float = .25, max_age_seconds: float = 1.0) -> None:
        self.threshold, self.max_age_seconds, self._next_id, self.tracks = threshold, max_age_seconds, 1, {}

    def update(self, detections: list[Detection], now: datetime | None = None) -> list[Detection]:
        now = now or datetime.now(timezone.utc)
        unmatched = set(self.tracks)
        for detection in detections:
            candidates = [(iou(detection.bbox, t.bbox), track_id) for track_id, t in self.tracks.items() if t.class_name == detection.class_name and track_id in unmatched]
            score, track_id = max(candidates, default=(0.0, None))
            if score < self.threshold or track_id is None:
                track_id = self._next_id; self._next_id += 1
                self.tracks[track_id] = Track(track_id, detection.class_name, detection.confidence, detection.bbox)
            track = self.tracks[track_id]
            old_center = ((track.bbox[0]+track.bbox[2])/2, (track.bbox[1]+track.bbox[3])/2)
            new_center = ((detection.bbox[0]+detection.bbox[2])/2, (detection.bbox[1]+detection.bbox[3])/2)
            elapsed = max((now-track.last_seen).total_seconds(), .001)
            track.velocity = ((new_center[0]-old_center[0])/elapsed, (new_center[1]-old_center[1])/elapsed)
            track.history = (track.history + [track.bbox])[-60:]
            track.bbox, track.confidence, track.last_seen = detection.bbox, detection.confidence, now
            detection.track_id = track_id; unmatched.discard(track_id)
        self.tracks = {tid: t for tid, t in self.tracks.items() if (now-t.last_seen).total_seconds() <= self.max_age_seconds}
        return detections
