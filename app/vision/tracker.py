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
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    velocity: tuple[float, float] = (0.0, 0.0)
    direction: tuple[float, float] = (0.0, 0.0)
    missing_seconds: float = 0.0


class IoUTracker:
    def __init__(self, threshold: float = .25, max_age_seconds: float = 1.0, history_size: int = 60) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("IoU threshold must be between 0 and 1.")
        if max_age_seconds < 0:
            raise ValueError("max_age_seconds must be non-negative.")
        self.threshold, self.max_age_seconds = threshold, max_age_seconds
        self.history_size, self._next_id, self.tracks = max(1, history_size), 1, {}

    def update(self, detections: list[Detection], now: datetime | None = None) -> list[Detection]:
        now = now or datetime.now(timezone.utc)
        valid = [detection for detection in detections if iou(detection.bbox, detection.bbox) > 0]
        candidates = sorted(
            ((iou(detection.bbox, track.bbox), detection_index, track_id)
             for detection_index, detection in enumerate(valid)
             for track_id, track in self.tracks.items()
             if detection.class_name == track.class_name),
            reverse=True,
        )
        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()
        matches: dict[int, int] = {}
        for score, detection_index, track_id in candidates:
            if score < self.threshold:
                break
            if detection_index not in matched_detections and track_id not in matched_tracks:
                matches[detection_index] = track_id
                matched_detections.add(detection_index)
                matched_tracks.add(track_id)
        for detection_index, detection in enumerate(valid):
            track_id = matches.get(detection_index)
            created = track_id is None
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self.tracks[track_id] = Track(track_id, detection.class_name, detection.confidence, detection.bbox, last_seen=now, first_seen=now)
            track = self.tracks[track_id]
            old_center = ((track.bbox[0]+track.bbox[2])/2, (track.bbox[1]+track.bbox[3])/2)
            new_center = ((detection.bbox[0]+detection.bbox[2])/2, (detection.bbox[1]+detection.bbox[3])/2)
            elapsed = max((now-track.last_seen).total_seconds(), .001)
            velocity = ((new_center[0]-old_center[0])/elapsed, (new_center[1]-old_center[1])/elapsed)
            speed = (velocity[0] ** 2 + velocity[1] ** 2) ** .5
            track.velocity = velocity
            track.direction = (velocity[0] / speed, velocity[1] / speed) if speed else (0.0, 0.0)
            if not created:
                track.history = (track.history + [track.bbox])[-self.history_size:]
            track.bbox, track.confidence, track.last_seen, track.missing_seconds = detection.bbox, detection.confidence, now, 0.0
            detection.track_id = track_id
        retained: dict[int, Track] = {}
        for track_id, track in self.tracks.items():
            track.missing_seconds = max(0.0, (now - track.last_seen).total_seconds())
            if track.missing_seconds <= self.max_age_seconds:
                retained[track_id] = track
        self.tracks = retained
        return valid
