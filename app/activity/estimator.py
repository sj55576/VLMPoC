"""Heuristic daily-activity estimation from pose keypoints and nearby objects."""
from __future__ import annotations

from collections import Counter, deque
from datetime import datetime

from app.vision.models import Keypoint, Observation

from .models import ActivityEstimate, ActivityLabel

CONFIDENCE_THRESHOLD = 0.3
WALKING_SPEED_THRESHOLD = 0.04
INTERACTION_RADIUS = 0.12
REACH_MARGIN = 0.05
DRINK_TOKENS = ("cup", "bottle", "glass")
PHONE_TOKENS = ("cell_phone", "phone", "smartphone")


def _kp(keypoints: dict[str, Keypoint], name: str) -> Keypoint | None:
    point = keypoints.get(name)
    return point if point and point.confidence >= CONFIDENCE_THRESHOLD else None


def _avg_y(keypoints: dict[str, Keypoint], *names: str) -> float | None:
    values = [k.y for n in names if (k := _kp(keypoints, n))]
    return sum(values) / len(values) if values else None


class ActivityEstimator:
    """Tracks a bounded history of pose-derived observations and emits a smoothed activity label."""

    def __init__(self, window_seconds: float = 2.0, min_hold_seconds: float = 0.6) -> None:
        self.window_seconds, self.min_hold_seconds = window_seconds, min_hold_seconds
        self._positions: deque[tuple[datetime, float]] = deque()
        self._raw_history: deque[tuple[datetime, str, dict]] = deque()
        self._emitted: str | None = None
        self._emitted_since: datetime | None = None
        self._candidate: str | None = None
        self._candidate_since: datetime | None = None

    def update(self, obs: Observation, now: datetime) -> ActivityEstimate | None:
        if not obs.poses:
            return None
        pose = obs.poses[0]
        keypoints = pose.keypoints

        center_point = _kp(keypoints, "left_hip") or _kp(keypoints, "right_hip") or _kp(keypoints, "nose")
        if center_point is not None:
            self._positions.append((now, center_point.x))
        self._trim(self._positions, now)

        posture = self._posture(keypoints)
        speed = self._speed()
        walking = speed > WALKING_SPEED_THRESHOLD and posture == ActivityLabel.STANDING

        interaction = self._interaction(obs, keypoints)
        if interaction is not None:
            raw, evidence = interaction
        elif walking:
            raw, evidence = ActivityLabel.WALKING, {"speed": round(speed, 4)}
        elif self._reaching(keypoints):
            raw, evidence = ActivityLabel.REACHING, {}
        elif posture is not None:
            raw, evidence = posture, {}
        else:
            raw, evidence = ActivityLabel.UNKNOWN, {}

        self._raw_history.append((now, str(raw), evidence))
        self._trim(self._raw_history, now)

        votes = Counter(label for _, label, _ in self._raw_history)
        majority, count = votes.most_common(1)[0]
        confidence = count / len(self._raw_history)
        majority_evidence = next((e for _, label, e in reversed(self._raw_history) if label == majority), {})

        if self._emitted is None:
            self._emitted, self._emitted_since = majority, now
            self._candidate = self._candidate_since = None
        elif majority == self._emitted:
            self._candidate = self._candidate_since = None
        else:
            if majority != self._candidate:
                self._candidate, self._candidate_since = majority, now
            elif self._candidate_since is not None and (now - self._candidate_since).total_seconds() >= self.min_hold_seconds:
                self._emitted, self._emitted_since = majority, self._candidate_since
                self._candidate = self._candidate_since = None

        duration = (now - self._emitted_since).total_seconds() if self._emitted_since else 0.0
        return ActivityEstimate(label=self._emitted or ActivityLabel.UNKNOWN, confidence=confidence, person_id=pose.person_id, since=self._emitted_since, duration_seconds=round(duration, 2), evidence=majority_evidence)

    def _trim(self, history: deque, now: datetime) -> None:
        while history and (now - history[0][0]).total_seconds() > self.window_seconds:
            history.popleft()

    def _speed(self) -> float:
        if len(self._positions) < 2:
            return 0.0
        (t0, x0), (t1, x1) = self._positions[0], self._positions[-1]
        dt = (t1 - t0).total_seconds()
        return abs(x1 - x0) / dt if dt > 0 else 0.0

    def _posture(self, keypoints: dict[str, Keypoint]) -> str | None:
        shoulder_y = _avg_y(keypoints, "left_shoulder", "right_shoulder")
        hip_y = _avg_y(keypoints, "left_hip", "right_hip")
        if shoulder_y is None or hip_y is None:
            return None
        knee_y = _avg_y(keypoints, "left_knee", "right_knee")
        ankle_y = _avg_y(keypoints, "left_ankle", "right_ankle")
        xs = [k.x for k in keypoints.values() if k.confidence >= CONFIDENCE_THRESHOLD]
        bbox_width = (max(xs) - min(xs)) if len(xs) >= 2 else 0.3
        torso = hip_y - shoulder_y
        if bbox_width > 0 and torso < bbox_width * 0.35:
            return ActivityLabel.LYING
        if knee_y is not None and ankle_y is not None:
            midpoint = (shoulder_y + ankle_y) / 2
            if abs(knee_y - hip_y) < 0.08 and hip_y > midpoint:
                return ActivityLabel.SITTING
        return ActivityLabel.STANDING

    def _reaching(self, keypoints: dict[str, Keypoint]) -> bool:
        for wrist_name, shoulder_name in (("left_wrist", "left_shoulder"), ("right_wrist", "right_shoulder")):
            wrist, shoulder = _kp(keypoints, wrist_name), _kp(keypoints, shoulder_name)
            if wrist and shoulder and wrist.y < shoulder.y - REACH_MARGIN:
                return True
        return False

    def _interaction(self, obs: Observation, keypoints: dict[str, Keypoint]) -> tuple[str, dict] | None:
        for det in obs.objects:
            name = det.class_name.lower()
            is_drink, is_phone = any(t in name for t in DRINK_TOKENS), any(t in name for t in PHONE_TOKENS)
            if not is_drink and not is_phone:
                continue
            bbox = obs.normalized_bbox(det)
            center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            for wrist_name, elbow_name in (("left_wrist", "left_elbow"), ("right_wrist", "right_elbow")):
                wrist = _kp(keypoints, wrist_name)
                if wrist is None:
                    continue
                distance = ((wrist.x - center_x) ** 2 + (wrist.y - center_y) ** 2) ** 0.5
                if distance > INTERACTION_RADIUS:
                    continue
                if is_drink:
                    elbow = _kp(keypoints, elbow_name)
                    if elbow is not None and wrist.y < elbow.y:
                        return ActivityLabel.DRINKING, {"object": det.class_name, "wrist": wrist_name}
                if is_phone:
                    return ActivityLabel.PHONE_USE, {"object": det.class_name, "wrist": wrist_name}
        return None
