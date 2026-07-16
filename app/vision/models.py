"""Common detector, pose, and observation models."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field


class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: int | None = None

    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


class Keypoint(BaseModel):
    x: float
    y: float
    confidence: float


class Pose(BaseModel):
    person_id: int = 1
    keypoints: dict[str, Keypoint] = Field(default_factory=dict)


class Observation(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    frame_id: int = 0
    width: int = 1
    height: int = 1
    objects: list[Detection] = Field(default_factory=list)
    poses: list[Pose] = Field(default_factory=list)
    candidate_events: list[str] = Field(default_factory=list)
    vlm_result: dict[str, Any] | None = None

    def normalized_bbox(self, det: Detection) -> list[float]:
        return [det.bbox[0] / self.width, det.bbox[1] / self.height, det.bbox[2] / self.width, det.bbox[3] / self.height]
