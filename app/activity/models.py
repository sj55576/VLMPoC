"""Daily-activity estimation data models."""
from __future__ import annotations
from datetime import datetime
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field


class ActivityLabel(StrEnum):
    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"
    WALKING = "walking"
    REACHING = "reaching"
    DRINKING = "drinking"
    PHONE_USE = "phone_use"
    IDLE = "idle"
    UNKNOWN = "unknown"


class ActivityEstimate(BaseModel):
    label: str
    confidence: float
    person_id: int = 1
    since: datetime | None = None
    duration_seconds: float = 0.0
    evidence: dict[str, Any] = Field(default_factory=dict)
