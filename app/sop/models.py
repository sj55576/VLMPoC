"""Validated SOP and evaluation domain models."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    PENDING = "PENDING"; ACTIVE = "ACTIVE"; COMPLETED = "COMPLETED"; SKIPPED = "SKIPPED"; FAILED = "FAILED"


class EventStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"; IN_PROGRESS = "IN_PROGRESS"; COMPLETED = "COMPLETED"; FAILED = "FAILED"; VIOLATION = "VIOLATION"; UNKNOWN = "UNKNOWN"


class StepDefinition(BaseModel):
    id: str; name: str; description: str = ""; conditions: dict[str, Any]
    minimum_duration_seconds: float = 0.0; timeout_seconds: float | None = None
    on_success: str | None = None; terminal: bool = False


class SOPMeta(BaseModel):
    id: str; name: str; version: str


class SOPDefinition(BaseModel):
    sop: SOPMeta
    objects: dict[str, list[str]] = Field(default_factory=dict)
    steps: list[StepDefinition]
    regions: dict[str, dict[str, float]] = Field(default_factory=dict)


class ConditionResult(BaseModel):
    condition_id: str
    type: str
    passed: bool
    confidence: float
    reason: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class StepRuntime(BaseModel):
    step_id: str; status: StepStatus = StepStatus.PENDING
    started_at: datetime | None = None; completed_at: datetime | None = None
    condition_true_since: datetime | None = None; confidence: float = 0.0; reason: str = ""
