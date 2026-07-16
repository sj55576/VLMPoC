"""Strict VLM response schema and safe parsing."""
from __future__ import annotations
import json
from typing import Any
from pydantic import BaseModel, Field, ValidationError


class VLMEvidence(BaseModel):
    type: str
    description: str


class VLMResponse(BaseModel):
    scene_summary: str
    detected_action: str = "UNKNOWN"
    current_step_id: str | None = None
    step_status: str = "UNKNOWN"
    confidence: float = Field(ge=0, le=1)
    safety_violation: bool = False
    violations: list[str] = Field(default_factory=list)
    evidence: list[VLMEvidence] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


def unknown_response(reason: str) -> VLMResponse:
    return VLMResponse(scene_summary="VLM analysis unavailable", step_status="UNKNOWN", confidence=0.0, uncertainties=[reason])


def parse_vlm_response(value: str | dict[str, Any]) -> VLMResponse:
    """Parse schema-valid JSON, falling back safely rather than trusting free text."""
    try:
        if isinstance(value, str):
            value = json.loads(value.removeprefix("```json").removesuffix("```").strip())
        return VLMResponse.model_validate(value)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        return unknown_response(f"invalid VLM JSON: {exc.__class__.__name__}")
