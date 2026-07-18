"""Strict VLM response schema and safe parsing."""
from __future__ import annotations
import json
import re
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
    provider_success: bool = True
    error_message: str | None = None


def unknown_response(reason: str) -> VLMResponse:
    return VLMResponse(scene_summary="VLM analysis unavailable", step_status="UNKNOWN", confidence=0.0, uncertainties=[reason], provider_success=False, error_message=reason)


def _normalize_response(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize common OpenAI-compatible local VLM schema variations."""
    normalized = dict(value)
    normalized["scene_summary"] = str(normalized.get("scene_summary") or "VLM response received")
    normalized["detected_action"] = str(normalized.get("detected_action") or "UNKNOWN")
    normalized["step_status"] = str(normalized.get("step_status") or "UNKNOWN").upper()
    if normalized["step_status"] not in {"COMPLETED", "IN_PROGRESS", "VIOLATION", "UNKNOWN"}:
        normalized["step_status"] = "UNKNOWN"
    try:
        normalized["confidence"] = max(0.0, min(1.0, float(normalized.get("confidence", 0))))
    except (TypeError, ValueError):
        normalized["confidence"] = 0.0
    for key in ("violations", "uncertainties"):
        item = normalized.get(key, [])
        normalized[key] = [str(entry) for entry in (item if isinstance(item, list) else [item]) if entry]
    evidence = normalized.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [evidence]
    normalized["evidence"] = [
        entry if isinstance(entry, dict) and {"type", "description"} <= entry.keys()
        else {"type": "model", "description": str(entry.get("description") or entry) if isinstance(entry, dict) else str(entry)}
        for entry in evidence if entry
    ]
    safety = normalized.get("safety_violation", False)
    normalized["safety_violation"] = safety if isinstance(safety, bool) else str(safety).lower() in {"true", "1", "yes"}
    return normalized


def _recover_nonstandard_json(text: str) -> VLMResponse | None:
    """Recover the essential fields from near-JSON returned by permissive local VLM servers."""
    def string(key: str, default: str = "") -> str:
        match = re.search(rf'"{key}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
        return bytes(match.group(1), "utf-8").decode("unicode_escape") if match else default

    summary = string("scene_summary")
    if not summary:
        return None
    confidence_match = re.search(r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    confidence = float(confidence_match.group(1)) if confidence_match else 0.0
    confidence = max(0.0, min(1.0, confidence))
    evidence_match = re.search(r'"evidence"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    evidence = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', evidence_match.group(1)) if evidence_match else []
    uncertainties_match = re.search(r'"uncertainties"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    uncertainties = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', uncertainties_match.group(1)) if uncertainties_match else []
    if uncertainties_match and not uncertainties:
        bare = uncertainties_match.group(1).strip().strip('"')
        if bare:
            uncertainties = [bare]
    uncertainties.append("VLM returned nonstandard JSON; fields were recovered.")
    return VLMResponse(
        scene_summary=summary,
        detected_action=string("detected_action", "UNKNOWN"),
        current_step_id=None if re.search(r'"current_step_id"\s*:\s*null', text) else string("current_step_id") or None,
        step_status=string("step_status", "UNKNOWN"),
        confidence=confidence,
        safety_violation=bool(re.search(r'"safety_violation"\s*:\s*true', text, re.IGNORECASE)),
        violations=[],
        evidence=[{"type": "model", "description": item} for item in evidence],
        uncertainties=uncertainties,
    )


def parse_vlm_response(value: str | dict[str, Any]) -> VLMResponse:
    """Parse schema-valid JSON, falling back safely rather than trusting free text."""
    try:
        if isinstance(value, str):
            text = value.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                # Some local VLMs append one quote after a final JSON array.
                repaired = re.sub(r'(\])"\s*}$', r'\1}', text)
                value = json.loads(repaired)
        if isinstance(value, dict) and isinstance(value.get("evidence"), str):
            value["evidence"] = [{"type": "model", "description": value["evidence"]}]
        return VLMResponse.model_validate(_normalize_response(value) if isinstance(value, dict) else value)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        if isinstance(value, str):
            recovered = _recover_nonstandard_json(value)
            if recovered:
                return recovered
        return unknown_response(f"invalid VLM JSON: {exc.__class__.__name__}")
