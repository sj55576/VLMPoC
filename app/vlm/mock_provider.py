"""Deterministic VLM that validates orchestration without network/model downloads."""
from typing import Any
from .base import VLMProvider
from .schemas import VLMResponse, VLMEvidence


class MockVLMProvider(VLMProvider):
    async def analyze(self, images: list[Any], observation: dict[str, Any], sop_context: dict[str, Any]) -> VLMResponse:
        names = {item["class_name"] for item in observation.get("objects", [])}
        current = sop_context.get("current_step", {}).get("id")
        if {"screwdriver", "part_a"} <= names:
            return VLMResponse(scene_summary="作業者がドライバーを部品Aの近くで使用している", detected_action="tightening", current_step_id=current, step_status="IN_PROGRESS", confidence=.86, evidence=[VLMEvidence(type="structured", description="screwdriver and part_a detected")], uncertainties=["回転は時系列からのみ確定可能"])
        return VLMResponse(scene_summary="確認可能な締結動作はありません", detected_action="UNKNOWN", current_step_id=current, step_status="UNKNOWN", confidence=.45, uncertainties=["モックVLMは物体リストのみを利用"])
