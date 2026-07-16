"""OpenAI-compatible JSON-schema API adapter using the standard library."""
from __future__ import annotations
import asyncio, json
from typing import Any
from urllib.request import Request, urlopen
from .base import VLMProvider
from .prompts import JSON_SCHEMA, SYSTEM_PROMPT
from .schemas import VLMResponse, parse_vlm_response


class OpenAICompatibleProvider(VLMProvider):
    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        if not (model and base_url): raise ValueError("VLM_MODEL and VLM_BASE_URL are required")
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key

    async def analyze(self, images: list[Any], observation: dict[str, Any], sop_context: dict[str, Any]) -> VLMResponse:
        payload = {"model": self.model, "messages": [{"role":"system", "content":SYSTEM_PROMPT}, {"role":"user", "content":json.dumps({"observation": observation, "sop_context": sop_context}, ensure_ascii=False)}], "response_format": {"type":"json_schema", "json_schema":{"name":"sop_scene", "strict":True, "schema":JSON_SCHEMA}}}
        def request() -> str:
            headers = {"Content-Type":"application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            raw = json.dumps(payload).encode(); req = Request(f"{self.base_url}/chat/completions", raw, headers)
            with urlopen(req, timeout=30) as response: return json.loads(response.read())["choices"][0]["message"]["content"]
        try: return parse_vlm_response(await asyncio.to_thread(request))
        except Exception as exc: return parse_vlm_response({"scene_summary":"VLM request failed", "step_status":"UNKNOWN", "confidence":0, "safety_violation":False, "violations":[], "evidence":[], "uncertainties":[exc.__class__.__name__]})
