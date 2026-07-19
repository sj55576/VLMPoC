"""OpenAI-compatible JSON-schema API adapter using the standard library."""
from __future__ import annotations
import asyncio, json, time
from typing import Any, TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .base import VLMProvider
from .prompts import DAILY_ACTIVITY_SYSTEM_PROMPT, JSON_SCHEMA, SYSTEM_PROMPT
from .schemas import VLMResponse, parse_vlm_response

if TYPE_CHECKING:
    from app.core.config import VLMSettings

_RETRY_BACKOFF_SECONDS = (1, 2)


class OpenAICompatibleProvider(VLMProvider):
    def __init__(self, settings: "VLMSettings") -> None:
        if not (settings.model and settings.base_url): raise ValueError("VLM_MODEL and VLM_BASE_URL are required")
        self.model, self.base_url, self.api_key = settings.model, settings.base_url.rstrip("/"), settings.api_key
        self.timeout_seconds = settings.timeout_seconds
        self.max_retries = settings.max_retries

    async def analyze(self, images: list[Any], observation: dict[str, Any], sop_context: dict[str, Any]) -> VLMResponse:
        system_prompt = SYSTEM_PROMPT if sop_context.get("current_step") else DAILY_ACTIVITY_SYSTEM_PROMPT
        text_part = {"type": "text", "text": json.dumps({"observation": observation, "sop_context": sop_context}, ensure_ascii=False)}
        image_parts = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}} for image in images]
        user_content: Any = [text_part, *image_parts]
        payload = {"model": self.model, "messages": [{"role":"system", "content":system_prompt}, {"role":"user", "content":user_content}]}
        if sop_context.get("current_step"):
            payload["response_format"] = {"type":"json_schema", "json_schema":{"name":"sop_scene", "strict":True, "schema":JSON_SCHEMA}}

        def request() -> str:
            headers = {"Content-Type":"application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            raw = json.dumps(payload).encode()
            attempt = 0
            while True:
                try:
                    req = Request(f"{self.base_url}/chat/completions", raw, headers)
                    with urlopen(req, timeout=self.timeout_seconds) as response:
                        return json.loads(response.read())["choices"][0]["message"]["content"]
                except HTTPError as exc:
                    if attempt < self.max_retries and (exc.code == 429 or exc.code >= 500):
                        time.sleep(_RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)])
                        attempt += 1
                        continue
                    raise
                except (URLError, OSError):
                    if attempt < self.max_retries:
                        time.sleep(_RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)])
                        attempt += 1
                        continue
                    raise

        try:
            return parse_vlm_response(await asyncio.to_thread(request))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip().replace("\n", " ")[:300]
            return VLMResponse(scene_summary="VLM request failed", confidence=0, uncertainties=[f"HTTP {exc.code}"], provider_success=False, error_message=detail or exc.reason)
        except Exception as exc:
            return VLMResponse(scene_summary="VLM request failed", confidence=0, uncertainties=[exc.__class__.__name__], provider_success=False, error_message=str(exc)[:300])
