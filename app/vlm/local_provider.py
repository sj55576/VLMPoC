"""Lazy Transformers-compatible local VLM adapter placeholder."""
from typing import Any

from .base import VLMProvider
from .schemas import VLMResponse, unknown_response


class LocalTransformersProvider(VLMProvider):
    def __init__(self, model: str) -> None:
        self.model = model
        if not model: raise ValueError("VLM_MODEL is required for provider=local")

    async def analyze(self, images: list[Any], observation: dict[str, Any], sop_context: dict[str, Any]) -> VLMResponse:
        # Integration is intentionally model-agnostic: model-specific chat templates belong in a deployment adapter.
        try:
            import transformers  # noqa: F401
        except ImportError:
            return unknown_response("Install optional local VLM dependencies: pip install '.[local-vlm]'")
        return unknown_response("Local VLM adapter requires a model-specific chat template configuration")
