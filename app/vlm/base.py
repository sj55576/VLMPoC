"""VLM provider interface and factory."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from .schemas import VLMResponse


class VLMProvider(ABC):
    @abstractmethod
    async def analyze(self, images: list[Any], observation: dict[str, Any], sop_context: dict[str, Any]) -> VLMResponse: ...


def create_vlm_provider(provider: str, model: str, base_url: str, api_key: str) -> VLMProvider:
    if provider == "mock":
        from .mock_provider import MockVLMProvider
        return MockVLMProvider()
    if provider == "openai_compatible":
        from .openai_compatible_provider import OpenAICompatibleProvider
        return OpenAICompatibleProvider(model, base_url, api_key)
    if provider == "local":
        from .local_provider import LocalTransformersProvider
        return LocalTransformersProvider(model)
    raise ValueError(f"Unknown VLM provider: {provider}")
