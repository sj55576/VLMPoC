"""VLM provider interface and factory."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from .schemas import VLMResponse

if TYPE_CHECKING:
    from app.core.config import VLMSettings


class VLMProvider(ABC):
    @abstractmethod
    async def analyze(self, images: list[Any], observation: dict[str, Any], sop_context: dict[str, Any]) -> VLMResponse: ...


def create_vlm_provider(settings: "VLMSettings") -> VLMProvider:
    if settings.provider == "mock":
        from .mock_provider import MockVLMProvider
        return MockVLMProvider()
    if settings.provider == "openai_compatible":
        from .openai_compatible_provider import OpenAICompatibleProvider
        return OpenAICompatibleProvider(settings)
    if settings.provider == "local":
        from .local_provider import LocalTransformersProvider
        return LocalTransformersProvider(settings.model)
    raise ValueError(f"Unknown VLM provider: {settings.provider}")
