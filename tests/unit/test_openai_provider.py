"""Tests for the OpenAI-compatible VLM provider: image attachment, timeout, and retries."""
from __future__ import annotations

import io
import json
from urllib.error import HTTPError

import pytest

from app.core.config import VLMSettings
from app.vlm import openai_compatible_provider as provider_module
from app.vlm.openai_compatible_provider import OpenAICompatibleProvider


def _settings(**overrides) -> VLMSettings:
    defaults = dict(model="test-model", base_url="http://127.0.0.1:8000/v1", api_key="secret", timeout_seconds=12.5, max_retries=1)
    defaults.update(overrides)
    return VLMSettings(**defaults)


def _chat_completion_body(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode()


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info) -> None:
        return None


SUCCESS_CONTENT = json.dumps({
    "scene_summary": "ok", "detected_action": "standing", "step_status": "UNKNOWN",
    "confidence": 0.7, "safety_violation": False, "violations": [], "evidence": [], "uncertainties": [],
})


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    monkeypatch.setattr(provider_module.time, "sleep", lambda seconds: None)


def test_analyze_includes_image_parts_as_data_urls(monkeypatch):
    captured_requests = []

    def fake_urlopen(req, timeout=None):
        captured_requests.append((req, timeout))
        return _FakeResponse(_chat_completion_body(SUCCESS_CONTENT))

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)
    settings = _settings()
    provider = OpenAICompatibleProvider(settings)

    import asyncio
    response = asyncio.run(provider.analyze(["QUJD", "REVG"], {"objects": []}, {}))

    assert response.provider_success is True
    assert len(captured_requests) == 1
    req, timeout = captured_requests[0]
    payload = json.loads(req.data)
    content = payload["messages"][1]["content"]
    assert isinstance(content, list)
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert [part["image_url"]["url"] for part in image_parts] == [
        "data:image/jpeg;base64,QUJD",
        "data:image/jpeg;base64,REVG",
    ]
    text_parts = [part for part in content if part["type"] == "text"]
    assert len(text_parts) == 1
    assert timeout == settings.timeout_seconds


def test_analyze_text_only_content_when_no_images(monkeypatch):
    captured_requests = []

    def fake_urlopen(req, timeout=None):
        captured_requests.append(req)
        return _FakeResponse(_chat_completion_body(SUCCESS_CONTENT))

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(_settings())

    import asyncio
    response = asyncio.run(provider.analyze([], {"objects": []}, {}))

    assert response.provider_success is True
    payload = json.loads(captured_requests[0].data)
    content = payload["messages"][1]["content"]
    assert isinstance(content, list)
    assert all(part["type"] == "text" for part in content)


def test_analyze_uses_configured_timeout(monkeypatch):
    seen_timeouts = []

    def fake_urlopen(req, timeout=None):
        seen_timeouts.append(timeout)
        return _FakeResponse(_chat_completion_body(SUCCESS_CONTENT))

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)
    settings = _settings(timeout_seconds=7.25)
    provider = OpenAICompatibleProvider(settings)

    import asyncio
    asyncio.run(provider.analyze([], {}, {}))

    assert seen_timeouts == [7.25]


def test_analyze_retries_on_500_then_succeeds(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(req, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise HTTPError(req.full_url, 500, "Internal Server Error", {}, io.BytesIO(b"boom"))
        return _FakeResponse(_chat_completion_body(SUCCESS_CONTENT))

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(_settings(max_retries=1))

    import asyncio
    response = asyncio.run(provider.analyze([], {}, {}))

    assert calls["count"] == 2
    assert response.provider_success is True
    assert response.detected_action == "standing"


def test_analyze_does_not_retry_on_400_and_reports_failure(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(req, timeout=None):
        calls["count"] += 1
        raise HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b"bad request detail"))

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(_settings(max_retries=1))

    import asyncio
    response = asyncio.run(provider.analyze([], {}, {}))

    assert calls["count"] == 1
    assert response.provider_success is False
    assert "HTTP 400" in response.uncertainties
