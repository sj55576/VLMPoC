"""Tests for config loading, new settings fields, precedence, and validation."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings, load_settings

ROOT = Path(__file__).resolve().parents[2]


def test_defaults_present_on_settings() -> None:
    settings = Settings()

    assert settings.source.type == "mock"
    assert settings.source.uri == ""
    assert settings.source.target_fps == 10.0
    assert settings.source.queue_size == 2
    assert settings.source.reconnect_seconds == 3.0
    assert settings.source.max_reconnect_attempts == 0
    assert settings.source.loop_file is False

    assert settings.vlm.max_result_age_seconds == 15.0

    assert settings.storage.frame_dir == "./data/frames"
    assert settings.storage.frame_jpeg_quality == 80
    assert settings.storage.frame_max_dim == 1280
    assert settings.storage.retention_interval_minutes == 60

    assert settings.security.api_key == ""
    assert settings.security.rate_limit_per_minute == 0


def test_load_settings_reads_defaults_from_disk(monkeypatch) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    # conftest's autouse fixture overrides FRAME_STORAGE_DIR for test isolation; unset it
    # here so we can observe the real config/default.yaml value.
    monkeypatch.delenv("FRAME_STORAGE_DIR", raising=False)
    settings = load_settings(ROOT)

    assert settings.source.type == "mock"
    assert settings.storage.frame_dir == "./data/frames"
    assert settings.security.rate_limit_per_minute == 0


def test_env_overrides_yaml_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SOURCE_TYPE", "camera")
    monkeypatch.setenv("SOURCE_URI", "0")
    monkeypatch.setenv("SOURCE_TARGET_FPS", "15")
    monkeypatch.setenv("SOURCE_QUEUE_SIZE", "5")
    monkeypatch.setenv("SOURCE_RECONNECT_SECONDS", "1.5")
    monkeypatch.setenv("SOURCE_MAX_RECONNECT_ATTEMPTS", "4")
    monkeypatch.setenv("SOURCE_LOOP_FILE", "true")
    monkeypatch.setenv("VLM_MAX_RESULT_AGE_SECONDS", "45")
    monkeypatch.setenv("FRAME_STORAGE_DIR", "./custom-frames")
    monkeypatch.setenv("FRAME_JPEG_QUALITY", "60")
    monkeypatch.setenv("FRAME_MAX_DIM", "800")
    monkeypatch.setenv("RETENTION_INTERVAL_MINUTES", "30")
    monkeypatch.setenv("API_KEY", "s3cr3t")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "120")
    monkeypatch.setenv("SAVE_ALL_FRAMES", "true")
    monkeypatch.setenv("ACTIVITY_ENABLED", "false")

    settings = load_settings(ROOT)

    assert settings.source.type == "camera"
    assert settings.source.uri == "0"
    assert settings.source.target_fps == 15.0
    assert settings.source.queue_size == 5
    assert settings.source.reconnect_seconds == 1.5
    assert settings.source.max_reconnect_attempts == 4
    assert settings.source.loop_file is True
    assert settings.vlm.max_result_age_seconds == 45.0
    assert settings.storage.frame_dir == "./custom-frames"
    assert settings.storage.frame_jpeg_quality == 60
    assert settings.storage.frame_max_dim == 800
    assert settings.storage.retention_interval_minutes == 30
    assert settings.security.api_key == "s3cr3t"
    assert settings.security.rate_limit_per_minute == 120
    assert settings.storage.save_all_frames is True
    assert settings.activity.enabled is False


def test_sop_enabled_false_env_coerces_to_bool(monkeypatch) -> None:
    """SOP_ENABLED=false (a string from the environment) must yield sop.enabled is False."""
    monkeypatch.setenv("SOP_ENABLED", "false")

    settings = load_settings(ROOT)

    assert settings.sop.enabled is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "bogus"),
    ],
)
def test_app_mode_validator_rejects_bad_value(field, value) -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate({"app": {field: value}})
    message = str(exc_info.value)
    assert "app.mode" in message
    assert "APP_MODE" in message
    assert "bogus" in message


def test_vlm_provider_validator_rejects_bad_value() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate({"vlm": {"provider": "not-a-provider"}})
    message = str(exc_info.value)
    assert "vlm.provider" in message
    assert "VLM_PROVIDER" in message


def test_source_type_validator_rejects_bad_value() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate({"source": {"type": "webcam"}})
    message = str(exc_info.value)
    assert "source.type" in message
    assert "SOURCE_TYPE" in message


@pytest.mark.parametrize("field", ["detection_confidence", "pose_confidence", "tracker_iou_threshold"])
def test_vision_confidence_fields_reject_out_of_range(field) -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate({"vision": {field: 1.5}})
    assert f"vision.{field}" in str(exc_info.value)

    with pytest.raises(ValidationError):
        Settings.model_validate({"vision": {field: -0.1}})


def test_vlm_jpeg_quality_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate({"vlm": {"jpeg_quality": 0}})
    assert "vlm.jpeg_quality" in str(exc_info.value)

    with pytest.raises(ValidationError):
        Settings.model_validate({"vlm": {"jpeg_quality": 101}})


def test_storage_frame_jpeg_quality_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate({"storage": {"frame_jpeg_quality": 200}})
    assert "storage.frame_jpeg_quality" in str(exc_info.value)


def test_vlm_image_max_dim_must_be_positive() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate({"vlm": {"image_max_dim": 0}})
    assert "vlm.image_max_dim" in str(exc_info.value)


def test_storage_frame_max_dim_must_be_positive() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate({"storage": {"frame_max_dim": -5}})
    assert "storage.frame_max_dim" in str(exc_info.value)


def test_source_target_fps_must_be_positive() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate({"source": {"target_fps": 0}})
    assert "source.target_fps" in str(exc_info.value)


def test_source_queue_size_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate({"source": {"queue_size": 0}})
    assert "source.queue_size" in str(exc_info.value)


def test_valid_config_round_trips() -> None:
    settings = Settings.model_validate(
        {
            "app": {"mode": "full"},
            "vlm": {"provider": "openai_compatible"},
            "source": {"type": "rtsp", "target_fps": 5, "queue_size": 1},
            "vision": {"detection_confidence": 0, "pose_confidence": 1, "tracker_iou_threshold": 0.5},
        }
    )
    assert settings.app.mode == "full"
    assert settings.vlm.provider == "openai_compatible"
    assert settings.source.type == "rtsp"
