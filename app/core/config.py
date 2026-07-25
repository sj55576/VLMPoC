"""Configuration loading with YAML defaults and environment overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


def _require_choice(value: str, choices: set[str], label: str) -> str:
    """Raise an actionable error naming label/choices when value is not an allowed choice."""
    if value not in choices:
        raise ValueError(f"{label} must be one of {sorted(choices)}, got {value!r}")
    return value


def _require_range(value: float, low: float, high: float, label: str) -> float:
    """Raise an actionable error naming label/bounds when value falls outside [low, high]."""
    if not low <= value <= high:
        raise ValueError(f"{label} must be between {low} and {high}, got {value!r}")
    return value


def _require_positive(value: float, label: str) -> float:
    """Raise an actionable error naming label when value is not strictly positive."""
    if value <= 0:
        raise ValueError(f"{label} must be > 0, got {value!r}")
    return value


class AppSettings(BaseModel):
    name: str = "VLM SOP Monitor"
    mode: str = "mock"
    host: str = "0.0.0.0"
    port: int = 8000

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        return _require_choice(value, {"mock", "full", "vision-only", "vlm-only"}, "app.mode (env var APP_MODE)")


class VisionSettings(BaseModel):
    device: str = "auto"
    detection_model_path: str = ""
    pose_model_path: str = ""
    detection_confidence: float = 0.4
    pose_confidence: float = 0.4
    tracker_iou_threshold: float = 0.25
    frame_queue_size: int = 4
    missing_tolerance_seconds: float = 1.0
    class_aliases: dict[str, str] = Field(default_factory=dict)

    @field_validator("detection_confidence")
    @classmethod
    def _validate_detection_confidence(cls, value: float) -> float:
        return _require_range(value, 0, 1, "vision.detection_confidence (env var DETECTION_CONFIDENCE)")

    @field_validator("pose_confidence")
    @classmethod
    def _validate_pose_confidence(cls, value: float) -> float:
        return _require_range(value, 0, 1, "vision.pose_confidence (env var POSE_CONFIDENCE)")

    @field_validator("tracker_iou_threshold")
    @classmethod
    def _validate_tracker_iou_threshold(cls, value: float) -> float:
        return _require_range(value, 0, 1, "vision.tracker_iou_threshold")


class VLMSettings(BaseModel):
    provider: str = "mock"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    interval_seconds: float = 5.0
    max_images: int = 4
    timeout_seconds: float = 30.0
    max_retries: int = 1
    jpeg_quality: int = 80
    image_max_dim: int = 640
    min_trigger_gap_seconds: float = 2.0
    failure_backoff_seconds: float = 30.0
    # How long a cached VLM result may still satisfy a SOP vlm_confirmation condition.
    max_result_age_seconds: float = 15.0

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        return _require_choice(value, {"mock", "openai_compatible", "local"}, "vlm.provider (env var VLM_PROVIDER)")

    @field_validator("jpeg_quality")
    @classmethod
    def _validate_jpeg_quality(cls, value: int) -> int:
        return int(_require_range(value, 1, 100, "vlm.jpeg_quality (env var VLM_JPEG_QUALITY)"))

    @field_validator("image_max_dim")
    @classmethod
    def _validate_image_max_dim(cls, value: int) -> int:
        return int(_require_positive(value, "vlm.image_max_dim (env var VLM_IMAGE_MAX_DIM)"))


class ActivitySettings(BaseModel):
    enabled: bool = True
    window_seconds: float = 2.0
    min_hold_seconds: float = 0.6


class SOPSettings(BaseModel):
    enabled: bool = True


class SourceSettings(BaseModel):
    """Server-side video ingestion: where frames come from and how fast to pull them."""
    type: str = "mock"
    uri: str = ""
    target_fps: float = 10.0
    queue_size: int = 2
    reconnect_seconds: float = 3.0
    max_reconnect_attempts: int = 0
    loop_file: bool = False

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        return _require_choice(value, {"mock", "camera", "file", "rtsp"}, "source.type (env var SOURCE_TYPE)")

    @field_validator("target_fps")
    @classmethod
    def _validate_target_fps(cls, value: float) -> float:
        return _require_positive(value, "source.target_fps (env var SOURCE_TARGET_FPS)")

    @field_validator("queue_size")
    @classmethod
    def _validate_queue_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"source.queue_size (env var SOURCE_QUEUE_SIZE) must be >= 1, got {value!r}")
        return value


class StorageSettings(BaseModel):
    database_url: str = "sqlite:///./data/app.db"
    save_event_frames: bool = True
    retention_days: int = 30
    # Directory saved event/session frames are written to.
    frame_dir: str = "./data/frames"
    frame_jpeg_quality: int = 80
    frame_max_dim: int = 1280
    # Minutes between periodic retention purges; <= 0 disables the periodic purge.
    retention_interval_minutes: int = 60

    @field_validator("frame_jpeg_quality")
    @classmethod
    def _validate_frame_jpeg_quality(cls, value: int) -> int:
        return int(_require_range(value, 1, 100, "storage.frame_jpeg_quality (env var FRAME_JPEG_QUALITY)"))

    @field_validator("frame_max_dim")
    @classmethod
    def _validate_frame_max_dim(cls, value: int) -> int:
        return int(_require_positive(value, "storage.frame_max_dim (env var FRAME_MAX_DIM)"))


class SecuritySettings(BaseModel):
    max_upload_mb: int = 100
    allowed_video_extensions: list[str] = Field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])
    # Empty disables API key auth entirely.
    api_key: str = ""
    # 0 disables rate limiting.
    rate_limit_per_minute: int = 0


class Settings(BaseModel):
    app: AppSettings = Field(default_factory=AppSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    vlm: VLMSettings = Field(default_factory=VLMSettings)
    activity: ActivitySettings = Field(default_factory=ActivitySettings)
    sop: SOPSettings = Field(default_factory=SOPSettings)
    source: SourceSettings = Field(default_factory=SourceSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_settings(root: Path | None = None) -> Settings:
    """Load config/default.yaml, optional environment config, then safe env overrides."""
    root = root or Path(__file__).resolve().parents[2]
    # Keep explicitly supplied process variables authoritative over values in .env.
    load_dotenv(root / ".env", override=False)
    with (root / "config" / "default.yaml").open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    env = os.getenv("APP_ENV", "development")
    env_file = root / "config" / f"{env}.yaml"
    if env_file.exists():
        with env_file.open(encoding="utf-8") as file:
            _merge(data, yaml.safe_load(file) or {})
    mappings = {
        "APP_MODE": ("app", "mode"), "APP_HOST": ("app", "host"), "APP_PORT": ("app", "port"),
        "VISION_DEVICE": ("vision", "device"), "DETECTION_MODEL_PATH": ("vision", "detection_model_path"),
        "POSE_MODEL_PATH": ("vision", "pose_model_path"), "DETECTION_CONFIDENCE": ("vision", "detection_confidence"),
        "POSE_CONFIDENCE": ("vision", "pose_confidence"), "VLM_PROVIDER": ("vlm", "provider"),
        "VLM_MODEL": ("vlm", "model"), "VLM_BASE_URL": ("vlm", "base_url"), "VLM_API_KEY": ("vlm", "api_key"),
        "VLM_INTERVAL_SECONDS": ("vlm", "interval_seconds"), "VLM_MAX_IMAGES": ("vlm", "max_images"),
        "VLM_TIMEOUT_SECONDS": ("vlm", "timeout_seconds"), "VLM_MAX_RETRIES": ("vlm", "max_retries"),
        "VLM_JPEG_QUALITY": ("vlm", "jpeg_quality"), "VLM_IMAGE_MAX_DIM": ("vlm", "image_max_dim"),
        "VLM_MIN_TRIGGER_GAP_SECONDS": ("vlm", "min_trigger_gap_seconds"),
        "VLM_FAILURE_BACKOFF_SECONDS": ("vlm", "failure_backoff_seconds"),
        "VLM_MAX_RESULT_AGE_SECONDS": ("vlm", "max_result_age_seconds"),
        "SOP_ENABLED": ("sop", "enabled"),
        "ACTIVITY_ENABLED": ("activity", "enabled"),
        "SOURCE_TYPE": ("source", "type"), "SOURCE_URI": ("source", "uri"),
        "SOURCE_TARGET_FPS": ("source", "target_fps"), "SOURCE_QUEUE_SIZE": ("source", "queue_size"),
        "SOURCE_RECONNECT_SECONDS": ("source", "reconnect_seconds"),
        "SOURCE_MAX_RECONNECT_ATTEMPTS": ("source", "max_reconnect_attempts"),
        "SOURCE_LOOP_FILE": ("source", "loop_file"),
        "DATABASE_URL": ("storage", "database_url"), "SAVE_EVENT_FRAMES": ("storage", "save_event_frames"),
        "RETENTION_DAYS": ("storage", "retention_days"),
        "FRAME_STORAGE_DIR": ("storage", "frame_dir"), "FRAME_JPEG_QUALITY": ("storage", "frame_jpeg_quality"),
        "FRAME_MAX_DIM": ("storage", "frame_max_dim"),
        "RETENTION_INTERVAL_MINUTES": ("storage", "retention_interval_minutes"),
        "API_KEY": ("security", "api_key"), "RATE_LIMIT_PER_MINUTE": ("security", "rate_limit_per_minute"),
    }
    for env_key, (section, key) in mappings.items():
        if env_key in os.environ:
            data.setdefault(section, {})[key] = os.environ[env_key]
    return Settings.model_validate(data)
