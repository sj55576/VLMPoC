"""Configuration loading with YAML defaults and environment overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    name: str = "VLM SOP Monitor"
    mode: str = "mock"
    host: str = "0.0.0.0"
    port: int = 8000


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


class ActivitySettings(BaseModel):
    enabled: bool = True
    window_seconds: float = 2.0
    min_hold_seconds: float = 0.6


class SOPSettings(BaseModel):
    enabled: bool = True


class StorageSettings(BaseModel):
    database_url: str = "sqlite:///./data/app.db"
    save_event_frames: bool = True
    save_all_frames: bool = False
    retention_days: int = 30


class SecuritySettings(BaseModel):
    max_upload_mb: int = 100
    allowed_video_extensions: list[str] = Field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])


class Settings(BaseModel):
    app: AppSettings = Field(default_factory=AppSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    vlm: VLMSettings = Field(default_factory=VLMSettings)
    activity: ActivitySettings = Field(default_factory=ActivitySettings)
    sop: SOPSettings = Field(default_factory=SOPSettings)
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
        "SOP_ENABLED": ("sop", "enabled"),
        "DATABASE_URL": ("storage", "database_url"), "SAVE_EVENT_FRAMES": ("storage", "save_event_frames"),
    }
    for env_key, (section, key) in mappings.items():
        if env_key in os.environ:
            data.setdefault(section, {})[key] = os.environ[env_key]
    return Settings.model_validate(data)
