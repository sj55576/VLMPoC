"""Pluggable object detection backends."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from .geometry import sanitize_bbox
from .models import Detection


class VisionConfigurationError(ValueError):
    """Raised when a selected vision mode cannot be started safely."""


def _canonical_label(value: str) -> str:
    """Make model labels stable across spaces, hyphens, and capitalization."""
    return "_".join(value.strip().lower().replace("-", " ").split())


class LabelNormalizer:
    """Normalize detector labels to the SOP vocabulary using configurable aliases."""

    def __init__(self, aliases: Mapping[str, str] | None = None) -> None:
        self.aliases = {
            _canonical_label(source): _canonical_label(target)
            for source, target in (aliases or {}).items()
            if source.strip() and target.strip()
        }

    def normalize(self, label: str) -> str:
        canonical = _canonical_label(label)
        return self.aliases.get(canonical, canonical)


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]: ...


class NullDetector(Detector):
    """Vision-free detector used in ``vlm-only`` mode without synthetic SOP evidence."""

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        return []


class MockDetector(Detector):
    """Deterministic synthetic detections that drive the example SOP."""
    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        # Timeline supports a repeatable mock session and is intentionally visible in the demo video.
        height, width = frame.shape[:2]

        def bbox(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
            return values[0] * width / 640, values[1] * height / 480, values[2] * width / 640, values[3] * height / 480

        objects = [Detection(class_name="person", confidence=.98, bbox=bbox((90, 35, 330, 450))),
                   Detection(class_name="helmet", confidence=.95, bbox=bbox((172, 45, 240, 92)))]
        if frame_id >= 8:
            objects.append(Detection(class_name="screwdriver", confidence=.91, bbox=bbox((215, 268, 242, 335))))
        if frame_id >= 14:
            objects.append(Detection(class_name="part_a", confidence=.92, bbox=bbox((245, 285, 335, 370))))
        if frame_id >= 20:
            objects.append(Detection(class_name="screw", confidence=.82, bbox=bbox((280, 315, 291, 326))))
        if frame_id >= 45:
            objects = [d for d in objects if d.class_name != "part_a"] + [Detection(class_name="part_a", confidence=.94, bbox=bbox((500, 350, 585, 425))), *[d for d in objects if d.class_name != "part_a"], Detection(class_name="completed_box", confidence=.90, bbox=bbox((450, 300, 630, 470)))]
        return objects


class UltralyticsDetector(Detector):
    """Optional Ultralytics YOLO adapter; only imported when selected."""
    def __init__(self, model_path: str, confidence: float = .4, device: str = "auto", label_aliases: Mapping[str, str] | None = None) -> None:
        if not model_path:
            raise VisionConfigurationError("DETECTION_MODEL_PATH is required when APP_MODE is 'full' or 'vision-only'. Set APP_MODE=mock for a no-model demo.")
        if not Path(model_path).is_file():
            raise VisionConfigurationError(f"Detection model was not found: {model_path}. Check DETECTION_MODEL_PATH or use APP_MODE=mock.")
        if not 0.0 <= confidence <= 1.0:
            raise VisionConfigurationError("DETECTION_CONFIDENCE must be between 0 and 1.")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install optional dependency: pip install '.[vision]'") from exc
        self.model: Any = YOLO(model_path)
        self.confidence = confidence
        self.labels = LabelNormalizer(label_aliases)
        if device == "auto":
            try:
                import torch
                self.device = 0 if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError("Detector expects an HxWx3 or HxWx4 image frame.")
        result = self.model(frame, conf=self.confidence, device=self.device, verbose=False)[0]
        names = result.names or {}
        detections: list[Detection] = []
        for box in result.boxes or []:
            raw_bbox = tuple(map(float, box.xyxy[0].tolist()))
            bbox = sanitize_bbox(raw_bbox)
            score = float(box.conf[0])
            if bbox is None or not np.isfinite(score) or score < self.confidence:
                continue
            class_id = int(box.cls[0])
            raw_label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            detections.append(Detection(class_name=self.labels.normalize(str(raw_label)), confidence=score, bbox=bbox))
        return detections


def create_detector(mode: str, model_path: str, confidence: float, device: str, label_aliases: Mapping[str, str] | None = None) -> Detector:
    """Build a detector with explicit no-model behavior for each runtime mode."""
    if mode == "mock":
        return MockDetector()
    if mode == "vlm-only":
        return NullDetector()
    if mode not in {"full", "vision-only"}:
        raise VisionConfigurationError("APP_MODE must be one of: full, vision-only, vlm-only, mock.")
    return UltralyticsDetector(model_path, confidence, device, label_aliases)
