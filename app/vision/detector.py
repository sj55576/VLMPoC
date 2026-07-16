"""Pluggable object detection backends."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from .models import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]: ...


class MockDetector(Detector):
    """Deterministic synthetic detections that drive the example SOP."""
    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        # Timeline supports a repeatable mock session and is intentionally visible in the demo video.
        objects = [Detection(class_name="person", confidence=.98, bbox=(90, 35, 330, 450)),
                   Detection(class_name="helmet", confidence=.95, bbox=(172, 45, 240, 92))]
        if frame_id >= 8:
            objects.append(Detection(class_name="screwdriver", confidence=.91, bbox=(215, 268, 242, 335)))
        if frame_id >= 14:
            objects.append(Detection(class_name="part_a", confidence=.92, bbox=(245, 285, 335, 370)))
        if frame_id >= 20:
            objects.append(Detection(class_name="screw", confidence=.82, bbox=(280, 315, 291, 326)))
        if frame_id >= 45:
            objects = [d for d in objects if d.class_name != "part_a"] + [Detection(class_name="part_a", confidence=.94, bbox=(500, 350, 585, 425)), *[d for d in objects if d.class_name != "part_a"], Detection(class_name="completed_box", confidence=.90, bbox=(450, 300, 630, 470))]
        return objects


class UltralyticsDetector(Detector):
    """Optional Ultralytics YOLO adapter; only imported when selected."""
    def __init__(self, model_path: str, confidence: float = .4, device: str = "auto") -> None:
        if not model_path:
            raise ValueError("DETECTION_MODEL_PATH is required for a YOLO detector")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install optional dependency: pip install '.[vision]'") from exc
        self.model: Any = YOLO(model_path)
        self.confidence, self.device = confidence, device

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        result = self.model(frame, conf=self.confidence, device=self.device, verbose=False)[0]
        names = result.names
        return [Detection(class_name=str(names[int(box.cls[0])]), confidence=float(box.conf[0]), bbox=tuple(map(float, box.xyxy[0].tolist()))) for box in result.boxes]


def create_detector(mode: str, model_path: str, confidence: float, device: str) -> Detector:
    return MockDetector() if mode == "mock" or not model_path else UltralyticsDetector(model_path, confidence, device)
