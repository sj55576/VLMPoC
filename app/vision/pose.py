"""Pluggable pose estimation backends with a shared normalized representation."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from .detector import VisionConfigurationError
from .models import Keypoint, Pose


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: np.ndarray, frame_id: int) -> list[Pose]: ...


class NullPoseEstimator(PoseEstimator):
    """Vision-free pose backend used in ``vlm-only`` mode."""

    def estimate(self, frame: np.ndarray, frame_id: int) -> list[Pose]:
        return []


class MockPoseEstimator(PoseEstimator):
    def estimate(self, frame: np.ndarray, frame_id: int) -> list[Pose]:
        # Coordinates are image-normalized, independent of source resolution.
        wrist_x = .36 if frame_id < 8 else .355
        return [Pose(person_id=1, keypoints={
            "nose": Keypoint(x=.32, y=.14, confidence=.96),
            "left_wrist": Keypoint(x=wrist_x, y=.58, confidence=.92),
            "right_wrist": Keypoint(x=.36, y=.58, confidence=.92),
            "left_elbow": Keypoint(x=.30, y=.46, confidence=.89),
            "right_elbow": Keypoint(x=.40, y=.46, confidence=.89),
        })]


class MediaPipePoseEstimator(PoseEstimator):
    """Optional MediaPipe adapter. It returns only stable keypoints needed by the rule engine."""
    def __init__(self, confidence: float = .4) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("Install optional dependency: pip install '.[vision]'") from exc
        self._mp = mp.solutions.pose.Pose(min_detection_confidence=confidence, min_tracking_confidence=confidence)

    def estimate(self, frame: np.ndarray, frame_id: int) -> list[Pose]:
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError("Pose estimator expects an HxWx3 or HxWx4 image frame.")
        result = self._mp.process(frame[:, :, ::-1])
        if not result.pose_landmarks:
            return []
        names = {"nose": 0, "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16}
        landmarks = result.pose_landmarks.landmark
        return [Pose(person_id=1, keypoints={n: Keypoint(x=landmarks[i].x, y=landmarks[i].y, confidence=landmarks[i].visibility) for n, i in names.items()})]


def create_pose_estimator(mode: str, confidence: float, model_path: str = "") -> PoseEstimator:
    """Build the default MediaPipe estimator or deterministic no-model fallback.

    MediaPipe ships its own task assets; ``POSE_MODEL_PATH`` is reserved for a
    future ONNX/YOLO adapter and is rejected here to avoid silently ignoring it.
    """
    if mode == "mock":
        return MockPoseEstimator()
    if mode == "vlm-only":
        return NullPoseEstimator()
    if mode not in {"full", "vision-only"}:
        raise VisionConfigurationError("APP_MODE must be one of: full, vision-only, vlm-only, mock.")
    if model_path:
        path = Path(model_path)
        if not path.is_file():
            raise VisionConfigurationError(f"Pose model was not found: {model_path}. Remove POSE_MODEL_PATH to use MediaPipe, or provide a supported adapter.")
        raise VisionConfigurationError("POSE_MODEL_PATH is set, but this build uses MediaPipe Pose. Remove it or install a compatible pose adapter.")
    if not 0.0 <= confidence <= 1.0:
        raise VisionConfigurationError("POSE_CONFIDENCE must be between 0 and 1.")
    return MediaPipePoseEstimator(confidence)
