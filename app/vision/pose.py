"""Pluggable pose estimation backends with a shared normalized representation."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from .models import Keypoint, Pose


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: np.ndarray, frame_id: int) -> list[Pose]: ...


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
        result = self._mp.process(frame[:, :, ::-1])
        if not result.pose_landmarks:
            return []
        names = {"nose": 0, "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16}
        landmarks = result.pose_landmarks.landmark
        return [Pose(person_id=1, keypoints={n: Keypoint(x=landmarks[i].x, y=landmarks[i].y, confidence=landmarks[i].visibility) for n, i in names.items()})]


def create_pose_estimator(mode: str, confidence: float) -> PoseEstimator:
    return MockPoseEstimator() if mode == "mock" else MediaPipePoseEstimator(confidence)
