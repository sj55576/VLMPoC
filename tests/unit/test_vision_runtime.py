from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.core.config import Settings
from app.vision.detector import (
    LabelNormalizer,
    MockDetector,
    NullDetector,
    VisionConfigurationError,
    create_detector,
)
from app.vision.models import Detection
from app.vision.pose import MockPoseEstimator, NullPoseEstimator, create_pose_estimator
from app.vision.tracker import IoUTracker


def test_label_normalizer_uses_canonical_aliases() -> None:
    normalizer = LabelNormalizer({"Hard Hat": "helmet", "Phillips-driver": "screwdriver"})

    assert normalizer.normalize(" hard-hat ") == "helmet"
    assert normalizer.normalize("PHILLIPS DRIVER") == "screwdriver"
    assert normalizer.normalize("Part A") == "part_a"


def test_class_aliases_are_configuration_backed() -> None:
    settings = Settings.model_validate({"vision": {"class_aliases": {"Hard Hat": "helmet"}}})

    assert settings.vision.class_aliases == {"Hard Hat": "helmet"}


def test_mock_mode_has_synthetic_vision() -> None:
    assert isinstance(create_detector("mock", "", 0.4, "cpu"), MockDetector)
    assert isinstance(create_pose_estimator("mock", 0.4), MockPoseEstimator)


def test_vlm_only_mode_has_no_synthetic_sop_evidence() -> None:
    assert isinstance(create_detector("vlm-only", "", 0.4, "cpu"), NullDetector)
    assert isinstance(create_pose_estimator("vlm-only", 0.4), NullPoseEstimator)


@pytest.mark.parametrize("mode", ["full", "vision-only"])
def test_real_vision_modes_explain_missing_detector_model(mode: str) -> None:
    with pytest.raises(VisionConfigurationError, match="DETECTION_MODEL_PATH"):
        create_detector(mode, "", 0.4, "cpu")


def test_mock_boxes_follow_frame_size() -> None:
    objects = MockDetector().detect(np.zeros((240, 320, 3), dtype=np.uint8), frame_id=0)
    person = next(item for item in objects if item.class_name == "person")

    assert person.bbox == (45.0, 17.5, 165.0, 225.0)


def test_tracker_preserves_identity_and_reports_motion() -> None:
    tracker = IoUTracker(threshold=0.2, max_age_seconds=2.0, history_size=2)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    first = tracker.update([Detection(class_name="part_a", confidence=0.9, bbox=(0, 0, 10, 10))], start)
    second = tracker.update([Detection(class_name="part_a", confidence=0.9, bbox=(2, 0, 12, 10))], start + timedelta(seconds=1))

    assert first[0].track_id == second[0].track_id
    track = tracker.tracks[first[0].track_id]
    assert track.velocity == (2.0, 0.0)
    assert track.direction == (1.0, 0.0)
    assert track.history == [(0.0, 0.0, 10.0, 10.0)]


def test_tracker_rejects_invalid_boxes_and_expires_missing_tracks() -> None:
    tracker = IoUTracker(max_age_seconds=1.0)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    assert tracker.update([Detection(class_name="part_a", confidence=0.9, bbox=(1, 1, 1, 2))], start) == []
    tracker.update([Detection(class_name="part_a", confidence=0.9, bbox=(0, 0, 10, 10))], start)
    tracker.update([], start + timedelta(seconds=0.5))
    assert next(iter(tracker.tracks.values())).missing_seconds == 0.5
    tracker.update([], start + timedelta(seconds=1.1))
    assert tracker.tracks == {}
