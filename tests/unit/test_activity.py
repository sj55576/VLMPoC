from datetime import UTC, datetime, timedelta

from app.activity import ActivityEstimator
from app.vision.models import Detection, Keypoint, Observation, Pose

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _standing_keypoints(x: float = 0.30) -> dict[str, Keypoint]:
    return {
        "nose": Keypoint(x=x + .02, y=.14, confidence=.96),
        "left_shoulder": Keypoint(x=x, y=.30, confidence=.9),
        "right_shoulder": Keypoint(x=x + .06, y=.30, confidence=.9),
        "left_hip": Keypoint(x=x - .01, y=.55, confidence=.88),
        "right_hip": Keypoint(x=x + .06, y=.55, confidence=.88),
        "left_knee": Keypoint(x=x - .01, y=.75, confidence=.85),
        "right_knee": Keypoint(x=x + .06, y=.75, confidence=.85),
        "left_ankle": Keypoint(x=x - .01, y=.95, confidence=.85),
        "right_ankle": Keypoint(x=x + .06, y=.95, confidence=.85),
        "left_wrist": Keypoint(x=x - .04, y=.58, confidence=.9),
        "right_wrist": Keypoint(x=x + .10, y=.58, confidence=.9),
        "left_elbow": Keypoint(x=x - .02, y=.46, confidence=.89),
        "right_elbow": Keypoint(x=x + .08, y=.46, confidence=.89),
    }


def _sitting_keypoints(x: float = 0.30) -> dict[str, Keypoint]:
    kp = _standing_keypoints(x)
    for name in ("left_hip", "right_hip"):
        kp[name] = Keypoint(x=kp[name].x, y=.70, confidence=.88)
    for name in ("left_knee", "right_knee"):
        kp[name] = Keypoint(x=kp[name].x, y=.72, confidence=.85)
    for name in ("left_ankle", "right_ankle"):
        kp[name] = Keypoint(x=kp[name].x, y=.90, confidence=.85)
    return kp


def _lying_keypoints() -> dict[str, Keypoint]:
    return {
        "left_shoulder": Keypoint(x=.20, y=.50, confidence=.9),
        "right_shoulder": Keypoint(x=.60, y=.50, confidence=.9),
        "left_hip": Keypoint(x=.65, y=.52, confidence=.88),
        "right_hip": Keypoint(x=.68, y=.52, confidence=.88),
    }


def _saturate(estimator: ActivityEstimator, keypoints_fn, seconds: float, step: float = 0.2, objects=None, start=BASE) -> tuple:
    """Feed enough updates to fill the window and satisfy hysteresis; return the final estimate."""
    now = start
    estimate = None
    elapsed = 0.0
    while elapsed <= seconds:
        obs = Observation(timestamp=now, frame_id=0, width=640, height=480, poses=[Pose(person_id=1, keypoints=keypoints_fn())], objects=objects or [])
        estimate = estimator.update(obs, now)
        now += timedelta(seconds=step)
        elapsed += step
    return estimate, now


def test_no_pose_returns_none():
    estimator = ActivityEstimator()
    obs = Observation(timestamp=BASE, frame_id=0, width=640, height=480, poses=[])
    assert estimator.update(obs, BASE) is None


def test_standing_detection():
    estimator = ActivityEstimator(window_seconds=1.0, min_hold_seconds=0.3)
    estimate, _ = _saturate(estimator, _standing_keypoints, seconds=1.0)
    assert estimate.label == "standing"


def test_sitting_detection():
    estimator = ActivityEstimator(window_seconds=1.0, min_hold_seconds=0.3)
    estimate, _ = _saturate(estimator, _sitting_keypoints, seconds=1.0)
    assert estimate.label == "sitting"


def test_lying_detection():
    estimator = ActivityEstimator(window_seconds=1.0, min_hold_seconds=0.3)
    estimate, _ = _saturate(estimator, _lying_keypoints, seconds=1.0)
    assert estimate.label == "lying"


def test_walking_via_moving_hips():
    estimator = ActivityEstimator(window_seconds=1.0, min_hold_seconds=0.3)
    # Warm up standing first so posture baseline is standing.
    _, now = _saturate(estimator, _standing_keypoints, seconds=1.0)
    x = 0.30
    for _ in range(8):
        x += 0.02
        obs = Observation(timestamp=now, frame_id=0, width=640, height=480, poses=[Pose(person_id=1, keypoints=_standing_keypoints(x))])
        estimate = estimator.update(obs, now)
        now += timedelta(seconds=0.2)
    assert estimate.label == "walking"


def test_drinking_via_cup_near_raised_wrist():
    estimator = ActivityEstimator(window_seconds=1.0, min_hold_seconds=0.3)

    def keypoints():
        kp = _standing_keypoints()
        kp["right_wrist"] = Keypoint(x=.40, y=.30, confidence=.9)
        return kp

    cup = Detection(class_name="cup", confidence=.9, bbox=(.40 * 640, .30 * 480, .44 * 640, .34 * 480))
    estimate, _ = _saturate(estimator, keypoints, seconds=1.0, objects=[cup])
    assert estimate.label == "drinking"


def test_phone_use_near_wrist():
    estimator = ActivityEstimator(window_seconds=1.0, min_hold_seconds=0.3)

    def keypoints():
        kp = _standing_keypoints()
        kp["right_wrist"] = Keypoint(x=.40, y=.40, confidence=.9)
        return kp

    phone = Detection(class_name="cell_phone", confidence=.9, bbox=(.40 * 640, .40 * 480, .43 * 640, .43 * 480))
    estimate, _ = _saturate(estimator, keypoints, seconds=1.0, objects=[phone])
    assert estimate.label == "phone_use"


def test_hysteresis_ignores_single_frame_flicker():
    estimator = ActivityEstimator(window_seconds=2.0, min_hold_seconds=1.0)
    estimate, now = _saturate(estimator, _standing_keypoints, seconds=2.0)
    assert estimate.label == "standing"
    # A single flicker to sitting shouldn't be enough to replace the emitted label immediately.
    obs = Observation(timestamp=now, frame_id=0, width=640, height=480, poses=[Pose(person_id=1, keypoints=_sitting_keypoints())])
    flickered = estimator.update(obs, now)
    assert flickered.label == "standing"
